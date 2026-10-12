import json
from pathlib import Path

import numpy as np
import pytest
import torch

from birdcls.config import REPO_ROOT, TrainConfig
from birdcls.fileio import read_jsonl
from birdcls.train import train

DATA = REPO_ROOT / "data" / "cub"
# The tests train on a handful of real images, so they are skipped when the dataset has not been unpacked.
pytestmark = pytest.mark.skipif(not (DATA / "train.txt").exists(), reason="dataset not unpacked")


def tiny_config(**overrides) -> TrainConfig:
    """Build a config small enough to train on CPU in seconds, with ``overrides`` applied."""
    base = dict(run_name="smoke", backbone="resnet18", pretrained=False, img_size=64, epochs=2,
                warmup_epochs=0.5, batch_size=4, accum_steps=2, num_workers=0, limit_train=16,
                limit_eval=8, amp="none", channels_last=False, checkpoint_minutes=0.0,
                prune_on_finish=False, layer_decay=0.8, drop_path=0.0)
    base.update(overrides)
    return TrainConfig(**base)


def test_train_finishes_then_resumes_from_checkpoint(tmp_path, monkeypatch):
    # Train to the end
    monkeypatch.setenv("BIRD_RUN_ROOT", str(tmp_path))
    cfg = tiny_config()
    train(cfg, "cpu")
    run = tmp_path / "smoke"

    # Check the saved files and predictions
    for name in ("DONE", "TRAINED", "best_ema.pt", "last.pt", "predictions.npz", "summary.json"):
        assert (run / name).exists(), name
    preds = np.load(run / "predictions.npz")
    assert preds["test_center"].shape == (8, 200) and preds["val_center"].shape == (8, 200)

    # Every epoch scores val and a training subset with both the EMA and raw weights.
    records = read_jsonl(run / "metrics.jsonl")
    assert set(records[0]) == set(records[1])  # every metric and event is logged every epoch
    for key in ("train_loss", "grad_norm_mean", "clipped_frac", "data_wait_s", "eval_time_s", "dist_from_init",
                "ema_head_dist_from_init", "best_ema_val_top1", "best_epoch", "elapsed_h"):
        assert key in records[0], key
    assert records[0]["new_best"] is True and records[0]["lr_cut"] is False and records[1]["early_stop"] is False
    for net in ("ema_", ""):
        for split in ("val", "train_eval"):
            for metric in ("top1", "mean_class_acc", "loss", "loss_smoothed", "ece", "top1_hflip"):
                assert all(f"{net}{split}_{metric}" in r for r in records), f"{net}{split}_{metric}"

    # Check the step log, the per-epoch logits and the environment record
    steps = read_jsonl(run / "steps.jsonl")
    assert [s["update"] for s in steps] == list(range(1, records[-1]["updates"] + 1))
    epoch_logits = np.load(run / "epochs" / "epoch_001.npz")
    assert set(epoch_logits.files) == {f"{net}{split}_{view}" for net in ("ema_", "") for split in ("val", "train_eval")
                                       for view in ("center", "hflip")}
    assert epoch_logits["ema_val_center"].shape == (8, 200) and epoch_logits["ema_val_center"].dtype == np.float16
    index = np.load(run / "epochs" / "index.npz")
    assert len(index["val_labels"]) == len(index["train_eval_files"]) == 8
    assert json.loads((run / "env.json").read_text())["commit"]

    # Finished runs are skipped on re-launch.
    train(cfg, "cpu")
    assert len((run / "metrics.jsonl").read_text().splitlines()) == 2

    # Simulate a disconnect after epoch 0: only the epoch-0 checkpoint survives.
    for name in ("DONE", "TRAINED", "predictions.npz", "summary.json"):
        (run / name).unlink()
    train(cfg, "cpu")
    epochs = [r["epoch"] for r in read_jsonl(run / "metrics.jsonl")]
    assert epochs == [0, 1, 1]  # resumed at epoch 1 instead of starting over
    assert (run / "DONE").exists()


def test_time_budget_pauses_with_a_checkpoint_then_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("BIRD_RUN_ROOT", str(tmp_path))
    run = tmp_path / "smoke"

    # A budget shorter than one epoch pauses right after epoch 0, even though the periodic checkpoint is not due yet.
    train(tiny_config(checkpoint_minutes=60.0, time_budget_hours=1e-9), "cpu")
    assert (run / "last.pt").exists() and not (run / "TRAINED").exists()

    # The budget may differ between sessions without the run being treated as a new config.
    train(tiny_config(checkpoint_minutes=60.0), "cpu")
    epochs = [r["epoch"] for r in read_jsonl(run / "metrics.jsonl")]
    assert epochs == [0, 1] and (run / "DONE").exists()


def test_diagnostics_can_be_switched_off_and_on_between_sessions(tmp_path, monkeypatch):
    # Train one epoch with the diagnostics off
    monkeypatch.setenv("BIRD_RUN_ROOT", str(tmp_path))
    run = tmp_path / "smoke"
    quiet = dict(train_eval_images=0, eval_raw_every_epoch=False, save_epoch_logits=False)
    train(tiny_config(checkpoint_minutes=60.0, time_budget_hours=1e-9, **quiet), "cpu")

    # Only the EMA weights on val are scored, and no logits are kept.
    first = json.loads((run / "metrics.jsonl").read_text().splitlines()[0])
    assert "ema_val_top1" in first and "val_top1" not in first and "ema_train_eval_top1" not in first
    assert not (run / "epochs").exists()

    # Diagnostics never change training, so a resumed session may turn them back on.
    train(tiny_config(checkpoint_minutes=60.0), "cpu")
    last = json.loads((run / "metrics.jsonl").read_text().splitlines()[-1])
    assert "val_top1" in last and "ema_train_eval_top1" in last
    assert (run / "epochs" / "epoch_001.npz").exists() and (run / "DONE").exists()


def test_resume_refuses_changed_config(tmp_path, monkeypatch):
    monkeypatch.setenv("BIRD_RUN_ROOT", str(tmp_path))
    run = Path(tmp_path) / "smoke"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(tiny_config().to_dict()))
    with pytest.raises(SystemExit):
        train(tiny_config(lr=5e-4), "cpu")


def test_early_stopping_ends_the_run_and_records_the_best_epoch(tmp_path, monkeypatch):
    monkeypatch.setenv("BIRD_RUN_ROOT", str(tmp_path))
    # Zero learning rate on a model without BatchNorm keeps every prediction fixed, so val never beats epoch 0
    # and patience 2 must stop the run after epoch 2 of 10.
    train(tiny_config(backbone="convnext_atto", epochs=10, lr=0.0, min_lr=0.0, early_stop_patience=2), "cpu")

    # Check the epoch log and the summary
    run = tmp_path / "smoke"
    records = read_jsonl(run / "metrics.jsonl")
    assert [r["epoch"] for r in records] == [0, 1, 2] and records[-1]["early_stop"]
    assert records[-1]["dist_from_init"] == 0 and records[-1]["ema_dist_from_init"] == pytest.approx(0, abs=1e-6)
    summary = json.loads((run / "summary.json").read_text())
    assert (summary["best_epoch"], summary["epochs_trained"]) == (0, 3)
    assert (run / "DONE").exists()


def test_plateau_schedule_cuts_the_lr_at_each_stall_and_checkpoints_it(tmp_path, monkeypatch):
    monkeypatch.setenv("BIRD_RUN_ROOT", str(tmp_path))
    # As above, val never beats epoch 0, so with patience 1 every later epoch cuts the LR and early stopping ends
    # the run after epoch 3.
    train(tiny_config(backbone="convnext_atto", epochs=10, lr=0.0, min_lr=0.0, lr_schedule="plateau",
                      plateau_patience=1, early_stop_patience=3), "cpu")

    # Check the epochs that cut the LR
    run = tmp_path / "smoke"
    records = read_jsonl(run / "metrics.jsonl")
    assert [r.get("lr_cut", False) for r in records] == [False, True, True, True] and records[-1]["early_stop"]

    # The checkpoint after epoch 2 holds two cuts, so a resumed run keeps the lowered LR.
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    assert state["scheduler"] == {"scale": pytest.approx(0.01), "bad_epochs": 0}
