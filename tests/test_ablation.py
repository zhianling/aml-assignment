import json
import sys
import tomllib

import numpy as np
import pytest

from birdcls.config import REPO_ROOT, TrainConfig, load_config

from script_loader import load_script

sys.path.insert(0, str(REPO_ROOT / "scripts"))  # ablation_report imports make_ablation_configs

abl = load_script("make_ablation_configs")
report = load_script("ablation_report")
# Config keys that name or seed a run without being part of the recipe.
BOOKKEEPING = {"run_name", "seed", "notes"}
STUDIES = list(abl.STUDIES.values())
CONVNEXT = abl.STUDIES["convnext"]


@pytest.mark.parametrize("study", STUDIES, ids=lambda s: s.key)
def test_every_variant_changes_only_its_own_keys(study):
    # Build every run of the study
    base = load_config(study.base).to_dict()
    runs = abl.build(study)
    assert len(runs) == len(study.control_seeds) + len(study.variants)

    # Compare each run with the base config
    for name, raw in runs.items():
        cfg = TrainConfig(**raw).to_dict()
        changed = {k for k in cfg if cfg[k] != base[k]} - BOOKKEEPING
        variant = name.split("__")[1]
        expected = set() if variant == "control" else set(
            study.variants[variant][1](tomllib.loads(study.base.read_text())))
        assert changed == expected, name
        # Paired comparison: every variant shares seed 0 (and so data order) with the control run.
        assert cfg["seed"] == (int(name[-1]) if variant == "control" else 0)
        assert cfg["use_val"] and cfg["head_init"] in ("", base["head_init"])


def test_generated_files_match_the_tables():
    files = sorted(p.stem for p in (REPO_ROOT / "configs" / "ablation").glob("*.toml"))
    assert files == sorted(name for study in STUDIES for name in abl.build(study))


def _fake_run(root, name, val_correct, test_correct, best=2):
    """Write a fake run folder whose val and test predictions are right where the masks say."""
    run = root / name
    run.mkdir(parents=True)
    rng = np.random.default_rng(0)

    def logits(correct):
        """Make logits that favor the true class where ``correct`` is true and the next class elsewhere."""
        labels = np.arange(len(correct)) % 200
        z = rng.normal(size=(len(correct), 200)).astype(np.float32)
        z[np.arange(len(correct)), np.where(correct, labels, (labels + 1) % 200)] += 10
        return z, labels

    # Save the logits of both splits
    arrays = {}
    for split, correct in (("val", val_correct), ("test", test_correct)):
        z, labels = logits(np.asarray(correct, bool))
        arrays.update({f"{split}_center": z, f"{split}_hflip": z, f"{split}_labels": labels})
    np.savez(run / "predictions.npz", **arrays)

    # Write the epoch log and the summary
    epochs = [{"epoch": e, "ema_val_top1": 0.9, "ema_train_eval_top1": 0.95 + 0.01 * e,
               "ema_val_loss": 0.5 + 0.1 * abs(e - 1), "ema_train_eval_loss": 0.3 - 0.05 * e} for e in range(4)]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in epochs))
    acc = {s: float(np.mean(c)) for s, c in (("val", val_correct), ("test", test_correct))}
    (run / "summary.json").write_text(json.dumps({
        "best_epoch": best, "epochs_trained": 4, "train_hours": 1.0,
        **{s: {"top1": acc[s], "mean_class_acc": acc[s]} for s in acc}}))


def test_report_compares_with_the_control_seed_band_and_pairs_with_seed_0(tmp_path):
    # Build fake runs for the control seeds and one variant
    n = 20
    base = [True] * 18 + [False] * 2
    _fake_run(tmp_path, CONVNEXT.control_run, base, base)
    _fake_run(tmp_path, CONVNEXT.control_name(1), [True] * 17 + [False] * 3, base)
    _fake_run(tmp_path, CONVNEXT.variant_name("no_reg"), [False] * 4 + [True] * 16, [False] * 5 + [True] * 15)
    # A run of another study in the same run root is not part of this report.
    _fake_run(tmp_path, abl.STUDIES["resnet"].variant_name("no_reg"), base, base)

    # Build the report
    rows, curves = report.build_report(tmp_path, CONVNEXT)
    by_run = {r["run"]: r for r in rows}
    assert set(by_run) == {CONVNEXT.control_run, CONVNEXT.control_name(1), CONVNEXT.variant_name("no_reg")}

    # Check the variant against the control mean and spread
    r = by_run[CONVNEXT.variant_name("no_reg")]
    assert r["group"] == "remove" and r["val_top1"] == pytest.approx(16 / n)
    ctrl_mean, ctrl_sd = np.mean([18 / n, 17 / n]), np.std([18 / n, 17 / n], ddof=1)
    assert r["dval"] == pytest.approx(16 / n - ctrl_mean) and r["dval_sd"] == pytest.approx(r["dval"] / ctrl_sd)
    # Test: the variant misses images 0-4 and the control misses 18-19.
    assert (r["test_only_run"], r["test_only_control"]) == (2, 5)
    assert r["gap_top1_at_best"] == pytest.approx(0.97 - 0.9) and r["min_val_loss_epoch"] == 1
    assert curves[CONVNEXT.control_run]["epoch"] == [0, 1, 2, 3]


def test_report_needs_control_seed_0(tmp_path):
    _fake_run(tmp_path, CONVNEXT.variant_name("no_ls"), [True] * 3, [True] * 3)
    with pytest.raises(SystemExit, match="control seed 0"):
        report.build_report(tmp_path, CONVNEXT)
