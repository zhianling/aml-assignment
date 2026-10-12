import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from birdcls.data import Sample
from birdcls.probe import fit_linear_probe
from birdcls.train import eval_metrics, train_eval_subset, weight_stats
from script_loader import load_script


def test_eval_loss_matches_cross_entropy_with_and_without_smoothing():
    torch.manual_seed(0)
    logits, labels = torch.randn(32, 200) * 3, torch.randint(0, 200, (32,))
    # With identical views the flip-averaged prediction is just softmax(logits).
    pred = {"center": logits.numpy(), "hflip": logits.numpy(), "labels": labels.numpy()}
    out = eval_metrics(pred, smoothing=0.1)
    assert out["loss"] == pytest.approx(F.cross_entropy(logits, labels).item(), rel=1e-5)
    assert out["loss_smoothed"] == pytest.approx(F.cross_entropy(logits, labels, label_smoothing=0.1).item(), rel=1e-5)
    assert out["top1"] == out["top1_center"] == out["top1_hflip"]
    assert 0 <= out["ece"] <= 1


def test_train_eval_subset_is_stratified_and_fixed_by_the_seed():
    samples = [Sample(f"{c}_{i}.jpg", c, "Train") for c in range(10) for i in range(4 + c)]
    subset = train_eval_subset(samples, 25, seed=0)
    assert set(s.label for s in subset) == set(range(10))  # every class is scored
    assert 20 <= len(subset) <= 30
    counts = np.bincount([s.label for s in subset])
    assert counts[9] > counts[0]  # larger classes keep their larger share
    assert subset == train_eval_subset(samples, 25, seed=0)
    assert train_eval_subset(samples, 1000, seed=0) == samples


def test_weight_stats_separate_the_head_from_the_backbone():
    net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    init = {n: p.detach().clone() for n, p in net.named_parameters()}
    with torch.no_grad():
        net[1].weight += 1.0  # move only the classifier: 8 weights by 1 each
    stats = weight_stats(net, init, head={"1.weight", "1.bias"})
    assert stats["dist_from_init"] == 0 and stats["head_dist_from_init"] == pytest.approx(8 ** 0.5)
    assert stats["param_norm"] > 0


def test_cam_zoom_logs_every_setting_and_ties_go_to_the_first():
    # Build a full view that is always wrong and one zoom setting that fixes it
    cam_zoom = load_script("cam_zoom")
    labels = np.arange(4)
    full = np.full((4, 200), 1e-4)
    full[labels, labels + 1], full[labels, labels] = 0.6, 0.4  # the full view is always wrong
    zoomed = np.full((4, 200), 1e-4)
    zoomed[labels, labels] = 0.9  # only one zoom setting fixes it, whatever the blend
    v = {"full": full, "labels": labels}
    v.update({f"zoom_{t}_{m}": full for t in cam_zoom.THRESHOLDS for m in cam_zoom.MARGINS})
    v["zoom_0.35_0.25"] = zoomed

    # Score the grid of settings
    grid = cam_zoom.val_grid(v)
    assert len(grid) == len(cam_zoom.THRESHOLDS) * len(cam_zoom.MARGINS) * len(cam_zoom.BLEND)
    assert {(g["threshold"], g["margin"]) for g in grid if g["top1"] == 1} == {(0.35, 0.25)}

    # Check the best setting is the first blend of the fixing setting
    best = max(grid, key=lambda g: g["top1"])  # as main() picks it
    assert (best["threshold"], best["margin"], best["blend_full"]) == (0.35, 0.25, cam_zoom.BLEND[0])


def test_probe_sweep_logs_fit_and_val_scores_for_every_strength(tmp_path):
    # Build a toy feature cache
    run_probes = load_script("run_probes")
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(3), 20)
    feats = rng.normal(size=(60, 2, 8)) + 3 * np.eye(8)[labels][:, None, :]  # separable blobs, 2 views
    splits = np.where(np.arange(60) % 20 < 15, "Train", "Test")  # 15 training + 5 test images per class
    filenames = np.array([f"img_{i}.jpg" for i in range(60)])
    meta = {"key": "toy", "name": "toy", "family": "CNN", "pretraining": "none", "in_domain": False,
            "img_size": 64, "params_m": 0.0, "images_per_second": 1.0}
    cache = tmp_path / "toy@64.npz"
    np.savez(cache, labels=labels.astype(np.int16), filenames=filenames, splits=splits,
             meta=np.array(json.dumps(meta)), feat_pooled=feats.astype(np.float16))
    val_names = {f"img_{i}.jpg" for i in (0, 1, 20, 21, 40, 41)}

    # Sweep the weight decays on the cache
    rows, grid, curves = run_probes.evaluate_cache(cache, val_names, lambda names: np.ones(len(names), bool), 3,
                                                   tmp_path / "heads")

    # Check the grid and the chosen strength
    assert len(rows) == 1 and len(grid) == len(run_probes.WEIGHT_DECAYS)
    assert all(set(g) == set(run_probes.GRID_FIELDS) for g in grid)
    assert [g["weight_decay"] for g in grid if g["chosen"]] == [rows[0]["best_wd"]]
    # Weaker regularization fits the training images at least as tightly.
    assert grid[0]["fit_loss"] <= grid[-1]["fit_loss"]

    # One training history per sweep probe plus the refit on all training images.
    runs = {(c["weight_decay"], c["fitted_on"]) for c in curves}
    assert runs == {(wd, "fit") for wd in run_probes.WEIGHT_DECAYS} | {(rows[0]["best_wd"], "train")}
    for wd, fitted_on in runs:
        history = [c for c in curves if (c["weight_decay"], c["fitted_on"]) == (wd, fitted_on)]
        assert [c["pass"] for c in history] == list(range(1, len(history) + 1))
        assert history[0]["objective_ce"] == pytest.approx(np.log(3), rel=1e-5)  # zero init: uniform
        assert history[-1]["train_loss"] < history[0]["train_loss"]
        assert ("val_loss" in history[0]) == (fitted_on == "fit")  # the refit has no held-out split
    assert set().union(*curves) <= set(run_probes.CURVE_FIELDS)


def test_probe_logging_does_not_change_the_fit():
    rng = np.random.default_rng(1)
    y = np.repeat(np.arange(4), 10)
    x = rng.normal(size=(40, 6)) + 2 * np.eye(6)[y]
    passes = []
    logged = fit_linear_probe(x, y, 4, 1e-3, on_pass=lambda probe, logits, ce: passes.append(ce))
    plain = fit_linear_probe(x, y, 4, 1e-3)
    assert torch.equal(logged.weight, plain.weight) and torch.equal(logged.bias, plain.bias)
    assert len(passes) > 1 and passes[-1] < passes[0]
