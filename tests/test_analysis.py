import numpy as np
import pytest

from birdcls.analysis import (
    PredictionSource,
    ensemble_probs,
    fit_temperature,
    greedy_ensemble,
    load_source,
    selective_threshold,
)
from birdcls.probs import softmax


def make_source(name, correct_rate, seed, n=200, c=10):
    """Build a synthetic model that puts most mass on the true class for a random subset of images."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % c
    probs = rng.dirichlet(np.ones(c), size=n) * 0.5
    right = rng.random(n) < correct_rate
    probs[np.arange(n), np.where(right, labels, (labels + 1) % c)] += 0.5
    files = np.array([f"{i:04d}.jpg" for i in range(n)])
    return PredictionSource(name, {"val": probs, "test": probs}, {"val": labels, "test": labels},
                            {"val": files, "test": files})


def test_greedy_ensemble_starts_with_best_model_and_skips_useless_one():
    good, ok = make_source("good", 0.9, 0), make_source("ok", 0.7, 1)
    bad = make_source("bad", 0.1, 2)
    chosen = greedy_ensemble([bad, ok, good], max_members=3)
    assert chosen[0] is good
    assert bad not in chosen


def test_ensemble_rejects_misaligned_sources():
    a, b = make_source("a", 0.8, 0), make_source("b", 0.8, 1)
    b.files["test"] = b.files["test"][::-1]
    with pytest.raises(ValueError):
        ensemble_probs([a, b], "test")


def test_load_source_averages_tta_views(tmp_path):
    logits = np.zeros((2, 3))
    flipped = np.array([[10.0, 0, 0], [0, 10.0, 0]])
    np.savez(tmp_path / "p.npz", test_center=logits, test_hflip=flipped,
             test_labels=np.array([0, 1]), test_files=np.array(["a", "b"]))
    src = load_source("p", tmp_path / "p.npz")
    assert not src.has_val
    np.testing.assert_allclose(src.probs["test"].sum(1), 1.0)
    assert list(src.probs["test"].argmax(1)) == [0, 1]


def test_calibrated_source_keeps_its_predictions_and_sharpens_underconfidence(tmp_path):
    # Save logits for val and test
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, 400)
    logits = rng.normal(size=(400, 5))
    logits[np.arange(400), labels] += 3.0
    files = np.array([f"{i}.jpg" for i in range(400)])
    arrays = {}
    for split in ("val", "test"):
        # A 4x under-confident model (logits shrunk), as label smoothing tends to produce.
        arrays |= {f"{split}_center": logits / 4, f"{split}_hflip": logits / 4,
                   f"{split}_labels": labels, f"{split}_files": files}
    np.savez(tmp_path / "p.npz", **arrays)

    # Load the raw and the calibrated source
    raw, cal = load_source("p", tmp_path / "p.npz"), load_source("p", tmp_path / "p.npz", calibrate=True)

    # Check calibration keeps the predictions and raises the confidence
    assert cal.temperature < 0.5
    np.testing.assert_array_equal(cal.probs["test"].argmax(1), raw.probs["test"].argmax(1))
    assert cal.probs["test"].max(1).mean() > raw.probs["test"].max(1).mean()
    np.testing.assert_allclose(cal.raw_probs["test"], raw.probs["test"])


def test_fit_temperature_recovers_overconfidence():
    # Build logits that favor the true class
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, 2000)
    true_logits = rng.normal(size=(2000, 5))
    true_logits[np.arange(2000), labels] += 1.5

    # Sample labels consistent with softmax(true_logits), then make the model 3x overconfident.
    probs = softmax(true_logits)
    sampled = np.array([rng.choice(5, p=p) for p in probs])
    t = fit_temperature(true_logits * 3, sampled)

    # Check the temperature is near 3
    assert 2.4 < t < 3.6


def test_selective_threshold_reaches_target():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.55, 0.45]])
    labels = np.array([0, 0, 1, 1])  # the two low-confidence predictions are wrong
    t, coverage, acc = selective_threshold(probs, labels, 0.95)
    assert acc == 1.0 and coverage == 0.5 and 0.6 < t <= 0.8
