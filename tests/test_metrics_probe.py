import numpy as np
import torch

from birdcls.metrics import (
    expected_calibration_error,
    mcnemar_exact,
    mean_per_class_accuracy,
    per_class_accuracy,
    stratified_bootstrap_ci,
    top1_accuracy,
    topk_accuracy,
)
from birdcls.probe import fit_linear_probe, knn_scores


def test_mean_per_class_differs_from_top1_on_imbalanced_data():
    # Class 0 is 3/4 correct and class 1 is 0/1, so top-1 is 3/5 and the per-class mean is (0.75 + 0) / 2.
    y_true = np.array([0, 0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 1, 0])
    assert top1_accuracy(y_true, y_pred) == 0.6
    assert mean_per_class_accuracy(y_true, y_pred, num_classes=2) == 0.375


def test_per_class_accuracy_marks_absent_classes_nan():
    acc = per_class_accuracy(np.array([0, 0]), np.array([0, 1]), num_classes=3)
    assert acc[0] == 0.5 and np.isnan(acc[1]) and np.isnan(acc[2])


def test_topk():
    scores = np.array([[0.1, 0.5, 0.4], [0.7, 0.2, 0.1]])
    assert topk_accuracy(np.array([2, 2]), scores, 1) == 0.0
    assert topk_accuracy(np.array([2, 2]), scores, 2) == 0.5


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(20), 6)
    pred = np.where(rng.random(len(y)) < 0.8, y, (y + 1) % 20)
    ci = stratified_bootstrap_ci(y, pred, 20, n_boot=300)
    point = top1_accuracy(y, pred)
    assert ci["top1"][0] <= point <= ci["top1"][1]


def test_mcnemar_identical_models_is_not_significant():
    a = np.array([1, 0, 1, 1, 0], bool)
    assert mcnemar_exact(a, a)["p_value"] == 1.0


def test_ece_zero_for_perfectly_calibrated_confident_model():
    probs = np.eye(4)
    assert expected_calibration_error(probs, np.arange(4)) == 0.0


def test_linear_probe_separates_blobs_and_fold_matches():
    # Build separable blobs
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(3, 8)) * 5
    y = np.repeat(np.arange(3), 30)
    x = centres[y] + rng.normal(size=(90, 8)) + 100.0  # large offset exercises standardization

    # Fit the probe
    probe = fit_linear_probe(x, y, num_classes=3, weight_decay=1e-3)
    assert (probe.logits(x).argmax(1) == y).mean() > 0.95

    # Folded weights on raw features must reproduce the standardized probe exactly.
    w, b = probe.fold_standardisation()
    raw = torch.as_tensor(x, dtype=torch.float32) @ w.T + b
    np.testing.assert_allclose(raw.numpy(), probe.logits(x), rtol=1e-4, atol=1e-3)


def test_knn_votes_for_nearest_class():
    train = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    scores = knn_scores(train, np.array([0, 0, 1]), np.array([[1.0, 0.05]]), num_classes=2, k=2)
    assert scores.argmax(1)[0] == 0
