"""Evaluation metrics (top-1, mean class accuracy, top-5, calibration) and the statistics that compare models."""

from __future__ import annotations

import numpy as np


def top1_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the fraction of images whose predicted class is correct."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Return the accuracy of each class; NaN for classes absent from ``y_true``."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    correct = np.bincount(y_true, weights=(y_true == y_pred).astype(float), minlength=num_classes)
    total = np.bincount(y_true, minlength=num_classes).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, correct / total, np.nan)


def mean_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Return the mean over classes of per-class accuracy, so every species counts equally."""
    return float(np.nanmean(per_class_accuracy(y_true, y_pred, num_classes)))


def topk_accuracy(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Return the fraction of images whose true class is among the ``k`` highest scores.

    Fine-grained errors are often between look-alike species, so top-5 shows whether the right
    answer is nearly found.
    """
    topk = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    return float(np.mean(np.any(topk == np.asarray(y_true)[:, None], axis=1)))


def summarise(y_true: np.ndarray, scores: np.ndarray, num_classes: int) -> dict[str, float]:
    """Return the standard metric bundle (top-1, mean class accuracy, top-5) reported for every model."""
    y_pred = scores.argmax(axis=1)
    return {
        "top1": top1_accuracy(y_true, y_pred),
        "mean_class_acc": mean_per_class_accuracy(y_true, y_pred, num_classes),
        "top5": topk_accuracy(y_true, scores, 5),
    }


def stratified_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Return percentile confidence intervals for top-1 and mean per-class accuracy.

    The test set has only about 6 images per class, so a 1-point gap may be noise. Images are
    resampled within each class so every replicate keeps all 200 classes; plain resampling would
    sometimes drop a class and bias the per-class mean.
    """
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    by_class = [np.flatnonzero(y_true == c) for c in range(num_classes)]
    by_class = [idx for idx in by_class if len(idx)]
    top1, mpca = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(members, size=len(members), replace=True) for members in by_class])
        top1[b] = top1_accuracy(y_true[idx], y_pred[idx])
        mpca[b] = mean_per_class_accuracy(y_true[idx], y_pred[idx], num_classes)
    q = (100 * alpha / 2, 100 * (1 - alpha / 2))
    return {"top1": tuple(np.percentile(top1, q)), "mean_class_acc": tuple(np.percentile(mpca, q))}


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    """Run the exact McNemar test on paired per-image correctness of two models.

    Only the discordant images (one model right, the other wrong) carry information about
    which model is better; under the null each discordant image is a fair coin flip.
    """
    from scipy.stats import binomtest  # imported lazily: only needed for offline analysis

    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue)
    return {"only_a": only_a, "only_b": only_b, "p_value": p}


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Return the gap between confidence and accuracy, averaged over equal-width confidence bins.

    It shows whether the confidence displayed in the web app can be trusted.
    """
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == np.asarray(y_true)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)
