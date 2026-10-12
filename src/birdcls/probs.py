"""Class probabilities from logits, and their log-loss."""

from __future__ import annotations

import numpy as np

# Probabilities are floored here before the log, so a zero probability gives a large finite loss.
MIN_PROB = 1e-12


def softmax(logits: np.ndarray) -> np.ndarray:
    """Turn each row of logits into class probabilities that sum to 1.

    The row maximum is subtracted first so ``np.exp`` cannot overflow on large logits.
    """
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def tta_probs(*view_logits: np.ndarray) -> np.ndarray:
    """Average the softmax probabilities of several views of the same images.

    Every model scores each image as it is (``center``) and mirrored (``hflip``); the average of
    the two softmax outputs is the prediction that every result in the project reports.
    """
    return np.mean([softmax(z) for z in view_logits], axis=0)


def log_probs(probs: np.ndarray) -> np.ndarray:
    """Natural log of probabilities, floored at ``MIN_PROB`` so it stays finite."""
    return np.log(np.clip(probs, MIN_PROB, None))


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood (cross-entropy) of the true labels under ``probs``."""
    return float(-np.mean(log_probs(probs[np.arange(len(labels)), labels])))
