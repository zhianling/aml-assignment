"""Offline analysis of saved logits: calibration, ensembling and selective prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .probs import log_probs, nll, softmax, tta_probs


@dataclass
class PredictionSource:
    """Test-time-augmentation probabilities, labels and file names per split for one model or probe."""

    name: str
    probs: dict[str, np.ndarray] = field(default_factory=dict)   # split -> TTA-averaged probabilities
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    files: dict[str, np.ndarray] = field(default_factory=dict)
    temperature: float = 1.0                                     # applied to ``probs`` (1 = raw)
    raw_probs: dict[str, np.ndarray] = field(default_factory=dict)  # before calibration, if calibrated

    @property
    def has_val(self) -> bool:
        """Return whether val probabilities are loaded."""
        return "val" in self.probs


def load_source(name: str, path: Path, views: tuple[str, ...] = ("center", "hflip"),
                calibrate: bool = False) -> PredictionSource:
    """Load logits and average the softmax over the requested test-time-augmentation views.

    The ``.npz`` holds ``{split}_{view}`` logits plus ``{split}_labels`` and ``{split}_files``, written by
    fine-tuned runs and linear probes alike. ``calibrate`` rescales the probabilities with a temperature
    fitted on val; calibrate before mixing, because label smoothing leaves fine-tuned networks
    under-confident and raw averaging would let the more confident probes dominate an ensemble.
    """
    src = PredictionSource(name)
    with np.load(path, allow_pickle=False) as z:
        for split in ("val", "test"):
            if f"{split}_labels" not in z.files:
                continue
            src.probs[split] = tta_probs(*(z[f"{split}_{v}"] for v in views))
            src.labels[split] = z[f"{split}_labels"].astype(np.int64)
            src.files[split] = z[f"{split}_files"]
    if calibrate and src.has_val:
        src.raw_probs = dict(src.probs)
        src.temperature = fit_temperature(log_probs(src.probs["val"]), src.labels["val"])
        src.probs = {split: apply_temperature(p, src.temperature) for split, p in src.probs.items()}
    return src


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Return ``softmax(log p / T)``, which rescales confidence without changing any prediction's rank."""
    return softmax(log_probs(probs) / temperature)


def aligned(sources: list[PredictionSource], split: str) -> None:
    """Raise unless every source lists the same images in the same order, as mixing probabilities requires."""
    ref = sources[0].files[split]
    for s in sources[1:]:
        if not np.array_equal(s.files[split], ref):
            raise ValueError(f"{s.name} and {sources[0].name} disagree on the {split} image order")


def greedy_ensemble(sources: list[PredictionSource], max_members: int = 4) -> list[PredictionSource]:
    """Pick an equal-weight ensemble by forward selection on val log-loss; the test set plays no part.

    Log-loss replaces accuracy because with about 960 val images accuracy moves in coarse 0.1% steps
    and ties are frequent, while log-loss rewards complementary confidence.
    """
    candidates = [s for s in sources if s.has_val]
    aligned(candidates, "val")
    labels = candidates[0].labels["val"]
    chosen: list[PredictionSource] = []
    best = float("inf")
    while len(chosen) < max_members:
        scored = []
        for s in candidates:
            if s in chosen:
                continue
            mix = np.mean([c.probs["val"] for c in chosen + [s]], axis=0)
            scored.append((nll(mix, labels), s))
        if not scored:
            break
        loss, pick = min(scored, key=lambda t: t[0])
        if loss >= best:
            break
        best = loss
        chosen.append(pick)
    return chosen


def ensemble_probs(members: list[PredictionSource], split: str) -> np.ndarray:
    """Return the mean probabilities of the ensemble members on one split."""
    aligned(members, split)
    return np.mean([m.probs[split] for m in members], axis=0)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Return the temperature T that minimizes val log-loss of ``softmax(logits / T)`` (Guo et al., 2017).

    One scalar rescaling fixes most of the over- or under-confidence of a cross-entropy network and
    leaves accuracy untouched.
    """
    grid = np.exp(np.linspace(np.log(0.05), np.log(10.0), 400))
    losses = [nll(softmax(logits / t), labels) for t in grid]
    return float(grid[int(np.argmin(losses))])


def selective_threshold(probs: np.ndarray, labels: np.ndarray, target_accuracy: float = 0.95
                        ) -> tuple[float, float, float]:
    """Return the smallest confidence threshold whose accepted predictions reach ``target_accuracy``.

    The result is (threshold, coverage, accuracy on the accepted predictions). The web app answers
    "not sure" below the threshold instead of showing a probably wrong species.
    """
    conf, correct = probs.max(axis=1), probs.argmax(axis=1) == labels
    for t in np.round(np.arange(0.05, 1.0, 0.05), 2):
        accepted = conf >= t
        if accepted.any() and correct[accepted].mean() >= target_accuracy:
            return float(t), float(accepted.mean()), float(correct[accepted].mean())
    accepted = conf >= 0.5  # target unreachable, so fall back to a plain majority-confidence rule
    return 0.5, float(accepted.mean()), float(correct[accepted].mean()) if accepted.any() else 0.0
