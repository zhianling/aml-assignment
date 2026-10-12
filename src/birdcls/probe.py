"""Linear probing, k-NN and zero-shot scoring on cached frozen features."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .probs import tta_probs


@dataclass
class LinearProbe:
    """Multinomial logistic regression trained on standardized features.

    A probe shows how linearly separable the 200 species are in a backbone's feature space without
    touching the backbone, a cheap, fair way to rank models before spending GPU time on fine-tuning.
    """

    weight: torch.Tensor  # [C, D], acts on standardized features
    bias: torch.Tensor    # [C]
    mean: torch.Tensor    # [D] feature mean used for standardization
    std: torch.Tensor     # [D] feature std used for standardization
    weight_decay: float

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Return the class logits for raw features ``x`` of shape [N, D]."""
        z = (torch.as_tensor(x, dtype=torch.float32) - self.mean) / self.std
        return (z @ self.weight.T + self.bias).numpy()

    def fold_standardisation(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the weights and bias that act on raw features: ``W((x - mu) / sigma) + b == (W / sigma) x + b'``.

        This lets the probe be copied straight into the network's final linear layer, which
        receives unstandardized features during fine-tuning (LP-FT).
        """
        w = self.weight / self.std
        b = self.bias - (self.weight * (self.mean / self.std)).sum(dim=1)
        return w, b


def fit_linear_probe(
    x: np.ndarray, y: np.ndarray, num_classes: int, weight_decay: float, max_iter: int = 300,
    on_pass: Callable[[LinearProbe, torch.Tensor, float], None] | None = None,
) -> LinearProbe:
    """Fit a probe by full-batch L-BFGS on the L2-regularized cross-entropy.

    The problem is convex, so L-BFGS converges to the unique optimum in a few hundred
    iterations (seconds on CPU for about 8k x 1k features), cheap enough to sweep the
    regularization strength for every backbone.

    Every loss evaluation is one pass over all training features, the probe's epoch.
    ``on_pass(probe, train_logits, train_loss)`` sees each one, including line-search trials: the
    probe at that point, its logits for ``x`` and the cross-entropy without the penalty. It
    only observes; the fit is the same with or without it.
    """
    # Standardize the features
    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    mean = x_t.mean(dim=0)
    std = x_t.std(dim=0).clamp_min(1e-6)  # guards against dead feature dimensions
    z = (x_t - mean) / std

    # Set up a zero-initialized layer and the optimizer
    torch.manual_seed(0)
    layer = torch.nn.Linear(z.shape[1], num_classes)
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    opt = torch.optim.LBFGS(layer.parameters(), lr=1.0, max_iter=max_iter, history_size=20,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        """Return the penalized loss for L-BFGS, reporting each pass to ``on_pass``."""
        opt.zero_grad()
        logits = layer(z)
        ce = F.cross_entropy(logits, y_t)
        # The bias is not penalized, matching standard logistic-regression practice.
        loss = ce + 0.5 * weight_decay * layer.weight.pow(2).sum()
        loss.backward()
        if on_pass is not None:
            on_pass(LinearProbe(layer.weight.detach(), layer.bias.detach(), mean, std, weight_decay),
                    logits.detach(), ce.item())
        return loss

    # Optimize and return the probe
    opt.step(closure)
    return LinearProbe(layer.weight.detach(), layer.bias.detach(), mean, std, weight_decay)


def view_averaged_probs(probe: LinearProbe, views: np.ndarray) -> np.ndarray:
    """Average the class probabilities over the views (original and flipped) as test-time augmentation."""
    return tta_probs(*(probe.logits(views[:, v]) for v in range(views.shape[1])))


def knn_scores(
    train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, num_classes: int,
    k: int = 10, temperature: float = 0.07,
) -> np.ndarray:
    """Return similarity-weighted k-NN votes on cosine similarity (the DINO evaluation protocol).

    No parameters are fitted at all, so k-NN isolates the raw quality of the feature space.
    """
    tr = F.normalize(torch.as_tensor(train_x, dtype=torch.float32), dim=1)
    q = F.normalize(torch.as_tensor(query_x, dtype=torch.float32), dim=1)
    sim = q @ tr.T
    top_sim, top_idx = sim.topk(k, dim=1)
    weights = (top_sim / temperature).exp()
    votes = torch.zeros(q.shape[0], num_classes)
    votes.scatter_add_(1, torch.as_tensor(train_y, dtype=torch.long)[top_idx], weights)
    return votes.numpy()


def zero_shot_scores(image_x: np.ndarray, text_x: np.ndarray) -> np.ndarray:
    """Return the cosine similarity between image embeddings and class-name text embeddings (CLIP-style)."""
    img = F.normalize(torch.as_tensor(image_x, dtype=torch.float32), dim=1)
    txt = F.normalize(torch.as_tensor(text_x, dtype=torch.float32), dim=1)
    return (img @ txt.T).numpy()
