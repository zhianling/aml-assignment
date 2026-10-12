"""Model construction for fine-tuning: a 200-way head, LP-FT head initialization and the LLRD optimizer."""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn as nn

from .backbones import BACKBONES, BackboneSpec
from .config import NUM_CLASSES, REPO_ROOT, TrainConfig


def resolve_spec(backbone: str) -> BackboneSpec:
    """Return the registry entry for ``backbone``; unknown names are treated as plain timm models (tests)."""
    if backbone in BACKBONES:
        return BACKBONES[backbone]
    return BackboneSpec(backbone, "timm", backbone, "CNN", "custom timm model")


def build_model(cfg: TrainConfig, timm_name: str | None = None) -> tuple[nn.Module, dict]:
    """Create the network and return it with its preprocessing config (mean/std).

    ``timm_name`` replaces the backbone's timm name, e.g. ``local-dir:<folder>`` holding a copy of
    its hub config.json, so the web app builds the network without asking the Hugging Face Hub.

    The pretrained classifier is replaced by a fresh 200-way linear layer. Stochastic depth
    (``drop_path_rate``) randomly skips residual blocks during training, a regularizer that matters
    a lot when fine-tuning a 300M-parameter model on under 5k images. ``drop_rate`` is plain dropout
    in front of the classifier (0 in every recipe; the ablation tests it).
    """
    import timm

    # Build the network with a fresh head
    spec = resolve_spec(cfg.backbone)
    if spec.source != "timm":
        raise ValueError(f"{spec.key} is an open_clip model; this project uses it for frozen probing only")
    kwargs = {"num_classes": NUM_CLASSES, "drop_path_rate": cfg.drop_path, "drop_rate": cfg.drop_rate}
    if cfg.global_pool:
        # DINOv3's timm head averages patch tokens, yet its linear probe is about 17 pp better on the CLS token
        # (Phase 1), and only "token" has a matching probe head for LP-FT, so only "token" is accepted.
        if cfg.global_pool != "token":
            raise ValueError(f"global_pool must be '' (model default) or 'token', not {cfg.global_pool!r}")
        kwargs["global_pool"] = cfg.global_pool
    if spec.family == "ViT" and spec.resizable:
        kwargs["img_size"] = cfg.img_size  # position embeddings interpolated to the new grid
    model = timm.create_model(timm_name or spec.name, pretrained=cfg.pretrained, **kwargs)
    data_cfg = timm.data.resolve_model_data_config(model)

    # Initialize the head and set memory options
    if cfg.head_init:
        load_probe_head(model, REPO_ROOT / cfg.head_init, cfg.img_size,
                        variant="cls" if cfg.global_pool == "token" else "pooled", validates=cfg.use_val)
    if cfg.grad_checkpointing:
        # Recompute activations in the backward pass instead of storing them: about 30% slower, but a ViT-L at
        # 448 px fits into the T4's 16 GB.
        model.set_grad_checkpointing(True)
    return model, data_cfg


def load_probe_head(model: nn.Module, path: Path, img_size: int, variant: str = "pooled",
                    validates: bool = True) -> None:
    """Copy a linear probe fitted on frozen features into the classifier (LP-FT).

    Starting fine-tuning from a sensible head means the first gradient steps no longer push
    large, random errors back into the pretrained features (Kumar et al., ICLR 2022), which
    preserves more of what pretraining learned when labeled data is scarce. ``variant`` names
    the cached feature the classifier reads: "pooled" for the model's own pooling, "cls" when
    the model was rebuilt to classify from the CLS token.

    A run that ``validates`` selects its checkpoint on the val split, so its head must come from
    a probe fitted on the fit split alone: a probe that also saw val starts fine-tuning with
    val memorized and makes every val score (and the checkpoint chosen by it) optimistic.
    """
    # Load the probe and check it matches the classifier
    state = torch.load(path, map_location="cpu", weights_only=False)
    fitted_on = state.get("fitted_on", "train")  # heads saved before this field saw all training images
    if validates and fitted_on != "fit":
        raise ValueError(f"{path.name}: probe was fitted on '{fitted_on}' (val included), but this run "
                         "selects on val; start from the fit-split head (<cache>__<variant>__fit.pt)")
    if state["variant"] != variant:
        raise ValueError(
            f"{path.name}: probe fitted on '{state['variant']}' features, but the classifier reads '{variant}'")
    head = model.get_classifier()
    if tuple(head.weight.shape) != tuple(state["weight"].shape):
        raise ValueError(
            f"Probe head {tuple(state['weight'].shape)} does not match classifier {tuple(head.weight.shape)}")
    probe_size = state["meta"].get("img_size")
    if probe_size != img_size:
        warnings.warn(f"Probe was fitted at {probe_size}px but fine-tuning runs at {img_size}px", stacklevel=2)

    # Copy the probe into the classifier
    with torch.no_grad():
        head.weight.copy_(state["weight"])
        head.bias.copy_(state["bias"])


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Build AdamW with layer-wise learning-rate decay (LLRD).

    Early layers hold generic edge and texture detectors that transfer as-is, so they get the
    smallest learning rate (``lr * layer_decay ** depth``); the head gets the full ``lr``.
    timm also exempts biases and normalization weights from weight decay.
    """
    from timm.optim import create_optimizer_v2

    return create_optimizer_v2(
        model, opt="adamw", lr=cfg.lr, weight_decay=cfg.weight_decay,
        layer_decay=cfg.layer_decay if cfg.layer_decay < 1.0 else None,
    )
