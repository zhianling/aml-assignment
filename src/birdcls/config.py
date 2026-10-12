"""Run configuration: one TOML file per experiment, loaded into a typed dataclass."""

from __future__ import annotations

import dataclasses
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
NUM_CLASSES = 200  # bird species in CUB-200-2010


@dataclass
class TrainConfig:
    """Every setting of one training run; a versioned TOML file holds the values so a run can be resumed exactly."""

    # Model and input
    run_name: str                    # run directory name under BIRD_RUN_ROOT (default runs/); unique per run
    backbone: str                    # key in backbones.BACKBONES, or a raw timm model name
    pretrained: bool = True          # start from pretrained weights; False = random init
    img_size: int = 448              # input side in pixels (square)
    eval_crop_pct: float = 1.0       # 1.0 = resize short side to img_size, keep the whole center
    global_pool: str = ""            # "" = the model's own pooling; "token" = classify from the CLS token

    # Optimizer
    epochs: int = 30                 # training epochs; a cap when the plateau schedule and early stopping are on
    warmup_epochs: float = 2.0       # linear LR warm-up from min_lr, in epochs (may be fractional)
    batch_size: int = 16             # images per forward pass (bounded by T4 memory)
    accum_steps: int = 4             # batches per optimizer update -> effective batch 64
    lr: float = 1e-4                 # peak LR of the classifier head
    min_lr: float = 1e-6             # LR floor and warm-up start value
    # "cosine": warm-up, then decay to min_lr over `epochs`. "plateau": the peak LR until EMA val
    # top-1 stalls for `plateau_patience` epochs, then `plateau_factor` times lower at each stall;
    # with early stopping, `epochs` is then only a cap. Needs use_val.
    lr_schedule: str = "cosine"
    plateau_patience: int = 5        # evaluated epochs without a new best before the LR is cut (plateau only)
    plateau_factor: float = 0.1      # LR multiplier at each cut (plateau only)
    layer_decay: float = 0.8         # LR multiplier per layer going down the network (LLRD); 1.0 = off
    weight_decay: float = 0.05       # AdamW decoupled weight decay
    clip_grad: float = 1.0           # max gradient norm (L2); must be > 0

    # Regularization (the brief warns about overfitting: ~24 images per class)
    drop_path: float = 0.2           # stochastic-depth rate; 0 = off
    drop_rate: float = 0.0           # dropout on the pooled features before the classifier
    label_smoothing: float = 0.1     # share of probability mass moved off the true class
    mixup_alpha: float = 0.0         # Mixup Beta parameter; 0 = off
    cutmix_alpha: float = 0.0        # CutMix Beta parameter; 0 = off
    mixup_prob: float = 0.0          # probability a batch gets Mixup or CutMix; 0 = off
    rrc_scale_min: float = 0.35      # RandomResizedCrop lower bound; too small crops lose the bird
    augment: str = "trivialaugment"  # "trivialaugment", "randaugment" or "none"
    random_erasing: float = 0.25     # per-image probability of erasing a random patch; 0 = off
    ema_decay: float = 0.998         # EMA decay of the weights per optimizer update

    # LP-FT
    head_init: str = ""              # path (relative to repo) of a folded linear-probe head; "" = fresh head

    # Data and bookkeeping
    use_val: bool = True             # True: train on fit split and select on val; False: train on all
    eval_every: int = 1              # evaluate every N epochs (the last epoch is always evaluated)
    # >0: stop once this many epochs pass without a new best EMA val top-1. The EMA averages
    # over ~500 updates (~8 epochs here), so val moves slowly and plateaus of 5 epochs occur
    # before the best epoch; patience must outlast them. Needs use_val.
    early_stop_patience: int = 0

    # Diagnostics (logging only: they never change training, so they may differ between resumed
    # sessions). Each evaluation costs about one val pass per epoch.
    train_eval_images: int = 959       # fixed stratified fit subset scored like val every epoch (0 = off)
    eval_raw_every_epoch: bool = True  # also score the raw (non-EMA) weights every epoch, not just the last
    save_epoch_logits: bool = True     # keep every epoch's logits (fp16) in <run>/epochs/ for later analysis

    # Runtime
    seed: int = 0                    # seeds Python, NumPy, torch and the training data order
    num_workers: int = 2             # DataLoader worker processes
    amp: str = "auto"                # "auto", "fp16", "bf16" or "none"
    grad_checkpointing: bool = False  # recompute activations in the backward pass: less memory, slower
    channels_last: bool = True       # use the channels-last memory format for speed
    checkpoint_minutes: float = 10.0  # minutes between full checkpoints
    time_budget_hours: float = 0.0   # >0: checkpoint and exit before a hosted session's limit (Kaggle)
    prune_on_finish: bool = True     # drop optimizer state once a run is complete (Drive quota)
    limit_train: int = 0             # >0: keep only the first N training images (smoke tests)
    limit_eval: int = 0              # >0: keep only the first N val and train-eval images (smoke tests)
    notes: str = ""                  # free text kept with the run

    def __post_init__(self) -> None:
        """Reject an unknown or inconsistent LR schedule."""
        if self.lr_schedule not in ("cosine", "plateau"):
            raise ValueError(f"lr_schedule must be 'cosine' or 'plateau', not {self.lr_schedule!r}")
        if self.lr_schedule == "plateau" and not self.use_val:
            raise ValueError("lr_schedule = 'plateau' reacts to val top-1, so it needs use_val")

    @property
    def data_root(self) -> Path:
        """Dataset directory: ``BIRD_DATA_ROOT`` if set, else ``data/cub`` in the repo."""
        return Path(os.environ.get("BIRD_DATA_ROOT", REPO_ROOT / "data" / "cub"))

    @property
    def run_dir(self) -> Path:
        """Output directory of this run: ``BIRD_RUN_ROOT`` (default ``runs``) joined with ``run_name``."""
        return Path(os.environ.get("BIRD_RUN_ROOT", REPO_ROOT / "runs")) / self.run_name

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as a plain dict."""
        return dataclasses.asdict(self)


def _parse_override(item: str) -> tuple[str, Any]:
    """Parse ``key=value`` where value is TOML syntax, e.g. ``epochs=2`` or ``augment="none"``."""
    key, sep, raw = item.partition("=")
    if not sep:
        raise ValueError(f"Override {item!r} must look like key=value")
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw  # allow unquoted strings on the command line
    return key.strip(), value


def load_config(path: Path, overrides: list[str] | None = None) -> TrainConfig:
    """Read the TOML file at ``path``, apply ``key=value`` overrides and build a ``TrainConfig``."""
    raw = tomllib.loads(Path(path).read_text())
    for item in overrides or []:
        key, value = _parse_override(item)
        raw[key] = value
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys {sorted(unknown)} in {path}")
    return TrainConfig(**raw)
