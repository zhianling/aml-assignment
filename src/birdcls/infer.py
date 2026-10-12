"""Turn trained weights into saved logits: ``python -m birdcls.infer --run runs/<run_name>``."""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import REPO_ROOT, TrainConfig
from .data import BirdDataset, Sample, build_eval_transform, load_val_split
from .fileio import save_npz_atomic
from .metrics import summarise
from .models import NUM_CLASSES, build_model
from .probs import tta_probs
from .runtime import pick_device, resolve_amp


@torch.no_grad()
def predict_logits(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, amp_dtype: torch.dtype | None,
    channels_last: bool = False,
) -> dict[str, np.ndarray]:
    """Return logits for the plain view and its horizontal flip (test-time augmentation), plus the labels."""
    model.eval()
    views: dict[str, list[np.ndarray]] = {"center": [], "hflip": []}
    labels: list[np.ndarray] = []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
        with ctx:
            views["center"].append(model(x).float().cpu().numpy())
            views["hflip"].append(model(torch.flip(x, dims=[3])).float().cpu().numpy())
        labels.append(y.numpy())
    out = {k: np.concatenate(v) for k, v in views.items()}
    out["labels"] = np.concatenate(labels)
    return out


def eval_loader(cfg: TrainConfig, samples: list[Sample], data_cfg: dict) -> DataLoader:
    """Build the unshuffled evaluation loader, with twice the training batch size."""
    transform = build_eval_transform(cfg.img_size, data_cfg["mean"], data_cfg["std"], cfg.eval_crop_pct)
    return DataLoader(BirdDataset(cfg.data_root, samples, transform), batch_size=cfg.batch_size * 2,
                      shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())


def write_predictions(
    model: torch.nn.Module, cfg: TrainConfig, data_cfg: dict, device: torch.device, out_path: Path
) -> dict[str, dict[str, float]]:
    """Save val (if held out) and test logits with their filenames; return headline metrics."""
    _, val, test = load_val_split(cfg.data_root, REPO_ROOT / "metadata" / "val_split.json")
    splits = {"test": test}
    if cfg.use_val:
        splits = {"val": val, "test": test}
    amp_dtype = resolve_amp(cfg.amp, device)
    arrays, report = {}, {}
    for name, samples in splits.items():
        if cfg.limit_eval:
            samples = samples[: cfg.limit_eval]
        pred = predict_logits(model, eval_loader(cfg, samples, data_cfg), device, amp_dtype,
                              channels_last=cfg.channels_last)
        for view in ("center", "hflip"):
            arrays[f"{name}_{view}"] = pred[view].astype(np.float32)
        arrays[f"{name}_labels"] = pred["labels"]
        arrays[f"{name}_files"] = np.array([s.filename for s in samples])
        tta = tta_probs(pred["center"], pred["hflip"])
        report[name] = summarise(pred["labels"], tta, NUM_CLASSES)
    save_npz_atomic(out_path, **arrays)
    return report


def load_run(run_dir: Path, weights: str, device: torch.device) -> tuple[torch.nn.Module, TrainConfig, dict]:
    """Rebuild a finished run's network from its saved config and fp16 EMA weights."""
    cfg = TrainConfig(**json.loads((run_dir / "config.json").read_text()))
    cfg.pretrained, cfg.head_init = False, ""  # weights come from the run itself
    cfg.grad_checkpointing = False  # a training-memory trick that only slows inference down
    model, data_cfg = build_model(cfg)
    state = torch.load(run_dir / weights, map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() for k, v in state.items()})
    model.to(device)
    if cfg.channels_last:
        model.to(memory_format=torch.channels_last)
    return model, cfg, data_cfg


def main() -> None:
    """Load a finished run, write its ``predictions.npz`` and print the headline metrics."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="run directory containing config.json")
    parser.add_argument("--weights", default="best_ema.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, cfg, data_cfg = load_run(args.run, args.weights, device)
    report = write_predictions(model, cfg, data_cfg, device, args.run / "predictions.npz")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
