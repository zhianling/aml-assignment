from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from birdcls.config import load_config
from birdcls.models import NUM_CLASSES, build_model, build_optimizer
from birdcls.runtime import pick_device, resolve_amp, synchronize


def main() -> None:
    """Time training steps of a config on random images; print images per second, time per epoch and peak memory."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Build the model and optimizer
    cfg = load_config(args.config, args.overrides)
    cfg.head_init = ""  # random classifier; its starting weights do not affect speed
    device = pick_device(args.device)
    amp_dtype = resolve_amp(cfg.amp, device)
    model, _ = build_model(cfg)
    model.to(device).train()
    if cfg.channels_last:
        model.to(memory_format=torch.channels_last)
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    # Make a random batch
    x = torch.randn(cfg.batch_size, 3, cfg.img_size, cfg.img_size, device=device)
    if cfg.channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    y = torch.randint(0, NUM_CLASSES, (cfg.batch_size,), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Run the warmup and timed steps
    for i in range(args.warmup + args.steps):
        if i == args.warmup:
            synchronize(device)
            start = time.perf_counter()
        ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
        with ctx:
            loss = F.cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    elapsed = time.perf_counter() - start

    # Report the speed
    img_s = args.steps * cfg.batch_size / elapsed
    # Images per epoch: the fit split, or all training images for a run without a val split.
    fit_images = 3870 if cfg.use_val else 4829
    line = (f"{cfg.run_name}: {cfg.backbone} @{cfg.img_size}px bs={cfg.batch_size} amp={amp_dtype} "
            f"ckpt={cfg.grad_checkpointing} on {device}: {img_s:.1f} img/s train, "
            f"~{fit_images / img_s / 60:.1f} min/epoch, "
            f"~{cfg.epochs * fit_images / img_s / 3600:.2f} h for {cfg.epochs} epochs")
    if device.type == "cuda":
        line += f", peak {torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB"
    print(line)


if __name__ == "__main__":
    main()
