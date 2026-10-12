"""Resumable fine-tuning for pre-emptible sessions: rerunning a command continues from the last checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpoint import AsyncCheckpointWriter, atomic_save, half_state_dict, restore_rng_state, rng_state
from .config import REPO_ROOT, TrainConfig, load_config
from .data import BirdDataset, Sample, build_train_transform, load_val_split
from .infer import eval_loader, predict_logits, write_predictions
from .metrics import expected_calibration_error, summarise, top1_accuracy
from .models import NUM_CLASSES, build_model, build_optimizer
from .probs import log_probs, nll, tta_probs
from .runtime import pick_device, resolve_amp


def seed_everything(seed: int) -> None:
    """Seed the Python, NumPy and torch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log_line(run_dir: Path, record: dict) -> None:
    """Append one JSON line per epoch to metrics.jsonl and print it."""
    with (run_dir / "metrics.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()), flush=True)


def soft_target_loss(logits: torch.Tensor, target: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Compute cross-entropy for hard labels or Mixup/CutMix soft labels.

    Label smoothing moves ``smoothing`` of the probability mass from the true class to all classes. With 200
    visually similar species this stops the model from becoming over-confident on the training set.
    """
    if target.ndim == 1:
        return F.cross_entropy(logits, target, label_smoothing=smoothing)
    return torch.sum(-target * F.log_softmax(logits, dim=-1), dim=-1).mean()


class PlateauLR:
    """Warm up, hold the peak LR until EMA val top-1 stalls, then cut it by ``factor`` at each further stall.

    A stall is ``patience`` evaluated epochs without improvement, and the LR never goes below ``min_lr``.
    Cosine decay ties the LR to a fixed epoch count: with a generous cap and early stopping the LR would
    still be near its peak when the run stops. Here it drops only once learning at the current LR has
    stalled, so a run can be given many epochs and end by itself. It offers the parts of timm's scheduler
    interface that ``train`` uses.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, min_lr: float, warmup_updates: int,
                 factor: float, patience: int) -> None:
        """Record each group's peak LR and apply the first warm-up value."""
        self.optimizer, self.min_lr, self.warmup_updates = optimizer, min_lr, warmup_updates
        self.factor, self.patience = factor, patience
        # timm's layer-decay groups carry their LR multiplier as "lr_scale".
        self.peaks = [g["lr"] * g.get("lr_scale", 1.0) for g in optimizer.param_groups]
        self.scale, self.bad_epochs, self.num_updates = 1.0, 0, 0
        self.step_update(0)

    def step_update(self, num_updates: int) -> None:
        """Set every group's LR for update number ``num_updates``."""
        self.num_updates = num_updates
        warm = min(1.0, num_updates / self.warmup_updates) if self.warmup_updates else 1.0
        for group, peak in zip(self.optimizer.param_groups, self.peaks):
            target = max(self.min_lr, peak * self.scale)
            group["lr"] = self.min_lr + (target - self.min_lr) * warm

    def step_epoch(self, improved: bool) -> bool:
        """Count one evaluated epoch and return True when it cut the LR."""
        self.bad_epochs = 0 if improved else self.bad_epochs + 1
        if self.bad_epochs < self.patience:
            return False
        self.scale, self.bad_epochs = self.scale * self.factor, 0
        self.step_update(self.num_updates)  # the next update already uses the lower LR
        return True

    def state_dict(self) -> dict:
        """Return the state a resumed run needs."""
        return {"scale": self.scale, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict) -> None:
        """Restore the state saved by ``state_dict``."""
        self.scale, self.bad_epochs = state["scale"], state["bad_epochs"]


def append_jsonl(path: Path, records: list[dict]) -> None:
    """Append ``records`` to ``path``, one JSON object per line."""
    with path.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)


def train_eval_subset(samples: list[Sample], n: int, seed: int) -> list[Sample]:
    """Pick a fixed, class-stratified sample of about ``n`` training images, scored like val every epoch.

    The training loss is measured under augmentation, label smoothing and drop-path, so it cannot be compared
    with the val loss. Scoring these images exactly like val gives like-for-like training and val curves.
    """
    if n >= len(samples):
        return list(samples)
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_class[s.label].append(i)
    picked = []
    for idx in by_class.values():
        k = min(len(idx), max(1, round(len(idx) * n / len(samples))))
        picked += rng.choice(idx, size=k, replace=False).tolist()
    return [samples[i] for i in sorted(picked)]


def eval_metrics(pred: dict[str, np.ndarray], smoothing: float) -> dict[str, float]:
    """Return the accuracy, loss and calibration of one scored split's flip-averaged prediction.

    Also the loss under the training label smoothing and the top-1 of each view on its own.
    """
    labels = pred["labels"]
    probs = tta_probs(pred["center"], pred["hflip"])
    loss = nll(probs, labels)
    out = summarise(labels, probs, NUM_CLASSES)
    out.update(loss=loss,
               loss_smoothed=(1 - smoothing) * loss
                             + smoothing * float(-log_probs(probs).mean(axis=1).mean()),
               ece=expected_calibration_error(probs, labels),
               top1_center=top1_accuracy(labels, pred["center"].argmax(1)),
               top1_hflip=top1_accuracy(labels, pred["hflip"].argmax(1)))
    return out


@torch.no_grad()
def weight_stats(net: torch.nn.Module, init: dict[str, torch.Tensor], head: set[str]) -> dict[str, float]:
    """Return the weight norm and the distance from the starting weights, backbone and classifier apart.

    The starting point is the pretrained backbone plus probe head; the distances show how far fine-tuning
    moved each part.
    """
    norm = body = head_sq = 0.0
    for name, p in net.named_parameters():
        w = p.detach().float()
        norm += w.pow(2).sum().item()
        d = (w - init[name].to(w.device)).pow(2).sum().item()
        if name in head:
            head_sq += d
        else:
            body += d
    return {"param_norm": norm ** 0.5, "dist_from_init": body ** 0.5, "head_dist_from_init": head_sq ** 0.5}


def timed_batches(loader: DataLoader, clock: dict[str, float]):
    """Iterate ``loader`` and add the time spent waiting for each batch to ``clock["wait"]``."""
    it = iter(loader)
    while True:
        start = time.monotonic()
        try:
            batch = next(it)
        except StopIteration:
            return
        clock["wait"] += time.monotonic() - start
        yield batch


def code_commit() -> str:
    """Return the short commit hash of the code, or "unknown".

    The Kaggle kernel sets ``BIRD_COMMIT`` because its code is a git archive without history; elsewhere
    the hash is read from the checkout.
    """
    if os.environ.get("BIRD_COMMIT"):
        return os.environ["BIRD_COMMIT"]
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# Settings that only affect speed or bookkeeping and may differ between resumed sessions.
RESUMABLE_KEYS = {"num_workers", "checkpoint_minutes", "time_budget_hours", "early_stop_patience", "notes",
                  "train_eval_images", "eval_raw_every_epoch", "save_epoch_logits"}


def check_resume_config(cfg: TrainConfig, run_dir: Path) -> None:
    """Exit if ``cfg`` differs from the run's saved config in any setting outside ``RESUMABLE_KEYS``."""
    saved = json.loads((run_dir / "config.json").read_text())
    changed = {k for k, v in cfg.to_dict().items() if saved.get(k) != v} - RESUMABLE_KEYS
    if changed:
        raise SystemExit(f"Config changed since this run started ({sorted(changed)}); use a new run_name")


def train(cfg: TrainConfig, device_name: str = "auto") -> None:
    """Train, evaluate and save one run, resuming from its checkpoint; a finished run is skipped."""
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "DONE").exists():
        print(f"[skip] {cfg.run_name} already finished ({run_dir / 'DONE'})")
        return

    import timm
    from timm.data import Mixup
    from timm.scheduler import CosineLRScheduler
    from timm.utils import ModelEmaV3

    # Set up the device and seed
    device = pick_device(device_name)
    amp_dtype = resolve_amp(cfg.amp, device)
    seed_everything(cfg.seed)

    # Load the data split
    fit, val, _ = load_val_split(cfg.data_root, REPO_ROOT / "metadata" / "val_split.json")
    train_samples = fit if cfg.use_val else fit + val
    if cfg.limit_train:
        train_samples = train_samples[: cfg.limit_train]
    if cfg.limit_eval:
        val = val[: cfg.limit_eval]

    # Build the model
    model, data_cfg = build_model(cfg)
    # Keep the starting weights on the CPU to log how far training moves them; resume rebuilds them identically.
    init_params = {n: p.detach().float().clone() for n, p in model.named_parameters()}
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    head_names = {n for n, p in model.named_parameters() if id(p) in head_ids}
    model.to(device)
    if cfg.channels_last:
        model.to(memory_format=torch.channels_last)

    # Build the data loaders
    train_tf = build_train_transform(cfg.img_size, data_cfg["mean"], data_cfg["std"], cfg.rrc_scale_min,
                                     cfg.augment, cfg.random_erasing)
    loader_gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        BirdDataset(cfg.data_root, train_samples, train_tf), batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda", drop_last=True,
        persistent_workers=cfg.num_workers > 0, generator=loader_gen,
    )
    # Seed 0 for every run, not cfg.seed: runs that differ only in seed are then measured on the
    # same images, so their train-val gaps compare like for like.
    train_eval = train_eval_subset(train_samples, cfg.train_eval_images, 0) if cfg.train_eval_images > 0 else []
    if cfg.limit_eval:
        train_eval = train_eval[: cfg.limit_eval]
    # Scored every epoch: val drives model selection, the training subset gives the like-for-like gap.
    eval_sets = {name: samples for name, samples in (("val", val if cfg.use_val else []), ("train_eval", train_eval))
                 if samples}
    eval_loaders = {name: eval_loader(cfg, samples, data_cfg) for name, samples in eval_sets.items()}
    val_loader = eval_loaders.get("val")
    epochs_dir = run_dir / "epochs"
    if cfg.save_epoch_logits and eval_sets:
        epochs_dir.mkdir(exist_ok=True)
        np.savez(epochs_dir / "index.npz", **{
            f"{name}_{key}": np.array([getattr(s, attr) for s in samples])
            for name, samples in eval_sets.items() for key, attr in (("labels", "label"), ("files", "filename"))})

    # Build the optimizer and scheduler
    updates_per_epoch = max(1, len(train_loader) // cfg.accum_steps)
    warmup_updates = int(cfg.warmup_epochs * updates_per_epoch)
    optimizer = build_optimizer(model, cfg)
    if cfg.lr_schedule == "plateau":
        scheduler = PlateauLR(optimizer, cfg.min_lr, warmup_updates, cfg.plateau_factor, cfg.plateau_patience)
    else:
        # Stepped per optimizer update rather than per epoch so the schedule is smooth even with
        # few updates per epoch (~60 at an effective batch of 64).
        scheduler = CosineLRScheduler(
            optimizer, t_initial=cfg.epochs * updates_per_epoch, lr_min=cfg.min_lr, warmup_t=warmup_updates,
            warmup_lr_init=cfg.min_lr, t_in_epochs=False,
        )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
    # A smoothed model that usually generalizes better than the last iterate, at no extra training cost.
    ema = ModelEmaV3(model, decay=cfg.ema_decay, use_warmup=True)
    mixup = None
    if cfg.mixup_prob > 0 and (cfg.mixup_alpha > 0 or cfg.cutmix_alpha > 0):
        mixup = Mixup(mixup_alpha=cfg.mixup_alpha, cutmix_alpha=cfg.cutmix_alpha, prob=cfg.mixup_prob,
                      label_smoothing=cfg.label_smoothing, num_classes=NUM_CLASSES)

    # Resume from a checkpoint or start fresh
    start_epoch, global_update, best_top1, elapsed_before = 0, 0, -1.0, 0.0
    best_epoch, epochs_trained = -1, 0
    last_path, trained_flag = run_dir / "last.pt", run_dir / "TRAINED"
    if (run_dir / "config.json").exists():
        check_resume_config(cfg, run_dir)
    if trained_flag.exists():
        # Training finished in an earlier session that died during final evaluation.
        start_epoch = cfg.epochs
        flag = json.loads(trained_flag.read_text())
        best_top1, elapsed_before = flag["best_top1"], flag["elapsed"]
        best_epoch, epochs_trained = flag.get("best_epoch", -1), flag.get("epochs_trained", cfg.epochs)
        print(f"[resume] {cfg.run_name}: training already complete, running final evaluation")
    elif last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng_state(state["rng"])
        loader_gen.set_state(state["loader_rng"])
        start_epoch, global_update = state["epoch"] + 1, state["global_update"]
        best_top1, elapsed_before = state["best_top1"], state["elapsed"]
        # Checkpoints from before early stopping existed restart the patience count here.
        best_epoch = state.get("best_epoch", state["epoch"])
        print(f"[resume] {cfg.run_name}: epoch {start_epoch}/{cfg.epochs}, best val top-1 {best_top1:.4f}")
    else:
        (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
        env = {"torch": torch.__version__, "timm": timm.__version__, "python": platform.python_version(),
               "device": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
               "amp": str(amp_dtype), "train_images": len(train_samples), "updates_per_epoch": updates_per_epoch,
               "val_images": len(eval_sets.get("val", [])), "train_eval_images": len(train_eval),
               "commit": code_commit(), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
               "gpus": torch.cuda.device_count(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        (run_dir / "env.json").write_text(json.dumps(env, indent=1))

    # Start the checkpoint writer and clocks
    writer = AsyncCheckpointWriter()
    last_save = time.monotonic()
    session_start = time.monotonic()

    def full_state(epoch: int) -> dict:
        """Return everything needed to continue after ``epoch`` exactly as if training had not stopped."""
        return {
            "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state(),
            "loader_rng": loader_gen.get_state(), "epoch": epoch, "global_update": global_update,
            "best_top1": best_top1, "best_epoch": best_epoch,
            "elapsed": elapsed_before + time.monotonic() - session_start,
        }

    try:
        for epoch in range(start_epoch, cfg.epochs):
            # Reset the epoch counters
            model.train()
            epoch_start = time.monotonic()
            loss_sum, correct, seen, skipped = 0.0, 0, 0, 0
            clock = {"wait": 0.0}
            steps: list[dict] = []           # one record per optimizer update, for steps.jsonl
            window = [0.0, 0, 0]             # loss sum, correct, images in the current accumulation window
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            # Train over the batches
            for step, (x, y, _) in enumerate(timed_batches(train_loader, clock)):
                if step >= updates_per_epoch * cfg.accum_steps:
                    break  # drop the incomplete accumulation window at the end of the epoch
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                if cfg.channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
                if mixup is not None:
                    x, y = mixup(x, y)
                ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
                with ctx:
                    logits = model(x)
                    loss = soft_target_loss(logits, y, cfg.label_smoothing)
                if not torch.isfinite(loss):
                    # An fp16 overflow guard: drop this micro-batch instead of poisoning the weights.
                    skipped += 1
                    continue
                scaler.scale(loss / cfg.accum_steps).backward()
                # Accuracy on the augmented batch as trained (soft Mixup targets count their top class).
                batch_correct = (logits.argmax(1) == (y if y.ndim == 1 else y.argmax(1))).sum().item()
                batch_loss = loss.item() * x.shape[0]
                loss_sum, correct, seen = loss_sum + batch_loss, correct + batch_correct, seen + x.shape[0]
                window = [window[0] + batch_loss, window[1] + batch_correct, window[2] + x.shape[0]]

                if (step + 1) % cfg.accum_steps == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad).item()
                    scale = scaler.get_scale()
                    scaler.step(optimizer)  # skipped automatically if gradients overflowed
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_update += 1
                    steps.append({"update": global_update, "epoch": epoch, "loss": window[0] / max(window[2], 1),
                                  "top1": window[1] / max(window[2], 1),
                                  "grad_norm": grad_norm if math.isfinite(grad_norm) else None,
                                  "lr_head": max(g["lr"] for g in optimizer.param_groups),
                                  "loss_scale": scale, "overflow": scaler.get_scale() < scale})
                    window = [0.0, 0, 0]
                    scheduler.step_update(global_update)
                    ema.update(model, step=global_update)

            # Build the epoch record
            train_time = time.monotonic() - epoch_start
            norms = [s["grad_norm"] for s in steps if s["grad_norm"] is not None]
            record = {"epoch": epoch, "train_loss": loss_sum / max(seen, 1), "train_top1": correct / max(seen, 1),
                      # With LLRD the head's group has the largest learning rate, the first layer the smallest.
                      "lr_head": max(g["lr"] for g in optimizer.param_groups),
                      "lr_min": min(g["lr"] for g in optimizer.param_groups),
                      "updates": global_update, "skipped_batches": skipped,
                      "overflow_steps": sum(s["overflow"] for s in steps), "loss_scale": scaler.get_scale(),
                      "grad_norm_mean": float(np.mean(norms)) if norms else None,
                      "grad_norm_max": max(norms) if norms else None,
                      "clipped_frac": sum(n > cfg.clip_grad for n in norms) / max(len(norms), 1),
                      "train_img_per_s": seen / max(train_time, 1e-6), "data_wait_s": clock["wait"],
                      "train_time_s": train_time}
            if device.type == "cuda":
                record["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                record["peak_mem_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1e9
            for prefix, net in (("", model), ("ema_", ema.module)):
                record.update({prefix + k: v for k, v in weight_stats(net, init_params, head_names).items()})

            # Evaluate the weights
            is_last = epoch == cfg.epochs - 1
            eval_start = time.monotonic()
            if eval_loaders and ((epoch + 1) % cfg.eval_every == 0 or is_last):
                # The EMA weights drive model selection; the raw weights show what the optimizer
                # is doing right now, which is where overfitting appears first.
                nets = (("ema_", ema.module),) + ((("", model),) if cfg.eval_raw_every_epoch or is_last else ())
                epoch_logits = {}
                for prefix, net in nets:
                    for name, loader in eval_loaders.items():
                        pred = predict_logits(net, loader, device, amp_dtype, cfg.channels_last)
                        metrics = eval_metrics(pred, cfg.label_smoothing)
                        record.update({f"{prefix}{name}_{k}": v for k, v in metrics.items()})
                        for view in ("center", "hflip"):
                            epoch_logits[f"{prefix}{name}_{view}"] = pred[view].astype(np.float16)
                if cfg.save_epoch_logits:
                    np.savez(epochs_dir / f"epoch_{epoch:03d}.npz", **epoch_logits)
            record["eval_time_s"] = time.monotonic() - eval_start

            # Select the best model and check early stopping; the event flags are logged every epoch so all
            # lines have the same keys.
            record.update(new_best=False, lr_cut=False, early_stop=False)
            if "ema_val_top1" in record:
                if record["ema_val_top1"] > best_top1:
                    # Selection uses the EMA weights on the val split only.
                    best_top1, best_epoch = record["ema_val_top1"], epoch
                    writer.save(half_state_dict(ema.module), run_dir / "best_ema.pt")
                    record["new_best"] = True
                if isinstance(scheduler, PlateauLR) and scheduler.step_epoch(record["new_best"]):
                    record["lr_cut"] = True
                # The selection state so far: what early stopping and the LR cuts react to.
                record.update(best_ema_val_top1=best_top1, best_epoch=best_epoch)
            # Early stopping only saves compute: best_ema.pt already holds the selected weights,
            # which later epochs can no longer change.
            stop_early = (not is_last and cfg.early_stop_patience > 0 and val_loader is not None
                          and epoch - best_epoch >= cfg.early_stop_patience)
            if stop_early:
                record["early_stop"] = True

            # Log the epoch
            record["epoch_time_s"] = time.monotonic() - epoch_start
            record["elapsed_h"] = (elapsed_before + time.monotonic() - session_start) / 3600
            log_line(run_dir, record)
            # A resumed session replays the epochs after its checkpoint, so both logs can hold an
            # epoch (update) twice; the later line is the one the run continued from.
            append_jsonl(run_dir / "steps.jsonl", steps)
            epochs_trained = epoch + 1
            if stop_early:
                print(f"[early-stop] {cfg.run_name}: no better val top-1 than {best_top1:.4f} (epoch {best_epoch}) "
                      f"for {cfg.early_stop_patience} epochs; stopping after epoch {epoch}", flush=True)
                break

            # Checkpoint on a timer, and stop while one more epoch would still fit in the session budget
            # (Kaggle kills jobs at its time limit and keeps no output); the last epoch needs no full
            # checkpoint because the TRAINED marker replaces it.
            out_of_time = cfg.time_budget_hours > 0 and (
                time.monotonic() - session_start + record["epoch_time_s"] > cfg.time_budget_hours * 3600)
            if not is_last and (out_of_time or time.monotonic() - last_save >= cfg.checkpoint_minutes * 60):
                writer.save(full_state(epoch), last_path)
                last_save = time.monotonic()
            if not is_last and out_of_time:
                writer.wait()
                print(f"[pause] {cfg.run_name}: {cfg.time_budget_hours} h session budget reached after "
                      f"epoch {epoch}; run the same config again to resume", flush=True)
                return
        writer.wait()

        # Mark training as finished
        final_weights = "best_ema.pt" if cfg.use_val else "final_ema.pt"
        if not trained_flag.exists():
            if not cfg.use_val:
                atomic_save(half_state_dict(ema.module), run_dir / "final_ema.pt")
            trained_flag.write_text(json.dumps({"best_top1": best_top1, "best_epoch": best_epoch,
                                                "epochs_trained": epochs_trained,
                                                "elapsed": elapsed_before + time.monotonic() - session_start}))

        # Evaluate the selected weights once on val (if any) and test, saving raw logits.
        state = torch.load(run_dir / final_weights, map_location="cpu", weights_only=True)
        ema.module.load_state_dict({k: v.float() for k, v in state.items()})
        report = write_predictions(ema.module, cfg, data_cfg, device, run_dir / "predictions.npz")
        summary = {"weights": final_weights, "best_val_top1": best_top1, "best_epoch": best_epoch,
                   "epochs_trained": epochs_trained, **report,
                   "train_hours": (elapsed_before + time.monotonic() - session_start) / 3600}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))

        # Write DONE before pruning so no crash window exists in which the run has neither a resumable
        # checkpoint nor a completion marker.
        (run_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        if cfg.prune_on_finish:
            last_path.unlink(missing_ok=True)  # optimizer state is only needed to resume
    finally:
        writer.close()


def main() -> None:
    """Parse the command line and train the run described by the config file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key=value (TOML value)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    train(load_config(args.config, args.overrides), args.device)


if __name__ == "__main__":
    main()
