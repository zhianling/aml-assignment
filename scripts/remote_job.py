from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = ("Train.zip", "Test.zip", "train.txt", "test.txt")
MIN_RUN_HOURS = 0.5  # a run started with less session time left would end before making real progress


def seed_data(seed: Path, dest: Path) -> None:
    """Copy the four data files from ``seed`` to ``dest`` for later sessions, skipping those already there."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        if not (dest / name).exists():
            print(f"[seed] {name} -> {dest}", flush=True)
            shutil.copyfile(seed / name, dest / f"{name}.partial")
            (dest / f"{name}.partial").rename(dest / name)


def find_source(src: Path, name: str) -> Path:
    """Return ``src/name``, or else its shallowest match below ``src`` (Kaggle may nest or unpack uploads)."""
    direct = src / name
    if direct.exists():
        return direct
    found = sorted(src.rglob(name), key=lambda p: len(p.parts))
    if not found:
        raise SystemExit(f"{name} not found under {src}")
    return found[0]


def stage_data(src: Path, dest: Path) -> None:
    """Copy or extract the Train and Test images to local disk, as a network mount is slow, and check the counts."""
    # Copy or extract each image folder
    dest.mkdir(parents=True, exist_ok=True)
    for folder in ("Train", "Test"):
        if (dest / folder).is_dir():
            continue
        partial = dest / f".{folder}.partial"
        shutil.rmtree(partial, ignore_errors=True)
        unpacked = [p for p in src.rglob(folder) if p.is_dir() and any(p.glob("*.jpg"))]
        if unpacked:
            shutil.copytree(unpacked[0], partial / folder)
        else:
            with zipfile.ZipFile(find_source(src, f"{folder}.zip")) as zf:
                zf.extractall(partial)  # the archives hold a top-level Train/ or Test/ folder
        # Rename only when extraction has finished, so an interrupted one is redone, not taken as complete.
        (partial / folder).rename(dest / folder)
        shutil.rmtree(partial)

    # Copy the txt files and check the image counts
    for name in ("train.txt", "test.txt"):
        shutil.copyfile(find_source(src, name), dest / name)
    for folder, txt in (("Train", "train.txt"), ("Test", "test.txt")):
        expected = sum(1 for line in (dest / txt).read_text().splitlines() if line.strip())
        found = sum(1 for _ in (dest / folder).glob("*.jpg"))
        if found != expected:
            raise SystemExit(f"{folder}: {found} images on disk but {expected} listed in {txt}")
        print(f"[data] {folder}: {found} images", flush=True)


def carry_over_runs(previous: Path, run_root: Path, keep_weights: frozenset[str] = frozenset()) -> None:
    """Copy the run folders of the previous session's output into ``run_root``, skipping .tmp files.

    Kaggle keeps only the latest output, so each session carries earlier runs forward. /kaggle/working holds about
    20 GB and a resuming run needs two 3 GB copies of last.pt while it saves, so finished runs are copied without
    their weights unless a zoom task needs them (``keep_weights``).
    """
    run_root.mkdir(parents=True, exist_ok=True)
    for run in sorted(p for p in previous.iterdir() if p.is_dir()):
        if not (run_root / run.name).exists():
            print(f"[resume] carrying over {run.name}", flush=True)
            skip = ["*.tmp"]
            if (run / "DONE").exists() and run.name not in keep_weights:
                skip.append("*.pt")
            shutil.copytree(run, run_root / run.name, ignore=shutil.ignore_patterns(*skip))


def run_streaming(cmd: list[str], env: dict | None = None, prefix: str = "") -> int:
    """Run ``cmd``, print its output line by line with ``prefix`` as it arrives, and return its exit code."""
    print(f"{prefix}$ {' '.join(cmd)}", flush=True)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                          env=env) as proc:
        for line in proc.stdout:
            print(f"{prefix}{line}", end="", flush=True)
    return proc.returncode


def task_command(kind: str, target: str, args: argparse.Namespace, left: float | None) -> list[str] | None:
    """Return the command for a "train" or "zoom" task, or None if done; ``left`` is the session hours left."""
    if kind == "zoom":
        run = args.run_root / target
        if (run / "zoom_raw.npz").exists():
            return None  # scored in an earlier session
        return [sys.executable, "scripts/cam_zoom.py", "--run", str(run)]
    overrides = [a for kv in args.overrides for a in ("--set", kv)]
    if left is not None:
        # Limit the run to the time left so it saves a checkpoint and exits before the platform ends the session.
        overrides += ["--set", f"time_budget_hours={left:.3f}"]
    entry = "scripts/benchmark_step.py" if args.benchmark_only else "-m birdcls.train"
    return [sys.executable, *entry.split(), "--config", target, *overrides]


def run_lane(gpu: int | None, tasks: list[tuple[str, str]], args: argparse.Namespace, start: float,
             failed: list) -> None:
    """Run the ``(kind, target)`` tasks in order, on GPU ``gpu`` if given; skip a task with under MIN_RUN_HOURS left."""
    env = dict(os.environ)
    prefix = ""
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        prefix = f"[gpu{gpu}] "
    for kind, target in tasks:
        left = None
        if args.session_hours > 0 and not args.benchmark_only:
            left = args.session_hours - (time.monotonic() - start) / 3600
            if left < MIN_RUN_HOURS:
                print(f"{prefix}[skip] {target}: {left:.2f} h of the session left; next session continues", flush=True)
                continue
        cmd = task_command(kind, target, args, left)
        if cmd is None:
            print(f"{prefix}[skip] {kind} {target}: already done", flush=True)
            continue
        # Record a failure (such as running out of GPU memory) and go on with the next task.
        if run_streaming(cmd, env, prefix) != 0:
            failed.append(target)


def print_summary(run_root: Path) -> None:
    """Print one line per run folder: its state and its test top-1 and mean class accuracy."""
    for run in sorted(p for p in run_root.glob("*") if p.is_dir()):
        state = ("DONE" if (run / "DONE").exists() else "resumable" if (run / "last.pt").exists()
                 else "trained" if (run / "TRAINED").exists() else "started")
        test = {}
        if (run / "summary.json").exists():
            test = json.loads((run / "summary.json").read_text()).get("test", {})
        print(f"{run.name:<32} {state:<10} test top-1={test.get('top1', float('nan')):.4f} "
              f"mean-class={test.get('mean_class_acc', float('nan')):.4f}", flush=True)


def main() -> None:
    """Run training configs and zoom scoring on a hosted GPU machine such as a Kaggle session."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="*", help="config files, run in this order")
    parser.add_argument("--data-src", type=Path, required=True, help="folder with the zips and txt files")
    parser.add_argument("--run-root", type=Path, required=True, help="persistent folder for run directories")
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/cub"), help="local staging folder")
    parser.add_argument("--seed-data", type=Path, help="copy the data files from here into --data-src first")
    parser.add_argument("--resume-from", type=Path, help="earlier session's run root to carry over")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key=value for every config")
    parser.add_argument("--benchmark-only", action="store_true", help="measure speed/memory instead of training")
    parser.add_argument("--session-hours", type=float, default=0.0,
                        help=">0: stay under this session length; each run gets the time that is left")
    parser.add_argument("--gpus", type=int, default=1, help=">1: train that many configs in parallel, one per GPU")
    parser.add_argument("--zoom", nargs="*", default=[], help="finished run names to score with CAM-guided zoom")
    args = parser.parse_args()
    start, failed = time.monotonic(), []  # the session clock starts now; data staging counts toward the limit

    # Prepare the data and earlier runs
    if args.seed_data:
        seed_data(args.seed_data, args.data_src)
    if args.resume_from and args.resume_from.is_dir():
        carry_over_runs(args.resume_from, args.run_root, frozenset(args.zoom))
    if not args.benchmark_only:
        stage_data(args.data_src, args.data_dir)
    os.environ["BIRD_DATA_ROOT"] = str(args.data_dir)
    os.environ["BIRD_RUN_ROOT"] = str(args.run_root)
    os.chdir(REPO_ROOT)

    # List zoom scoring first: it is short, so training that runs out of session time cannot crowd it out.
    tasks = [("zoom", name) for name in args.zoom] + [("train", cfg) for cfg in args.configs]
    if not tasks:
        parser.error("nothing to do: give configs and/or --zoom run names")

    # Run the tasks
    if args.gpus > 1:
        # Deal the tasks out to the GPUs in turn; longest configs first, so the GPUs finish at about the same time.
        lanes = [tasks[i::args.gpus] for i in range(args.gpus)]
        threads = [threading.Thread(target=run_lane, args=(i, lane, args, start, failed))
                   for i, lane in enumerate(lanes) if lane]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        run_lane(None, tasks, args, start, failed)

    # Report the results
    if not args.benchmark_only:
        print_summary(args.run_root)
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
