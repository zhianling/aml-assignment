from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import torch

# Kaggle mounts the attached datasets read-only under INPUT and saves WORK as the kernel's output.
INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
CODE_DIR, DATA_DIR = Path("/tmp/code"), Path("/tmp/cub")  # local working copies of the code and images
JOB_FILE = "kaggle_job.json"  # written into the code upload by kaggle_push.py: what this session runs


def shallowest(paths) -> Path | None:
    """Return the path with the fewest folder levels (ties broken by name), or None if there is none."""
    found = sorted(paths, key=lambda p: (len(p.parts), str(p)))
    return found[0] if found else None


def locate_inputs(root: Path) -> tuple[Path | None, Path | None, Path | None]:
    """Find the code (folder or zip), the data folder and the previous session's run folder; each is None if missing."""
    job = shallowest(root.rglob(JOB_FILE))
    code = job.parent if job else shallowest(p for p in root.rglob("*.zip") if _holds_job(p))
    data = shallowest(p.parent for p in root.rglob("test.txt") if (p.parent / "train.txt").exists())
    previous = shallowest(p for p in root.rglob("runs") if p.is_dir() and any(p.glob("*/config.json")))
    return code, data, previous


def _holds_job(archive: Path) -> bool:
    """Return True if ``archive`` is a zip file that contains the job file."""
    try:
        with zipfile.ZipFile(archive) as zf:
            return JOB_FILE in zf.namelist()
    except zipfile.BadZipFile:
        return False


def stage_code(code: Path, dest: Path) -> None:
    """Copy or unzip the code to a writable ``dest`` (/kaggle/input is read-only), replacing any earlier copy."""
    shutil.rmtree(dest, ignore_errors=True)
    if code.is_file():
        with zipfile.ZipFile(code) as zf:
            zf.extractall(dest)
    else:
        shutil.copytree(code, dest)


def main() -> None:
    """Run this session's job: install the missing packages, then run remote_job.py on the job's configs."""
    # Locate and stage the inputs
    start = time.monotonic()
    code, data, previous = locate_inputs(INPUT)
    if code is None or data is None:
        raise SystemExit(f"inputs missing under {INPUT}: code={code} data={data}")
    stage_code(code, CODE_DIR)
    job = json.loads((CODE_DIR / JOB_FILE).read_text())
    print(f"[kaggle] commit {job['commit']}: train {job['configs']}, zoom {job.get('zoom', [])}", flush=True)
    print(f"[kaggle] data={data} previous runs={previous}", flush=True)

    # Install only the packages Kaggle's image lacks; its own CUDA build of torch is kept.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *job["pip"]], check=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)

    # Check for a GPU
    gpus = torch.cuda.device_count()
    if gpus == 0:
        raise SystemExit("no GPU attached: check the kernel's accelerator setting")

    # Run remote_job.py on the time left after setup
    left = job["session_hours"] - (time.monotonic() - start) / 3600
    cmd = [sys.executable, str(CODE_DIR / "scripts" / "remote_job.py"), *job["configs"],
           "--data-src", str(data), "--data-dir", str(DATA_DIR), "--run-root", str(WORK / "runs"),
           "--session-hours", f"{left:.3f}", "--gpus", str(gpus),
           *(["--resume-from", str(previous)] if previous else []),
           *(["--zoom", *job["zoom"]] if job.get("zoom") else []),
           *[a for kv in job["overrides"] for a in ("--set", kv)]]
    print("$", " ".join(cmd), flush=True)
    # The staged code has no git history, so BIRD_COMMIT passes the commit hash for the runs to record.
    returncode = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": str(CODE_DIR / "src"),
                                          "BIRD_COMMIT": job["commit"]}).returncode

    # Exit normally even if a config failed: Kaggle may discard a failed kernel's output, and the next session needs
    # the checkpoints of the configs that did train.
    (WORK / "job_status.json").write_text(json.dumps({"commit": job["commit"], "returncode": returncode}))
    print(f"[kaggle] remote_job exit code {returncode}", flush=True)


if __name__ == "__main__":
    main()
