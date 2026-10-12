from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from birdcls.config import load_config  # noqa: E402

# Names of the Kaggle datasets (data, code) and the kernel created under the user's account.
DATA_SLUG, CODE_SLUG, KERNEL_SLUG = "cos30082-cub200-2010", "cos30082-bird-code", "cos30082-bird-training"
DATA_FILES = ("Train.zip", "Test.zip", "train.txt", "test.txt")  # uploaded as the data dataset
# Paths the kernel runs from. The upload is built from the last commit, so they must have no uncommitted changes.
CODE_PATHS = ("src", "scripts", "configs", "heads", "pyproject.toml")
# Kaggle's image ships a CUDA build of these; reinstalling them from PyPI could break it.
PREINSTALLED = {"torch", "torchvision"}
# Words in Kaggle CLI errors that mean the dataset or kernel does not exist.
MISSING_MARKERS = ("403", "404", "not found", "cannot access", "denied")
SESSION_HOURS = 11.0  # Kaggle stops GPU sessions at 12 h; the last hour is left for evaluation and saving


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a Kaggle CLI command and capture its output."""
    return subprocess.run(["kaggle", *args], capture_output=True, text=True, check=check)


def git(*args: str) -> str:
    """Run a git command in the repository and return its output without surrounding whitespace."""
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def detect_user() -> str:
    """Return the username the Kaggle CLI is logged in with."""
    match = re.search(r"username:\s*(\S+)", kaggle("config", "view", check=False).stdout)
    if not match or match.group(1) == "None":
        raise SystemExit("could not read the Kaggle username: run `kaggle auth login` or pass --user")
    return match.group(1)


def status(kind: str, ref: str, attempts: int = 4) -> str | None:
    """Return a Kaggle dataset's or kernel's status text (lower case), or None if it does not exist."""
    # Only the missing markers mean "missing"; retry other errors, as a wrong "missing" would start a second kernel
    # next to a running one or recreate an existing dataset.
    for attempt in range(attempts):
        result = kaggle(kind, "status", ref, check=False)
        text = (result.stdout + result.stderr).lower()
        if any(marker in text for marker in MISSING_MARKERS):
            return None
        if result.returncode == 0:
            return text.strip()
        if attempt < attempts - 1:
            time.sleep(15)
    raise SystemExit(f"kaggle {kind} status {ref} kept failing:\n{text.strip()}")


def pip_requirements(pyproject: Path) -> list[str]:
    """List the project's dependencies from pyproject.toml, minus those Kaggle's image already has."""
    deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    return [d for d in deps if re.split(r"[<>=!~ \[]", d, maxsplit=1)[0] not in PREINSTALLED]


def job_spec(configs: list[str], commit: str, session_hours: float, overrides: list[str],
             zoom: list[str] | None = None) -> dict:
    """Build the job file the kernel reads: commit, configs, zoom runs, session hours, overrides, pip packages."""
    return {"commit": commit, "configs": configs, "zoom": zoom or [], "session_hours": session_hours,
            "overrides": overrides, "pip": pip_requirements(REPO_ROOT / "pyproject.toml")}


def kernel_metadata(user: str, resume: bool, machine_shape: str) -> dict:
    """Build kernel-metadata.json: a private GPU script kernel that reads the data and code datasets."""
    ref = f"{user}/{KERNEL_SLUG}"
    return {
        "id": ref, "title": KERNEL_SLUG, "code_file": "kaggle_kernel.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": machine_shape,
        "dataset_sources": [f"{user}/{DATA_SLUG}", f"{user}/{CODE_SLUG}"],
        # Attach the kernel's previous output, so runs resume from the last session's checkpoints.
        "kernel_sources": [ref] if resume else [],
        "competition_sources": [], "model_sources": [],
    }


def build_code_zip(dest: Path, job: dict) -> None:
    """Zip the last commit's files (git HEAD) into ``dest`` and add the job file."""
    git("archive", "--format=zip", "-o", str(dest), "HEAD")
    with zipfile.ZipFile(dest, "a") as zf:
        zf.writestr("kaggle_job.json", json.dumps(job, indent=1))


def served_job(ref: str) -> dict | None:
    """Download the job file the code dataset serves now (what a kernel started now would run), or None."""
    with tempfile.TemporaryDirectory() as tmp:
        kaggle("datasets", "download", ref, "-f", "kaggle_job.json", "-p", tmp, "-o", "-q", check=False)
        path = Path(tmp) / "kaggle_job.json"
        if (Path(tmp) / "kaggle_job.json.zip").exists():  # Kaggle may send a single file zipped
            with zipfile.ZipFile(Path(tmp) / "kaggle_job.json.zip") as zf:
                zf.extractall(tmp)
        return json.loads(path.read_text()) if path.exists() else None


def upload_dataset(folder: Path, ref: str, title: str, message: str, job: dict | None = None) -> None:
    """Create or update a Kaggle dataset from ``folder`` and wait until it serves the new version (and ``job``)."""
    # Write the metadata and upload
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": ref, "licenses": [{"name": "unknown"}]}, indent=1))
    args = ("version", "-p", str(folder), "-m", message) if status("datasets", ref) else ("create", "-p", str(folder))
    print(f"[kaggle] datasets {args[0]} {ref}", flush=True)
    result = kaggle("datasets", *args, check=False)
    if result.returncode != 0:
        raise SystemExit(f"kaggle datasets {args[0]} failed:\n{result.stdout}{result.stderr}")

    # Wait for "ready" and this job file, as Kaggle can serve the previous version for minutes and a kernel started
    # then would run the old job.
    for _ in range(90):
        state = status("datasets", ref) or ""
        if "error" in state:
            raise SystemExit(f"{ref}: processing failed ({state})")
        if "ready" in state and (job is None or served_job(ref) == job):
            return
        time.sleep(10)
    raise SystemExit(f"{ref}: this version is still not served after 15 minutes; push the kernel again later")


def main() -> None:
    """Upload the data (first push only) and the committed code, then start the Kaggle training kernel.

    Refuses to push with uncommitted changes under CODE_PATHS or while the kernel is running.
    Pushing again resumes the unfinished runs from the last session's output.
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="*", type=Path, help="config files, longest run first")
    parser.add_argument("--zoom", nargs="*", default=[],
                        help="finished run names (from earlier sessions) to score with CAM-guided zoom")
    parser.add_argument("--data-src", type=Path, help="folder with the zips and txt files (first push only)")
    parser.add_argument("--user", help="Kaggle username (default: from the CLI's login)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key=value for every config")
    parser.add_argument("--session-hours", type=float, default=SESSION_HOURS)
    parser.add_argument("--machine-shape", default="NvidiaTeslaT4", help="Kaggle accelerator (T4 x2 by default)")
    parser.add_argument("--fresh", action="store_true",
                        help="do not attach the previous session's output (after that session was invalidated)")
    parser.add_argument("--dry-run", type=Path, help="write the upload folders here instead of pushing")
    args = parser.parse_args()
    if not args.configs and not args.zoom:
        parser.error("nothing to run: give configs and/or --zoom run names")
    if args.zoom and args.fresh:
        parser.error("--zoom scores runs from the previous session, which --fresh leaves out")

    # Check the configs, the repository and the kernel state
    configs = []
    for path in args.configs:
        load_config(path, args.overrides)  # fail on a typo here, not on Kaggle
        configs.append(path.resolve().relative_to(REPO_ROOT).as_posix())
    if git("status", "--porcelain", "--untracked-files=no", "--", *CODE_PATHS):
        raise SystemExit(f"commit changes under {', '.join(CODE_PATHS)} first: Kaggle runs HEAD")
    commit = git("rev-parse", "--short", "HEAD")
    user = args.user or ("dry-run-user" if args.dry_run else detect_user())
    kernel_ref = f"{user}/{KERNEL_SLUG}"
    kernel_state = None if args.dry_run else status("kernels", kernel_ref)
    if kernel_state and ("running" in kernel_state or "queued" in kernel_state):
        raise SystemExit(f"{kernel_ref} is still {kernel_state}; wait for it to finish before pushing again")

    # Build the upload folders
    with tempfile.TemporaryDirectory() as tmp:
        root = args.dry_run or Path(tmp)
        data_dir, code_dir, kernel_dir = root / "data", root / "code", root / "kernel"
        for d in (data_dir, code_dir, kernel_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Upload the data on the first push
        data_ref = f"{user}/{DATA_SLUG}"
        if args.dry_run or status("datasets", data_ref) is None:
            if args.data_src is None:
                raise SystemExit(f"{data_ref} does not exist yet: pass --data-src <Assignment dir>")
            for name in DATA_FILES:
                shutil.copyfile(args.data_src / name, data_dir / name)
            if not args.dry_run:
                upload_dataset(data_dir, data_ref, "COS30082 CUB-200-2010 subset", "data")

        # Build the code zip and kernel metadata
        job = job_spec(configs, commit, args.session_hours, args.overrides, args.zoom)
        build_code_zip(code_dir / "code.zip", job)
        shutil.copyfile(REPO_ROOT / "scripts" / "kaggle_kernel.py", kernel_dir / "kaggle_kernel.py")
        resume = kernel_state is not None and not args.fresh
        metadata = kernel_metadata(user, resume=resume, machine_shape=args.machine_shape)
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=1))
        if args.dry_run:
            print(f"[dry-run] upload folders written to {root}")
            return

        # Upload the code and push the kernel
        upload_dataset(code_dir, f"{user}/{CODE_SLUG}", "COS30082 bird code", f"commit {commit}", job=job)
        result = kaggle("kernels", "push", "-p", str(kernel_dir), check=False)
        print(result.stdout + result.stderr, flush=True)
        if result.returncode != 0:
            raise SystemExit("kaggle kernels push failed")
        print(f"[kaggle] pushed {kernel_ref} at {commit} (resume={metadata['kernel_sources'] != []})")


if __name__ == "__main__":
    main()
