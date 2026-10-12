"""Reading and writing the project's files: JSON Lines logs, CSV tables, NumPy archives and checksums."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSON Lines file (one JSON object per line) into a list of dicts, skipping blank lines."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict]:
    """Read a CSV file into one dict per row, keyed by the header; every value stays a string."""
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Write dicts as CSV rows under a header row of ``fieldnames`` (default: the first row's keys)."""
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    """Save arrays to the .npz file ``path``.

    The arrays go to a temporary file that then replaces ``path`` in one step, so a crash
    mid-write never leaves a truncated file behind.
    """
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    """SHA-256 checksum of a file, read in 1 MB blocks so multi-GB weight files need little memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
