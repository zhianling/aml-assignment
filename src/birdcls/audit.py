"""Data-quality checks: exact and near-duplicate detection."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


def file_md5(path: Path) -> str:
    """Return the md5 hex digest of a file's bytes."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Return a 64-bit difference hash that is robust to resizing and JPEG re-compression.

    The image is shrunk to (hash_size + 1) x hash_size grayscale pixels and each bit records whether a
    pixel is brighter than its right-hand neighbor. Copies of one photo land within a few bits of each
    other, while unrelated photos differ in roughly half of the 64 bits.
    """
    small = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    px = np.asarray(small, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def group_exact_duplicates(keys: Sequence[str]) -> list[list[int]]:
    """Return index groups (size >= 2) whose keys (e.g. md5 digests) are identical."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        buckets[k].append(i)
    return [idx for idx in buckets.values() if len(idx) > 1]


def near_duplicate_pairs(
    hashes: Iterable[int], max_distance: int, chunk: int = 512
) -> list[tuple[int, int, int]]:
    """Return all pairs (i, j, hamming) with i < j whose 64-bit hashes differ in at most ``max_distance`` bits.

    The comparison runs blockwise so the full N x N distance matrix never has to live in memory.
    """
    h = np.fromiter(hashes, dtype=np.uint64)
    pairs: list[tuple[int, int, int]] = []
    for start in range(0, len(h), chunk):
        block = h[start : start + chunk, None] ^ h[None, :]
        dist = np.bitwise_count(block)
        rows, cols = np.nonzero(dist <= max_distance)
        for r, c in zip(rows.tolist(), cols.tolist()):
            i = start + r
            if i < c:
                pairs.append((i, c, int(dist[r, c])))
    return pairs


def nearest_neighbours(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the index of, and cosine similarity to, the most similar ``reference`` row for every ``query`` row."""
    q = query / np.linalg.norm(query, axis=1, keepdims=True)
    r = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    sim = q @ r.T
    idx = sim.argmax(axis=1)
    return idx, sim[np.arange(len(q)), idx]


def flagged_test_images(metadata_dir: Path) -> dict[str, str]:
    """Return the test filenames that have a copy in train, mapped to the reason they are flagged.

    ``exact-conflict`` is byte-identical to a training image with a different label, so no single
    prediction can match both annotations. ``near-same-label`` is a burst-shot near-copy (dHash) of a
    training image with the same label, which makes the image easier than a genuinely unseen photo.
    ``same-session`` is another shot of the same bird on the same perch, found in embedding space
    because the pose differs too much for dHash (``session-conflict`` if the labels differ).
    The official test set is never modified; the flags define an extra clean subset on which accuracy
    is also reported, to show how much the duplicates move the numbers.
    """
    # Load the audit, the review and the rejected pairs
    metadata_dir = Path(metadata_dir)
    audit = json.loads((metadata_dir / "audit.json").read_text())
    review = json.loads((metadata_dir / "dedup_review.json").read_text())
    rejected = {frozenset(p) for p in review.get("false_positive_pairs", [])}
    flags: dict[str, str] = {}

    # Flag exact duplicates shared by train and test
    for group in audit["exact_duplicates"]["groups"]:
        if {r["split"] for r in group} == {"Train", "Test"}:
            conflict = len({r["label"] for r in group}) > 1
            for r in group:
                if r["split"] == "Test":
                    flags[r["filename"]] = "exact-conflict" if conflict else "exact-same-label"

    # Flag near duplicates the review did not reject
    for pair in audit["near_duplicates"]["pairs"]:
        a, b = pair["a"], pair["b"]
        if {a["split"], b["split"]} != {"Train", "Test"}:
            continue
        if frozenset((a["filename"], b["filename"])) in rejected:
            continue
        test = a if a["split"] == "Test" else b
        reason = "near-same-label" if a["label"] == b["label"] else "near-conflict"
        flags.setdefault(test["filename"], reason)

    # Flag same-session shots once the review has set a threshold
    threshold = review.get("same_session_threshold")
    if threshold is not None:
        embedding = json.loads((metadata_dir / "embedding_audit.json").read_text())
        for r in embedding["test"]:
            if r["cosine"] >= threshold:
                flags.setdefault(r["file"], "same-session" if r["same_label"] else "session-conflict")
    return flags
