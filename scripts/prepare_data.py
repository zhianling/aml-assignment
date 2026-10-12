from __future__ import annotations

import argparse
import json
import shutil
import statistics
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

from birdcls.audit import dhash, file_md5, group_exact_duplicates, near_duplicate_pairs
from birdcls.config import REPO_ROOT
from birdcls.data import Sample, derive_class_names, load_splits, stratified_holdout

METADATA = REPO_ROOT / "metadata"

# Hold out 20% of each class's training images (3-6 per class) for val, leaving 13-25 per class to train on.
# The fixed seed gives the same split every time.
VAL_FRACTION = 0.2
VAL_SEED = 20260926
# Images whose dHash fingerprints differ in at most this many of 64 bits count as the same photo;
# unrelated photos differ in about 32 bits and re-saved copies in fewer than 5.
NEAR_DUP_BITS = 6


def unpack(src: Path, data_root: Path) -> None:
    """Extract Train.zip and Test.zip into ``data_root`` unless already extracted, and copy train.txt and test.txt."""
    data_root.mkdir(parents=True, exist_ok=True)
    for archive, folder in (("Train.zip", "Train"), ("Test.zip", "Test")):
        if (data_root / folder).is_dir() and any((data_root / folder).iterdir()):
            print(f"[unpack] {folder}/ already present, skipping")
            continue
        print(f"[unpack] extracting {archive}")
        with zipfile.ZipFile(src / archive) as zf:
            zf.extractall(data_root)
    for txt in ("train.txt", "test.txt"):
        shutil.copy2(src / txt, data_root / txt)


def check_files(data_root: Path, samples: list[Sample], folder: str) -> None:
    """Stop with an error unless every image listed for ``folder`` exists and every image in it is listed."""
    listed = {s.filename for s in samples}
    on_disk = {p.name for p in (data_root / folder).glob("*.jpg")}
    missing, unlisted = listed - on_disk, on_disk - listed
    if missing or unlisted:
        raise SystemExit(f"{folder}: {len(missing)} missing files, {len(unlisted)} unannotated files")


def describe(values: list[int]) -> dict:
    """Return the min, median, max and mean of ``values``."""
    return {"min": min(values), "median": statistics.median(values), "max": max(values),
            "mean": round(statistics.fmean(values), 2)}


def main() -> None:
    """Unpack and check the dataset, split off the val set, audit the images and write metadata/*.json."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="folder holding Train.zip, Test.zip, *.txt")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "cub")
    args = parser.parse_args()

    # Unpack and check the files
    unpack(args.src, args.data_root)
    train, test = load_splits(args.data_root)
    check_files(args.data_root, train, "Train")
    check_files(args.data_root, test, "Test")

    # Split off the val set
    classes = derive_class_names(train)
    fit, val = stratified_holdout(train, VAL_FRACTION, VAL_SEED, min_per_class=3)

    # Read the size, color mode, perceptual hash (dHash) and file checksum (MD5) of every image
    everything = train + test
    widths, heights, hashes, digests, modes = [], [], [], [], Counter()
    for s in everything:
        path = s.path(args.data_root)
        with Image.open(path) as img:
            widths.append(img.width)
            heights.append(img.height)
            modes[img.mode] += 1
            hashes.append(dhash(img))
        digests.append(file_md5(path))
    short_sides = [min(w, h) for w, h in zip(widths, heights)]

    # Find the exact and near duplicates
    def record(i: int) -> dict:
        """Describe image ``i`` for the audit: split, filename, label and species."""
        s = everything[i]
        return {"split": s.split, "filename": s.filename, "label": s.label, "species": classes[s.label]}

    exact_groups = group_exact_duplicates(digests)
    exact = [[record(i) for i in group] for group in exact_groups]
    group_of = {i: g for g, members in enumerate(exact_groups) for i in members}
    near = [
        {"distance": d, "a": record(i), "b": record(j)}
        for i, j, d in near_duplicate_pairs(hashes, NEAR_DUP_BITS)
        # byte-identical copies are already listed as exact duplicates
        if not (i in group_of and group_of[i] == group_of.get(j))
    ]

    def summarise(pairs: list[list[dict]]) -> dict:
        """Count the duplicate groups, the groups spanning train and test, and those with conflicting labels."""
        cross = sum(1 for g in pairs if {r["split"] for r in g} == {"Train", "Test"})
        conflicting = sum(1 for g in pairs if len({r["label"] for r in g}) > 1)
        return {"groups": len(pairs), "train_test_groups": cross, "label_conflicts": conflicting}

    # Assemble the audit
    per_class_train = Counter(s.label for s in train)
    per_class_test = Counter(s.label for s in test)
    audit = {
        "num_train": len(train),
        "num_test": len(test),
        "num_classes": len(classes),
        "train_images_per_class": describe(list(per_class_train.values())),
        "test_images_per_class": describe(list(per_class_test.values())),
        "val_split": {"fraction": VAL_FRACTION, "seed": VAL_SEED, "fit": len(fit), "val": len(val)},
        "image_width": describe(widths),
        "image_height": describe(heights),
        "short_side": describe(short_sides),
        "colour_modes": dict(modes),
        "exact_duplicates": {"summary": summarise(exact), "groups": exact},
        "near_duplicates": {
            "max_dhash_bits": NEAR_DUP_BITS,
            "summary": summarise([[p["a"], p["b"]] for p in near]),
            "pairs": sorted(near, key=lambda p: p["distance"]),
        },
    }

    # Write the metadata and print a summary
    METADATA.mkdir(exist_ok=True)
    (METADATA / "classes.json").write_text(json.dumps(classes, indent=1))
    (METADATA / "val_split.json").write_text(
        json.dumps({"fraction": VAL_FRACTION, "seed": VAL_SEED, "val": sorted(s.filename for s in val)}, indent=1)
    )
    (METADATA / "audit.json").write_text(json.dumps(audit, indent=1))

    print(f"[audit] train={len(train)} test={len(test)} classes={len(classes)} fit={len(fit)} val={len(val)}")
    print(f"[audit] short side {audit['short_side']}, modes {dict(modes)}")
    print(f"[audit] exact duplicates {audit['exact_duplicates']['summary']}")
    print(f"[audit] near duplicates  {audit['near_duplicates']['summary']}")


if __name__ == "__main__":
    main()
