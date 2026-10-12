from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from birdcls.audit import nearest_neighbours
from birdcls.config import REPO_ROOT
from birdcls.features import load_features

# Cosine-similarity bands for the printed summary table.
BANDS = [0.94, 0.96, 0.975, 0.985, 0.99, 1.001]
THUMB_H = 150  # thumbnail height (px) on the contact sheet


def contact_sheet(pairs: list[tuple[str, str, float]], data_root: Path, out: Path) -> None:
    """Save a grid of test photos beside their nearest training photos, labeled with their cosine similarity."""
    w = THUMB_H * 4 // 3
    rows = (len(pairs) + 1) // 2
    canvas = Image.new("RGB", (4 * w + 30, rows * THUMB_H), "white")
    draw = ImageDraw.Draw(canvas)
    for k, (test_file, train_file, cos) in enumerate(pairs):
        x0, y0 = (k % 2) * (2 * w + 30), (k // 2) * THUMB_H
        for j, path in enumerate([data_root / "Test" / test_file, data_root / "Train" / train_file]):
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((w, THUMB_H - 14))
                canvas.paste(im, (x0 + j * w, y0 + 12))
        draw.text((x0 + 2, y0), f"{k}: {test_file} vs {train_file[:30]} {cos:.3f}", fill="black")
    canvas.save(out, quality=85)


def main() -> None:
    """Match each test image to its most similar training image by frozen-backbone cosine and save the matches."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=REPO_ROOT / "features" / "vit_l_dinov3@448.npz")
    parser.add_argument("--feature", default="pooled", help="which cached embedding to compare")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "cub")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "metadata" / "embedding_audit.json")
    parser.add_argument("--sheet", type=Path, help="also render a contact sheet of the pairs above --min-cos")
    parser.add_argument("--min-cos", type=float, default=0.985)
    args = parser.parse_args()

    # Find the nearest training image of each test image
    data = load_features(args.features)
    # Average each image's embedding over its original and mirrored view so mirrored copies still match.
    emb = data[f"feat_{args.feature}"].astype(np.float32).mean(axis=1)
    is_train = data["splits"] == "Train"
    idx, cos = nearest_neighbours(emb[~is_train], emb[is_train])
    test_files, test_labels = data["filenames"][~is_train], data["labels"][~is_train]
    train_files, train_labels = data["filenames"][is_train], data["labels"][is_train]
    same = test_labels == train_labels[idx]

    # Write the matches to JSON
    records = [{"file": str(f), "nearest_train": str(train_files[i]), "cosine": round(float(c), 4),
                "same_label": bool(s)} for f, i, c, s in zip(test_files, idx, cos, same)]
    args.out.write_text(json.dumps({
        "features": args.features.stem, "feature": args.feature,
        "note": "Nearest training image of every test image by cosine similarity of the flip-averaged "
                "embedding. The same-session threshold is set in dedup_review.json.",
        "test": records,
    }, indent=1))

    # Print the similarity bands
    print(f"{'cosine band':<16}{'test images':>12}{'same label':>12}")
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        m = (cos >= lo) & (cos < hi)
        share = f"{same[m].mean():.2f}" if m.any() else "-"
        print(f"[{lo:.3f}, {min(hi, 1):.3f}){m.sum():>12}{share:>12}")
    print(f"wrote {args.out}")

    # Save the contact sheet
    if args.sheet:
        order = [i for i in np.argsort(-cos) if cos[i] >= args.min_cos]
        contact_sheet([(str(test_files[i]), str(train_files[idx[i]]), float(cos[i])) for i in order],
                      args.data_root, args.sheet)
        print(f"wrote {args.sheet} ({len(order)} pairs)")


if __name__ == "__main__":
    main()
