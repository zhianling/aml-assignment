from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch

from birdcls.backbones import BACKBONES, get_backbone
from birdcls.config import REPO_ROOT
from birdcls.data import load_splits
from birdcls.features import extract_to_file
from birdcls.runtime import pick_device


def main() -> None:
    """Cache the frozen-backbone embeddings of every training and test image, one file per backbone."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["all"], help=f"keys from {sorted(BACKBONES)} or 'all'")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "cub")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "features")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=None,
                        help="override input resolution (fixed-resolution backbones are skipped)")
    parser.add_argument("--limit", type=int, default=0, help="embed only the first N images (smoke test)")
    parser.add_argument("--force", action="store_true", help="recompute even if the cache file exists")
    args = parser.parse_args()

    # Load the images and class names
    keys = sorted(BACKBONES) if args.models == ["all"] else args.models
    device = pick_device(args.device)
    train, test = load_splits(args.data_root)
    samples = train + test
    if args.limit:
        samples = samples[: args.limit]
    class_names = json.loads((REPO_ROOT / "metadata" / "classes.json").read_text())

    # Extract the features of each backbone
    for key in keys:
        spec = get_backbone(key)
        if args.img_size is not None and not spec.resizable:
            print(f"[skip] {key}: fixed input resolution", flush=True)
            continue
        # The file is <key>.npz at the native resolution and <key>@<size>.npz with --img-size.
        suffix = (f"@{args.img_size}" if args.img_size else "") + (f".limit{args.limit}" if args.limit else "")
        out_path = args.out_dir / f"{key}{suffix}.npz"
        if out_path.exists() and not args.force:
            print(f"[skip] {key}: {out_path.name} already cached", flush=True)
            continue
        print(f"[start] {key} ({spec.name}) on {device}", flush=True)
        try:
            meta = extract_to_file(spec, samples, args.data_root, class_names, out_path, device,
                                   img_size=args.img_size, batch_size=args.batch_size,
                                   num_workers=args.num_workers)
        except Exception:  # keep going so one broken backbone does not stop the rest
            traceback.print_exc()
            print(f"[fail] {key}", flush=True)
            continue
        finally:
            if device.type == "mps":
                torch.mps.empty_cache()  # free GPU memory before the next backbone loads
        print(f"[done] {key}: {meta['img_size']}px, {meta['params_m']}M params, "
              f"{meta['images_per_second']} img/s, {meta['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
