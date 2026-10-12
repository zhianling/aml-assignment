from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from birdcls.checkpoint import atomic_save, half_state_dict
from birdcls.config import REPO_ROOT, TrainConfig
from birdcls.models import build_model


def main() -> None:
    """Package a linear-probe head as a run folder so it can be scored like a fine-tuned run."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=Path, required=True, help="folded probe head from run_probes.py")
    parser.add_argument("--runs-dir", type=Path, default=Path(os.environ.get("BIRD_RUN_ROOT", REPO_ROOT / "runs")))
    args = parser.parse_args()

    # Load the probe and name the run
    state = torch.load(args.head, map_location="cpu", weights_only=False)
    meta = state["meta"]
    # Only single-feature probes (pooled or CLS) fit a standard timm classifier; concatenated ones cannot be packaged.
    pooling = {"pooled": "", "cls": "token"}
    if state["variant"] not in pooling:
        raise SystemExit(f"{args.head.name}: '{state['variant']}' probes cannot be packaged as a model")
    run_name = "lp_" + args.head.stem.split("__")[0].replace("@", "_")
    if state["variant"] != "pooled":
        run_name += "_" + state["variant"]

    # Build the model with the probe as its head
    cfg = TrainConfig(
        run_name=run_name, backbone=meta["key"], img_size=meta["img_size"], epochs=0,
        global_pool=pooling[state["variant"]],
        head_init=str(args.head.resolve().relative_to(REPO_ROOT)), drop_path=0.0,
        # A head fitted on all training images has seen the val images, so the run gets no val scores.
        use_val=state.get("fitted_on", "train") == "fit",
        notes="Frozen pretrained backbone + linear probe (no fine-tuning).",
    )
    model, data_cfg = build_model(cfg)  # loads pretrained weights and copies the probe into the head
    # Crop exactly as when the probe's features were extracted, so the head sees the inputs it was fitted on.
    cfg.eval_crop_pct = float(data_cfg["crop_pct"])

    # Write the run folder
    run_dir = args.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    atomic_save(half_state_dict(model), run_dir / "best_ema.pt")
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
