from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from birdcls.config import REPO_ROOT, load_config

OUT = REPO_ROOT / "configs" / "ablation"

# One variant: (group, function from the base config to the settings it changes, description).
# Groups: remove = turn one regularizer off, scheme = change the fine-tuning scheme,
# add = add or strengthen a regularizer, hp = change the learning rate.
Variant = tuple[str, Callable[[dict], dict], str]

# Variants used by both studies.
NO_REG: Variant = ("remove", lambda b: {"label_smoothing": 0.0, "augment": "none", "random_erasing": 0.0,
                                        "drop_path": 0.0, "weight_decay": 0.0, "ema_decay": 0.0},
                   "all six regularisers off together; crop + flip kept")
MIXUP_CUTMIX: Variant = ("add", lambda b: {"mixup_alpha": 0.8, "cutmix_alpha": 1.0, "mixup_prob": 1.0},
                         "Mixup (alpha 0.8) or CutMix (alpha 1.0) on every batch")


@dataclass(frozen=True)
class Study:
    """One ablation study: a base config, extra control seeds, and variants that each change one setting of it."""
    key: str
    base: Path
    control_run: str        # the base config's own finished run, used as control seed 0
    variants: dict[str, Variant]
    report_stem: str       # output files: results/<report_stem>.csv and results/<report_stem>_curves.json
    control_seeds: tuple[int, ...] = (1, 2)  # extra control runs, to measure how much results vary by seed

    def control_name(self, seed: int) -> str:
        """Run name of the control run trained with ``seed``."""
        return f"abl_{self.key}__control__s{seed}"

    def variant_name(self, variant: str) -> str:
        """Run name of a variant (always trained with seed 0)."""
        return f"abl_{self.key}__{variant}__s0"


# The two studies: ConvNeXt-L, the best model, and ResNet-50, the model that overfits.
STUDIES: dict[str, Study] = {s.key: s for s in [
    Study("convnext", REPO_ROOT / "configs" / "convnext_l_inat21_448.toml", "convnext_l_inat21_448", {
        "no_reg": NO_REG,
        "no_lpft": ("scheme", lambda b: {"head_init": ""}, "random classifier instead of the linear-probe head"),
        "no_llrd": ("scheme", lambda b: {"layer_decay": 1.0}, "one LR for every layer"),
        "mixup_cutmix": MIXUP_CUTMIX,
        "no_ls": ("remove", lambda b: {"label_smoothing": 0.0}, "no label smoothing"),
        "no_augpolicy": ("remove", lambda b: {"augment": "none"}, "no TrivialAugment; crop + flip kept"),
        "no_erasing": ("remove", lambda b: {"random_erasing": 0.0}, "no random erasing"),
        "no_droppath": ("remove", lambda b: {"drop_path": 0.0}, "no stochastic depth"),
        "no_wd": ("remove", lambda b: {"weight_decay": 0.0}, "no weight decay"),
        # With decay 0 the EMA copy equals the current weights after every update, so the run keeps the raw weights.
        "no_ema": ("remove", lambda b: {"ema_decay": 0.0}, "no weight averaging"),
        "head_dropout": ("add", lambda b: {"drop_rate": 0.3}, "dropout 0.3 before the classifier"),
        "droppath_x2": ("add", lambda b: {"drop_path": round(2 * b["drop_path"], 4)}, "stochastic depth doubled"),
        "randaugment": ("add", lambda b: {"augment": "randaugment"},
                        "RandAugment (2 ops, magnitude 9) instead of TrivialAugment"),
        "lr_half": ("hp", lambda b: {"lr": b["lr"] / 2}, "peak LR halved"),
        "lr_double": ("hp", lambda b: {"lr": b["lr"] * 2}, "peak LR doubled"),
    }, report_stem="ablation"),
    Study("resnet", REPO_ROOT / "configs" / "resnet50_448.toml", "resnet50_448", {
        "no_reg": NO_REG,
        "mixup_cutmix": MIXUP_CUTMIX,
        # timm splits ResNet-50 into 18 learning-rate groups: 0.8 per group gives the first layer 0.02x
        # the head's learning rate, about the same range as ConvNeXt-L's 0.9 over its 38 groups.
        "llrd": ("scheme", lambda b: {"layer_decay": 0.8}, "layer-wise LR decay 0.8 instead of one LR for every layer"),
        "droppath_0.2": ("add", lambda b: {"drop_path": 0.2}, "stochastic depth 0.2 instead of 0.05"),
        "lr_third": ("hp", lambda b: {"lr": 1e-4}, "peak LR 1e-4 instead of 3e-4"),
    }, report_stem="ablation_resnet"),
]}


def build(study: Study) -> dict[str, dict]:
    """Return the config of every run in ``study`` as {run name: config}, control seeds first.

    Each config is the base config plus the variant's changes; a variant that changes nothing is an error.
    """
    base = tomllib.loads(study.base.read_text())
    defaults = load_config(study.base).to_dict()  # defaults filled in, to catch a variant that changes nothing
    runs: dict[str, dict] = {}
    for seed in study.control_seeds:
        name = study.control_name(seed)
        runs[name] = {**base, "run_name": name, "seed": seed,
                      "notes": f"ablation control: {study.control_run} recipe, seed {seed}"}
    for variant, (group, make, what) in study.variants.items():
        overrides = make(base)
        if all(defaults[k] == v for k, v in overrides.items()):
            raise SystemExit(f"{study.key}/{variant}: changes nothing in {study.base.name}")
        name = study.variant_name(variant)
        runs[name] = {**base, **overrides, "run_name": name, "seed": 0,
                      "notes": f"ablation {group}/{variant}: {what}"}
    return runs


def main() -> None:
    """Delete the old ablation configs and write configs/ablation/<run>.toml for every run of both studies."""
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.toml"):
        old.unlink()  # so configs of removed variants do not linger
    for study in STUDIES.values():
        runs = build(study)
        for name, raw in runs.items():
            path = OUT / f"{name}.toml"
            # json.dumps output is valid TOML for strings, numbers and booleans.
            path.write_text("".join(f"{k} = {json.dumps(v)}\n" for k, v in raw.items()))
            load_config(path)  # check that the written file is a valid config
        print(f"{study.key}: wrote {len(runs)} configs to {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
