from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from birdcls.audit import flagged_test_images
from birdcls.config import REPO_ROOT
from birdcls.features import load_features
from birdcls.fileio import read_csv, write_csv
from birdcls.metrics import expected_calibration_error, summarise, top1_accuracy
from birdcls.probe import fit_linear_probe, knn_scores, view_averaged_probs, zero_shot_scores
from birdcls.probs import nll

# L2 penalty strengths tried for each probe; the one with the best val top-1 is kept.
WEIGHT_DECAYS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
# Columns of the main results table (--out): one row per feature cache and feature variant.
FIELDS = [
    "cache", "backbone", "img_size", "family", "in_domain", "pretraining", "params_m", "variant",
    "dim", "best_wd", "val_top1", "val_mean_class_acc", "test_top1", "test_mean_class_acc",
    "test_top5", "test_top1_clean", "knn_val_top1", "knn_test_top1", "zeroshot_test_top1",
    "zeroshot_test_mean_class_acc", "extract_img_per_s",
]
# Columns of probe_curves.csv: the scores after each optimizer pass over the training features.
CURVE_FIELDS = [
    "cache", "variant", "weight_decay", "fitted_on", "pass", "objective_ce", "train_top1", "train_mean_class_acc",
    "train_top5", "train_loss", "train_ece", "val_top1", "val_mean_class_acc", "val_top5", "val_loss", "val_ece",
]
# Columns of probe_wd_grid.csv: the fit and val scores for every weight decay tried.
GRID_FIELDS = [
    "cache", "variant", "weight_decay", "chosen", "fit_top1", "fit_mean_class_acc", "fit_top5", "fit_loss",
    "fit_ece", "val_top1", "val_mean_class_acc", "val_top5", "val_loss", "val_ece",
]


def feature_variants(data: dict) -> dict[str, np.ndarray]:
    """Return the feature sets to fit probes on, by name.

    ``pooled`` is always there. A ViT cache adds the other token summary, as DINOv2 does for its probes: the mean
    patch token (``pooled+patch``) or the CLS token (``cls``, ``pooled+cls``) when the model pools by averaging.
    """
    variants = {"pooled": data["feat_pooled"].astype(np.float32)}
    if "feat_cls" in data:
        variants["cls"] = data["feat_cls"].astype(np.float32)
        variants["pooled+cls"] = np.concatenate([variants["pooled"], variants["cls"]], axis=2)
    if "feat_patch_mean" in data:
        variants["pooled+patch"] = np.concatenate(
            [variants["pooled"], data["feat_patch_mean"].astype(np.float32)], axis=2)
    return variants


def flatten_views(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reshape features [N images, V views, D] to [N*V, D] and repeat the labels, so each view is its own sample."""
    n, v, d = x.shape
    return x.reshape(n * v, d), np.repeat(y, v)


def split_scores(prefix: str, labels: np.ndarray, probs: np.ndarray, num_classes: int) -> dict[str, float]:
    """Score one split's predictions: accuracy, loss (NLL) and calibration error (ECE), keys prefixed by ``prefix``."""
    scores = summarise(labels, probs, num_classes)
    scores["loss"] = nll(probs, labels)
    scores["ece"] = expected_calibration_error(probs, labels)
    return {f"{prefix}_{k}": v for k, v in scores.items()}


def pass_recorder(rows: list[dict], key: dict, views: int, train_labels: np.ndarray,
                  val: tuple[np.ndarray, np.ndarray] | None, num_classes: int):
    """Return an ``on_pass`` callback for fit_linear_probe that adds a row to ``rows`` after each training pass.

    ``objective_ce`` is the L-BFGS loss per view, without the weight penalty. The ``train_`` and ``val_`` scores
    average each image's two views, as val and test are scored.
    """
    passes = itertools.count(1)

    def on_pass(probe, train_logits, objective_ce):
        """Score the probe after one pass and append the row."""
        probs = torch.softmax(train_logits, 1).view(len(train_labels), views, -1).mean(1).numpy()
        row = {**key, "pass": next(passes), "objective_ce": objective_ce,
               **split_scores("train", train_labels, probs, num_classes)}
        if val is not None:
            row.update(split_scores("val", val[1], view_averaged_probs(probe, val[0]), num_classes))
        rows.append(row)
    return on_pass


def evaluate_cache(path: Path, val_names: set[str], clean_mask_fn, num_classes: int,
                   heads_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Fit and score linear probes and k-NN on one feature cache; save the heads under ``heads_dir``.

    Returns the result rows, the weight-decay grid rows and the per-pass curve rows.
    """
    # Load the cache and build the split masks
    data = load_features(path)
    meta = data["meta"]
    labels = data["labels"].astype(np.int64)
    is_train = data["splits"] == "Train"
    is_val = is_train & np.isin(data["filenames"], list(val_names))
    is_fit, is_test = is_train & ~is_val, ~is_train
    clean = clean_mask_fn(data["filenames"]) & is_test

    rows, grid, curves = [], [], []
    for variant, x in feature_variants(data).items():
        # Fit a probe on the fit split for every weight decay and keep the one with the best val top-1
        best = None
        fx, fy = flatten_views(x[is_fit], labels[is_fit])
        for wd in WEIGHT_DECAYS:
            key = {"cache": path.stem, "variant": variant, "weight_decay": wd, "fitted_on": "fit"}
            probe = fit_linear_probe(fx, fy, num_classes, wd, on_pass=pass_recorder(
                curves, key, x.shape[1], labels[is_fit], (x[is_val], labels[is_val]), num_classes))
            val_probs = view_averaged_probs(probe, x[is_val])
            val_metrics = summarise(labels[is_val], val_probs, num_classes)
            grid.append({"cache": path.stem, "variant": variant, "weight_decay": wd,
                         **split_scores("fit", labels[is_fit], view_averaged_probs(probe, x[is_fit]), num_classes),
                         **split_scores("val", labels[is_val], val_probs, num_classes)})
            if best is None or val_metrics["top1"] > best[1]["top1"]:
                best = (wd, val_metrics, probe)

        wd, val_metrics, val_probe = best
        for g in grid[-len(WEIGHT_DECAYS):]:
            g["chosen"] = g["weight_decay"] == wd

        # Refit on all training images (fit + val) with the chosen weight decay and score once on test
        tx, ty = flatten_views(x[is_train], labels[is_train])
        key = {"cache": path.stem, "variant": variant, "weight_decay": wd, "fitted_on": "train"}
        probe = fit_linear_probe(tx, ty, num_classes, wd, on_pass=pass_recorder(
            curves, key, x.shape[1], labels[is_train], None, num_classes))  # no val scores; val is training data now
        test_probs = view_averaged_probs(probe, x[is_test])
        test_metrics = summarise(labels[is_test], test_probs, num_classes)
        clean_pred = view_averaged_probs(probe, x[clean]).argmax(axis=1)

        # Score the k-NN baseline on the embedding averaged over the two views
        mean_x = x.mean(axis=1)
        knn_val = knn_scores(mean_x[is_fit], labels[is_fit], mean_x[is_val], num_classes)
        knn_test = knn_scores(mean_x[is_train], labels[is_train], mean_x[is_test], num_classes)

        # Build the result row
        row = {
            "cache": path.stem, "backbone": meta["key"], "img_size": meta["img_size"],
            "family": meta["family"], "in_domain": meta["in_domain"], "pretraining": meta["pretraining"],
            "params_m": meta["params_m"], "variant": variant, "dim": x.shape[2], "best_wd": wd,
            "val_top1": val_metrics["top1"], "val_mean_class_acc": val_metrics["mean_class_acc"],
            "test_top1": test_metrics["top1"], "test_mean_class_acc": test_metrics["mean_class_acc"],
            "test_top5": test_metrics["top5"],
            "test_top1_clean": top1_accuracy(labels[clean], clean_pred),
            "knn_val_top1": top1_accuracy(labels[is_val], knn_val.argmax(1)),
            "knn_test_top1": top1_accuracy(labels[is_test], knn_test.argmax(1)),
            "zeroshot_test_top1": "", "zeroshot_test_mean_class_acc": "",
            "extract_img_per_s": meta["images_per_second"],
        }
        if "text" in data and variant == "pooled":
            zs = summarise(labels[is_test], zero_shot_scores(mean_x[is_test], data["text"]), num_classes)
            row["zeroshot_test_top1"] = zs["top1"]
            row["zeroshot_test_mean_class_acc"] = zs["mean_class_acc"]
        rows.append(row)

        # Save each probe as a plain linear head (standardization folded in) so fine-tuning can start from it (LP-FT)
        stem = f"{path.stem}__{variant.replace('+', '_')}"
        heads_dir.mkdir(parents=True, exist_ok=True)
        for fitted_on, model, suffix in (("train", probe, ""), ("fit", val_probe, "__fit")):
            w, b = model.fold_standardisation()
            torch.save({"weight": w, "bias": b, "variant": variant, "weight_decay": wd, "fitted_on": fitted_on,
                        "meta": meta}, heads_dir / f"{stem}{suffix}.pt")

        # Save logits in a fine-tuned run's predictions.npz format so probes can join ensembles and tests.
        # <stem> holds the fit-split probe's val and test logits; <stem>_full the refit probe's test logits only.
        preds_dir = heads_dir.parent / "preds"
        preds_dir.mkdir(parents=True, exist_ok=True)
        for suffix, model, splits in (("", val_probe, (("val", is_val), ("test", is_test))),
                                      ("_full", probe, (("test", is_test),))):
            arrays = {}
            for split, mask in splits:
                arrays[f"{split}_center"] = model.logits(x[mask][:, 0])
                arrays[f"{split}_hflip"] = model.logits(x[mask][:, 1])
                arrays[f"{split}_labels"] = labels[mask]
                arrays[f"{split}_files"] = data["filenames"][mask]
            np.savez(preds_dir / f"{stem}{suffix}.npz", **arrays)
        print(f"  {path.stem:<28} {variant:<13} wd={wd:<7g} val={val_metrics['top1']:.4f} "
              f"test={test_metrics['top1']:.4f} mca={test_metrics['mean_class_acc']:.4f} "
              f"knn={row['knn_test_top1']:.4f}", flush=True)
    return rows, grid, curves


def main() -> None:
    """Fit and score linear probes and k-NN on every feature cache; write the results, grid and curve tables."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, default=REPO_ROOT / "features")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "phase1_probes.csv")
    parser.add_argument("--only", nargs="*", help="cache stems to (re)evaluate; others keep their old rows")
    args = parser.parse_args()

    # Load the metadata
    classes = json.loads((REPO_ROOT / "metadata" / "classes.json").read_text())
    val_names = set(json.loads((REPO_ROOT / "metadata" / "val_split.json").read_text())["val"])
    flags = flagged_test_images(REPO_ROOT / "metadata")
    # Clean test images have no copy in the training set.
    clean_mask_fn = lambda names: ~np.isin(names, list(flags))  # noqa: E731

    # Pick the caches and keep the rows of the others
    caches = sorted(p for p in args.features_dir.glob("*.npz")
                    if ".limit" not in p.name and not p.name.endswith(".tmp.npz"))
    if args.only:
        caches = [p for p in caches if p.stem in set(args.only)]

    grid_out, curves_out = args.out.with_name("probe_wd_grid.csv"), args.out.with_name("probe_curves.csv")
    previous, previous_grid, previous_curves = [], [], []
    for out, kept in ((args.out, previous), (grid_out, previous_grid), (curves_out, previous_curves)):
        if out.exists():
            kept.extend(r for r in read_csv(out) if r["cache"] not in {p.stem for p in caches})

    # Evaluate each cache
    rows, grid, curves = [], [], []
    for path in caches:
        cache_rows, cache_grid, cache_curves = evaluate_cache(path, val_names, clean_mask_fn, len(classes),
                                                              args.features_dir / "heads")
        rows.extend(cache_rows)
        grid.extend(cache_grid)
        curves.extend(cache_curves)

    # Write the tables
    all_rows = previous + rows
    all_rows.sort(key=lambda r: float(r["val_top1"]), reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, [{k: (f"{v:.4f}" if isinstance(v, float) and k != "best_wd" else v) for k, v in r.items()}
                         for r in all_rows], FIELDS)
    write_csv(grid_out, [{k: (f"{v:.4f}" if isinstance(v, float) and k != "weight_decay" else v)
                          for k, v in r.items()} for r in previous_grid + grid], GRID_FIELDS)
    # Curves keep 6 significant digits because late passes change the scores only slightly.
    write_csv(curves_out, [{k: (f"{v:.6g}" if isinstance(v, float) and k != "weight_decay" else v)
                            for k, v in r.items()} for r in previous_curves + curves], CURVE_FIELDS)
    print(f"wrote {len(all_rows)} rows to {args.out}, the weight-decay sweep to {grid_out} and "
          f"{len(previous_curves) + len(curves)} training passes to {curves_out}")


if __name__ == "__main__":
    main()
