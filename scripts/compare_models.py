from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from birdcls.analysis import PredictionSource, apply_temperature, ensemble_probs, greedy_ensemble, load_source
from birdcls.audit import flagged_test_images
from birdcls.config import NUM_CLASSES, REPO_ROOT
from birdcls.fileio import write_csv
from birdcls.metrics import expected_calibration_error, mcnemar_exact, stratified_bootstrap_ci, summarise, top1_accuracy


def collect_sources(probe_dir: Path, runs_dirs: list[Path], include: str | None) -> list[PredictionSource]:
    """Load the val-calibrated predictions of every probe and run as sources named "<kind>:<name>".

    Kinds: LP (linear probe), FT (fine-tuned network) and ZOOM (CAM-guided zoom). ``include`` is a name regex.
    """
    sources = [load_source(f"LP:{p.stem}", p, calibrate=True) for p in sorted(probe_dir.glob("*.npz"))]
    for runs_dir in (d for d in runs_dirs if d.is_dir()):
        # Run folders named lp_* are linear probes packaged by make_probe_run.py, so they count as LP.
        sources += [load_source(f"{'LP' if p.parent.name.startswith('lp_') else 'FT'}:{p.parent.name}", p,
                                calibrate=True)
                    for p in sorted(runs_dir.glob("*/predictions.npz"))]
        sources += [load_source(f"ZOOM:{p.parent.name}", p, calibrate=True)
                    for p in sorted(runs_dir.glob("*/predictions_zoom.npz"))]
    # A <name>_full run has no val split to calibrate on, so it reuses the temperature of <name>, its val-split twin.
    by_name = {s.name: s for s in sources}
    for s in sources:
        twin = by_name.get(s.name.removesuffix("_full"))
        if s.name.endswith("_full") and not s.has_val and twin is not None and twin.raw_probs:
            s.raw_probs, s.temperature = dict(s.probs), twin.temperature
            s.probs = {split: apply_temperature(p, s.temperature) for split, p in s.probs.items()}
    if include:
        sources = [s for s in sources if re.search(include, s.name)]
    return sources


def metrics_row(name: str, kind: str, val: tuple | None, test: tuple, clean_mask: np.ndarray,
                temperature: float | str = "", raw_test_probs: np.ndarray | None = None) -> dict:
    """Build one comparison-table row for a model.

    ``val`` and ``test`` are (probabilities, labels) pairs; ``val`` is None without a val split.
    ``clean_mask`` selects the test images that are not copies of training images.
    """
    probs, labels = test
    pred = probs.argmax(1)
    ci = stratified_bootstrap_ci(labels, pred, NUM_CLASSES, n_boot=1000)
    s = summarise(labels, probs, NUM_CLASSES)
    row = {
        "name": name, "kind": kind,
        "val_top1": "", "val_mean_class_acc": "",
        "test_top1": s["top1"], "test_top1_lo": ci["top1"][0], "test_top1_hi": ci["top1"][1],
        "test_mean_class_acc": s["mean_class_acc"],
        "test_mca_lo": ci["mean_class_acc"][0], "test_mca_hi": ci["mean_class_acc"][1],
        "test_top5": s["top5"],
        "test_top1_clean": top1_accuracy(labels[clean_mask], pred[clean_mask]),
        "test_ece": expected_calibration_error(probs, labels),
        "test_ece_raw": expected_calibration_error(raw_test_probs, labels) if raw_test_probs is not None else "",
        "temperature": temperature,
    }
    if val is not None:
        v = summarise(val[1], val[0], NUM_CLASSES)
        row["val_top1"], row["val_mean_class_acc"] = v["top1"], v["mean_class_acc"]
    return row


def main() -> None:
    """Score every model, choose an ensemble on val, write comparison.csv and ensemble.json, and print the table."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, default=REPO_ROOT / "features" / "preds")
    parser.add_argument("--runs-dir", type=Path, default=Path(os.environ.get("BIRD_RUN_ROOT", REPO_ROOT / "runs")))
    parser.add_argument("--include", help="regex filter on source names")
    parser.add_argument("--max-members", type=int, default=4)
    parser.add_argument("--use-full-twins", action="store_true",
                        help="after choosing ensemble members on val, score run X with its '<X>_full' "
                             "twin (same recipe retrained on all training images) where available; the "
                             "twins are archived runs, read from <runs-dir>/archive")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()

    # Load the sources; --use-full-twins also loads the archived all-data retrains.
    runs_dirs = [args.runs_dir] + ([args.runs_dir / "archive"] if args.use_full_twins else [])
    sources = collect_sources(args.probe_dir, runs_dirs, args.include)
    if not sources:
        raise SystemExit("No prediction files found")
    by_name = {s.name: s for s in sources}

    # Mark the clean test subset
    flags = flagged_test_images(REPO_ROOT / "metadata")
    test_files = sources[0].files["test"]
    clean_mask = ~np.isin(test_files, list(flags))

    # Score every model
    rows = []
    for s in sources:
        val = (s.probs["val"], s.labels["val"]) if s.has_val else None
        rows.append(metrics_row(s.name, s.name.split(":")[0], val, (s.probs["test"], s.labels["test"]), clean_mask,
                                s.temperature if s.raw_probs else "", s.raw_probs.get("test")))

    # Build a greedy ensemble on val log-loss; zoom runs are left out, as their settings were tuned on val.
    members = greedy_ensemble([s for s in sources if s.has_val and not s.name.startswith("ZOOM:")], args.max_members)
    scored = members
    if args.use_full_twins:  # score each member's all-data retrain on test instead
        scored = [by_name.get(f"{m.name}_full", m) for m in members]
    ens_val = (ensemble_probs(members, "val"), members[0].labels["val"])
    ens_test = (ensemble_probs(scored, "test"), scored[0].labels["test"])
    ens_name = "ENS:" + "+".join(m.name for m in scored)
    rows.append(metrics_row(ens_name, "ENS", ens_val, ens_test, clean_mask))

    # Run McNemar tests of every model against the single model with the best val top-1
    singles = [r for r in rows if r["kind"] != "ENS" and r["val_top1"] != ""]
    ref_name = max(singles, key=lambda r: r["val_top1"])["name"]
    ref_correct = by_name[ref_name].probs["test"].argmax(1) == by_name[ref_name].labels["test"]
    for r in rows:
        probs = ens_test[0] if r["kind"] == "ENS" else by_name[r["name"]].probs["test"]
        test = mcnemar_exact(ref_correct, probs.argmax(1) == by_name[ref_name].labels["test"])
        r["mcnemar_vs_ref_p"] = test["p_value"] if r["name"] != ref_name else ""
        r["only_ref_right"], r["only_this_right"] = test["only_a"], test["only_b"]

    # Write the table and ensemble
    rows.sort(key=lambda r: (r["val_top1"] == "", -(r["val_top1"] or 0)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "comparison.csv"
    write_csv(out, [{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()} for r in rows])
    (args.out_dir / "ensemble.json").write_text(json.dumps({
        "selected_on_val": [m.name for m in members], "scored_on_test": [m.name for m in scored],
        "reference_single_model": ref_name, "clean_subset_size": int(clean_mask.sum()),
    }, indent=1))

    # Print the table
    print(f"reference (best val) single model: {ref_name}")
    print(f"{'name':<52} {'val':>6} {'test':>6} {'mca':>6} {'clean':>6} {'ece':>6} {'raw':>6} {'T':>5}  p(McNemar)")
    for r in rows:
        val = f"{r['val_top1']:.4f}" if r["val_top1"] != "" else "   -  "
        p = f"{r['mcnemar_vs_ref_p']:.3g}" if r["mcnemar_vs_ref_p"] != "" else "ref"
        raw = f"{r['test_ece_raw']:.4f}" if r["test_ece_raw"] != "" else "   -  "
        t = f"{r['temperature']:.2f}" if r["temperature"] != "" else "  -  "
        print(f"{r['name'][:52]:<52} {val:>6} {r['test_top1']:.4f} {r['test_mean_class_acc']:.4f} "
              f"{r['test_top1_clean']:.4f} {r['test_ece']:.4f} {raw:>6} {t:>5}  {p}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
