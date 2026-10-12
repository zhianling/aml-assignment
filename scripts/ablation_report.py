from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from birdcls.analysis import apply_temperature, fit_temperature
from birdcls.config import REPO_ROOT
from birdcls.fileio import read_jsonl, write_csv
from birdcls.metrics import expected_calibration_error, mcnemar_exact
from birdcls.probs import log_probs, nll, tta_probs

import make_ablation_configs as abl

# Per-epoch metrics copied into the curves file, for the EMA weights (the ones kept) and the raw weights.
CURVE_KEYS = ("ema_val_top1", "ema_train_eval_top1", "ema_val_loss", "ema_train_eval_loss",
              "val_top1", "train_eval_top1", "val_loss", "train_eval_loss", "train_loss", "train_top1", "lr_head")


def run_table(study: abl.Study) -> list[tuple[str, str, str, int]]:
    """List every planned run of ``study`` as (run name, variant, group, seed)."""
    rows = [(study.control_run, "control", "control", 0)]
    rows += [(study.control_name(s), "control", "control", s) for s in study.control_seeds]
    rows += [(study.variant_name(v), v, group, 0) for v, (group, _, _) in study.variants.items()]
    return rows


def score_run(run_dir: Path) -> tuple[dict, dict, dict]:
    """Score one finished run: its table row, its per-epoch curves, and which val and test images it got right."""
    # Load the summary, epoch log and predictions
    summary = json.loads((run_dir / "summary.json").read_text())
    epochs = [r for r in read_jsonl(run_dir / "metrics.jsonl") if "epoch" in r]
    pred = np.load(run_dir / "predictions.npz")

    # Read the scores and overfitting gaps
    best = summary["best_epoch"]
    at_best = next(r for r in epochs if r["epoch"] == best)
    gap_loss = [r["ema_val_loss"] - r["ema_train_eval_loss"] for r in epochs]
    row = {
        "epochs_trained": summary["epochs_trained"], "best_epoch": best,
        "train_hours": round(summary["train_hours"], 3),
        "val_top1": summary["val"]["top1"], "val_mca": summary["val"]["mean_class_acc"],
        "test_top1": summary["test"]["top1"], "test_mca": summary["test"]["mean_class_acc"],
        # train_eval is a fixed subset of training images, scored exactly like val.
        "train_eval_top1_at_best": at_best["ema_train_eval_top1"],
        "gap_top1_at_best": at_best["ema_train_eval_top1"] - at_best["ema_val_top1"],
        "gap_loss_at_best": at_best["ema_val_loss"] - at_best["ema_train_eval_loss"],
        "gap_loss_final": gap_loss[-1], "gap_loss_max": max(gap_loss),
        "min_val_loss_epoch": min(epochs, key=lambda r: r["ema_val_loss"])["epoch"],
    }

    # Fit one temperature on val and apply it to test; it changes the confidence, never the predicted class.
    probs = {s: tta_probs(pred[f"{s}_center"], pred[f"{s}_hflip"]) for s in ("val", "test")}
    labels = {s: pred[f"{s}_labels"] for s in ("val", "test")}
    temperature = fit_temperature(log_probs(probs["val"]), labels["val"])
    test_cal = apply_temperature(probs["test"], temperature)
    row.update(val_nll=nll(probs["val"], labels["val"]), temperature=temperature,
               test_nll_cal=nll(test_cal, labels["test"]),
               test_ece_cal=expected_calibration_error(test_cal, labels["test"]))

    # Collect the curves and which images it got right
    curves = {k: [r.get(k) for r in epochs] for k in CURVE_KEYS}
    curves["epoch"] = [r["epoch"] for r in epochs]
    correct = {s: probs[s].argmax(axis=1) == labels[s] for s in ("val", "test")}
    return row, curves, correct


def build_report(run_root: Path, study: abl.Study) -> tuple[list[dict], dict]:
    """Score every finished run of ``study`` against the control; return the table rows and per-epoch curves.

    ``dval`` and ``dtest`` are changes from the control mean, ``dval_sd`` is ``dval`` in control
    standard deviations, and the McNemar tests compare each run with the control's seed-0 run.
    """
    # Score the finished runs
    rows, curves, correct = [], {}, {}
    for name, variant, group, seed in run_table(study):
        run_dir = run_root / name
        if not (run_dir / "summary.json").exists():
            continue  # not finished yet
        fields, curves[name], correct[name] = score_run(run_dir)
        rows.append({"run": name, "variant": variant, "group": group, "seed": seed, **fields})

    # Get the control mean and its spread across seeds
    controls = [r for r in rows if r["variant"] == "control"]
    if not controls or controls[0]["run"] != study.control_run:
        raise SystemExit(f"{run_root / study.control_run}: control seed 0 is missing; every comparison needs it")
    ctrl_val = np.array([r["val_top1"] for r in controls])
    ctrl_test = np.array([r["test_top1"] for r in controls])
    val_sd = float(ctrl_val.std(ddof=1)) if len(controls) > 1 else float("nan")

    # Compare each run with the control
    for r in rows:
        r["dval"] = r["val_top1"] - float(ctrl_val.mean())
        r["dtest"] = r["test_top1"] - float(ctrl_test.mean())
        r["control_val_sd"] = val_sd
        r["dval_sd"] = r["dval"] / val_sd if val_sd > 0 else float("nan")
        # Control seed 0 shares the seed and data order, so only the variant's change differs.
        for split in ("val", "test"):
            test = mcnemar_exact(correct[r["run"]][split], correct[study.control_run][split])
            r[f"{split}_only_run"], r[f"{split}_only_control"] = test["only_a"], test["only_b"]
            r[f"{split}_mcnemar_p"] = test["p_value"]
    return rows, curves


def main() -> None:
    """Write results/<stem>.csv and results/<stem>_curves.json for one study, then print a summary."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=sorted(abl.STUDIES), default="convnext")
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()

    # Score the study
    study = abl.STUDIES[args.study]
    rows, curves = build_report(args.run_root, study)

    # Write the table and curves
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{study.report_stem}.csv"
    write_csv(csv_path, [{k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()} for r in rows])
    (args.out_dir / f"{study.report_stem}_curves.json").write_text(json.dumps(curves))

    # Print a summary
    missing = len(run_table(study)) - len(rows)
    print(f"scored {len(rows)} runs ({missing} not finished yet) -> {csv_path}")
    for r in rows:
        print(f"{r['run']:<36} val {r['val_top1']:.4f} ({r['dval_sd']:+.1f} SD)  test {r['test_top1']:.4f}  "
              f"gap {r['gap_top1_at_best']:+.3f}  McNemar test p={r['test_mcnemar_p']:.3f}")


if __name__ == "__main__":
    main()
