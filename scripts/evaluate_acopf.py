"""Evaluate trained AC-OPF DNNs and report test-set metrics.

Standalone results collection: reloads the self-contained checkpoints written by
train_acopf.py and computes all metrics on the held-out test CSVs, with no
dependency on the training session. Re-running is idempotent — a config's rows
are replaced, not duplicated.

Per (case, relaxation) it writes three CSVs under results/acopf/:
  - eval_summary.csv   : ensemble metrics, mean + 95% CI for
        relax_cost, nn_ape_relax (|pred-relax|/relax %),
        nn_ape_local (|pred-local|/local %), gap_pct ((local-relax)/relax %).
  - fold_metrics.csv   : the same metric means for each individual fold model.
  - confusion_sweep.csv: optimality-certification confusion matrix across a
        sweep of relative (vs relaxed cost) and absolute thresholds.

Usage:
    python scripts/evaluate_acopf.py                       # full grid
    python scripts/evaluate_acopf.py --cases case9 --relax socp
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nn.training import load_checkpoint, predict_denorm
from nn.metrics import mean_ci, optimality_confusion_matrix, overprediction_summary

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"    / "acopf-hpc"   # SAVIO-generated CSVs
MODELS_DIR       = PROJECT_ROOT / "models"  / "acopf"
RESULTS_DIR      = PROJECT_ROOT / "results" / "acopf"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS = ("Cost", "Exact", "LocalCost")
ALL_CASES = ["case9", "case14", "case39", "case89pegase",
             "case118", "case300", "case1354pegase", "case2869pegase"]
ALL_RELAX = ["socp", "chordal_sdp"]

# Metric key -> human label, for the printed table.
_METRICS = ["relax_cost", "nn_ape_relax", "nn_ape_local", "gap_pct"]


def _test_csv(data_dir, case, relax, n):
    return data_dir / f"test_{n}_{relax}_{case}.csv"


def _ckpt_path(case, relax, n, fold):
    return MODELS_DIR / f"dnn_{relax}_{case}_n{n}_fold{fold}.pt"


def _per_sample_metrics(pred, relax, local):
    """The four per-sample arrays (percentages where noted)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "relax_cost":   relax,
            "nn_ape_relax": 100.0 * np.abs(pred - relax) / relax,
            "nn_ape_local": 100.0 * np.abs(pred - local) / local,
            "gap_pct":      100.0 * (local - relax) / relax,
        }


def _upsert_csv(path, new_df, key_cols):
    """Append new_df to path, replacing any existing rows with matching keys."""
    if path.exists():
        old = pd.read_csv(path)
        keys = new_df[key_cols].apply(tuple, axis=1)
        mask = ~old[key_cols].apply(tuple, axis=1).isin(set(keys))
        out = pd.concat([old[mask], new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(path, index=False)


def evaluate_config(case, relax, args):
    print(f"\n{'='*70}\n  EVAL  {case.upper()}  |  {relax.upper()}\n{'='*70}", flush=True)

    # ── reload fold checkpoints ────────────────────────────────────────────────
    models, scalers = [], []
    for fold in range(args.folds):
        p = _ckpt_path(case, relax, args.n_train, fold)
        if not p.exists():
            continue
        m, s, _ = load_checkpoint(p)
        models.append(m); scalers.append(s)
    if not models:
        print(f"  SKIP: no checkpoints for n={args.n_train} (expected {args.folds} folds)", flush=True)
        return None
    if len(models) < args.folds:
        print(f"  WARNING: found {len(models)}/{args.folds} fold checkpoints", flush=True)

    # ── load + clean test data ─────────────────────────────────────────────────
    csv_path = _test_csv(args.data_dir, case, relax, args.n_test)
    if not csv_path.exists():
        print(f"  SKIP: missing {csv_path.name}", flush=True)
        return None
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in LABEL_COLS]
    for col in ("Cost", "LocalCost"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_raw = len(df)
    df = df[np.isfinite(df["Cost"]) & np.isfinite(df["LocalCost"])].reset_index(drop=True)
    n_drop = n_raw - len(df)
    if n_drop:
        print(f"  dropped {n_drop}/{n_raw} test rows with NaN Cost/LocalCost", flush=True)

    X  = df[feat_cols].values.astype(np.float64)
    relax_cost = df["Cost"].values.astype(np.float64)
    local = df["LocalCost"].values.astype(np.float64)

    # ── predictions: per fold + ensemble ───────────────────────────────────────
    fold_preds = [predict_denorm(m, X, s) for m, s in zip(models, scalers)]
    ens_pred = np.mean(fold_preds, axis=0)

    # ── ensemble metrics (mean + 95% CI) ───────────────────────────────────────
    ens = _per_sample_metrics(ens_pred, relax_cost, local)
    summary_row = {"case": case, "relaxation": relax,
                   "n_train": args.n_train, "n_test_used": len(df),
                   "n_folds": len(models)}
    for k in _METRICS:
        ci = mean_ci(ens[k])
        summary_row[f"{k}_mean"]  = ci["mean"]
        summary_row[f"{k}_ci_lo"] = ci["ci_lower"]
        summary_row[f"{k}_ci_hi"] = ci["ci_upper"]

    # Worst-case OVER-prediction of the relaxation value — the certification-
    # critical tail: bounds how far a "certified optimal" solution's true gap can
    # exceed the NN's relative tolerance.
    over = overprediction_summary(ens_pred, relax_cost, q=0.95)
    summary_row["max_overpred_pct"] = over["max_overpred_pct"]
    summary_row["q95_overpred_pct"] = over["q_overpred_pct"]

    # ── per-fold metric means (fold variance) ──────────────────────────────────
    fold_rows = []
    for fold, fp in enumerate(fold_preds):
        fm = _per_sample_metrics(fp, relax_cost, local)
        row = {"case": case, "relaxation": relax, "n_train": args.n_train, "fold": fold}
        for k in _METRICS:
            row[f"{k}_mean"] = mean_ci(fm[k])["mean"]
        fold_rows.append(row)

    # ── confusion-matrix sweep (ensemble) ──────────────────────────────────────
    conf_rows = []
    for tol in args.tol_rel_sweep:
        cm = optimality_confusion_matrix(relax_cost, local, ens_pred, tol=tol, relative=True)
        conf_rows.append(_conf_row(case, relax, args, "relative", tol, cm))
    for tol in args.tol_abs_sweep:
        cm = optimality_confusion_matrix(relax_cost, local, ens_pred, tol=tol, relative=False)
        conf_rows.append(_conf_row(case, relax, args, "absolute", tol, cm))

    _print_headline(case, relax, summary_row, conf_rows)
    return summary_row, fold_rows, conf_rows


def _conf_row(case, relax, args, mode, tol, cm):
    n_cert = cm["tp"] + cm["fn"]
    return {"case": case, "relaxation": relax, "n_train": args.n_train,
            "mode": mode, "threshold": tol,
            "tp": cm["tp"], "fp": cm["fp"], "fn": cm["fn"], "tn": cm["tn"],
            "fpr": cm["fpr"], "fnr": cm["fnr"],
            "n_certifiable": n_cert,
            "certifiable_pct": 100.0 * n_cert / cm["n"] if cm["n"] else float("nan")}


def _print_headline(case, relax, s, conf_rows):
    print(f"  relax cost      : {s['relax_cost_mean']:.1f} "
          f"[{s['relax_cost_ci_lo']:.1f}, {s['relax_cost_ci_hi']:.1f}]")
    print(f"  NN APE vs relax : {s['nn_ape_relax_mean']:.3f}% "
          f"[{s['nn_ape_relax_ci_lo']:.3f}, {s['nn_ape_relax_ci_hi']:.3f}]")
    print(f"  NN APE vs local : {s['nn_ape_local_mean']:.3f}% "
          f"[{s['nn_ape_local_ci_lo']:.3f}, {s['nn_ape_local_ci_hi']:.3f}]")
    print(f"  gap (local-relax): {s['gap_pct_mean']:.3f}% "
          f"[{s['gap_pct_ci_lo']:.3f}, {s['gap_pct_ci_hi']:.3f}]")
    print(f"  OVER-pred vs relax: max={s['max_overpred_pct']:.3f}%  "
          f"q95={s['q95_overpred_pct']:.3f}%")
    rep = [c for c in conf_rows if c["mode"] == "relative" and abs(c["threshold"] - 0.005) < 1e-9]
    if rep:
        c = rep[0]
        print(f"  confusion @ rel 0.5%: TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']}  "
              f"FPR={c['fpr']:.3f} FNR={c['fnr']:.3f}  cert={c['certifiable_pct']:.1f}%")


def main():
    p = argparse.ArgumentParser(description="Evaluate trained AC-OPF DNNs.")
    p.add_argument("--cases", nargs="+", default=ALL_CASES, metavar="CASE")
    p.add_argument("--relax", nargs="+", default=ALL_RELAX,
                   choices=ALL_RELAX, metavar="RELAX")
    p.add_argument("--n-train", type=int, default=20_000)
    p.add_argument("--n-test", type=int, default=5_000)
    p.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR,
                   help="Directory holding the test CSVs (default: data/acopf-hpc)")
    p.add_argument("--folds", type=int, default=4)   # matches train_acopf.py
    p.add_argument("--tol-rel-sweep", type=float, nargs="+",
                   default=[0.001, 0.005, 0.01, 0.02, 0.05])
    p.add_argument("--tol-abs-sweep", type=float, nargs="+",
                   default=[10.0, 50.0, 100.0])
    args = p.parse_args()

    configs = [(c, r) for c in args.cases for r in args.relax
               if not (c == "case2869pegase" and r == "chordal_sdp")]

    # Write ONE small part-file per (relaxation, case) config, under results/
    # acopf/parts/.  Each config has a unique filename, so parallel SLURM jobs
    # (one per case) never write the same file — no races on shared CSVs.
    # scripts/merge_acopf_results.py assembles the master CSVs afterwards.
    parts_dir = RESULTS_DIR / "parts"
    parts_dir.mkdir(exist_ok=True)
    n_done = 0
    for case, relax in configs:
        res = evaluate_config(case, relax, args)
        if res is None:
            continue
        s, f, cf = res
        tag = f"{relax}__{case}"
        pd.DataFrame([s]).to_csv(parts_dir / f"eval__{tag}.csv", index=False)
        pd.DataFrame(f).to_csv(parts_dir / f"fold__{tag}.csv", index=False)
        pd.DataFrame(cf).to_csv(parts_dir / f"confusion__{tag}.csv", index=False)
        n_done += 1

    if n_done:
        print(f"\nWrote per-config result parts for {n_done} configs to {parts_dir}/",
              flush=True)
        print("Run `python scripts/merge_acopf_results.py` to assemble the master CSVs.",
              flush=True)


if __name__ == "__main__":
    main()
