"""Evaluate the trained robust-knapsack DNN on the held-out test set.

Standalone: reloads the self-contained checkpoints from train_knapsack.py and
computes metrics with no dependency on the training session (same design as
evaluate_acopf.py). Since there is no relaxation/local split for this problem
(Cost is already the certified global optimum), the metrics are: Cost itself,
NN absolute-percent error vs Cost, and the (over/under)-prediction tail.

Usage:
    python scripts/evaluate_knapsack.py
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nn.training import load_checkpoint, predict_denorm
from nn.metrics import mean_ci, overprediction_summary

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"    / "robust_knapsack"
MODELS_DIR       = PROJECT_ROOT / "models"  / "robust_knapsack"
RESULTS_DIR      = PROJECT_ROOT / "results" / "robust_knapsack"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS = ("Cost",)


def _test_csv(data_dir, n):
    return data_dir / f"test_{n}.csv"


def _ckpt_path(n, fold):
    return MODELS_DIR / f"dnn_knapsack_n{n}_fold{fold}.pt"


def main():
    p = argparse.ArgumentParser(description="Evaluate the trained robust-knapsack DNN.")
    p.add_argument("--n-train", type=int, default=20_000)
    p.add_argument("--n-test", type=int, default=5_000)
    p.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--folds", type=int, default=4)
    args = p.parse_args()

    models, scalers = [], []
    for fold in range(args.folds):
        path = _ckpt_path(args.n_train, fold)
        if not path.exists():
            continue
        m, s, _ = load_checkpoint(path)
        models.append(m); scalers.append(s)
    if not models:
        print(f"SKIP: no checkpoints for n={args.n_train}")
        return
    if len(models) < args.folds:
        print(f"WARNING: found {len(models)}/{args.folds} fold checkpoints")

    csv_path = _test_csv(args.data_dir, args.n_test)
    if not csv_path.exists():
        print(f"SKIP: missing {csv_path.name}")
        return
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in LABEL_COLS]
    df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce")
    n_raw = len(df)
    df = df[np.isfinite(df["Cost"])].reset_index(drop=True)
    n_drop = n_raw - len(df)
    if n_drop:
        print(f"  dropped {n_drop}/{n_raw} test rows with NaN Cost")

    X = df[feat_cols].values.astype(np.float64)
    cost = df["Cost"].values.astype(np.float64)

    fold_preds = [predict_denorm(m, X, s) for m, s in zip(models, scalers)]
    ens_pred = np.mean(fold_preds, axis=0)

    ape = 100.0 * np.abs(ens_pred - cost) / cost
    ci_cost = mean_ci(cost)
    ci_ape = mean_ci(ape)
    over = overprediction_summary(ens_pred, cost, q=0.95)
    # Under-prediction tail is the mirror image (swap pred/target sign).
    under = overprediction_summary(cost, ens_pred, q=0.95)

    summary_row = {
        "n_train": args.n_train, "n_test_used": len(df), "n_folds": len(models),
        "cost_mean": ci_cost["mean"], "cost_ci_lo": ci_cost["ci_lower"], "cost_ci_hi": ci_cost["ci_upper"],
        "nn_ape_mean": ci_ape["mean"], "nn_ape_ci_lo": ci_ape["ci_lower"], "nn_ape_ci_hi": ci_ape["ci_upper"],
        "max_overpred_pct": over["max_overpred_pct"], "q95_overpred_pct": over["q_overpred_pct"],
        "max_underpred_pct": under["max_overpred_pct"], "q95_underpred_pct": under["q_overpred_pct"],
    }

    fold_rows = []
    for fold, fp in enumerate(fold_preds):
        fold_ape = 100.0 * np.abs(fp - cost) / cost
        fold_rows.append({"n_train": args.n_train, "fold": fold,
                           "nn_ape_mean": float(mean_ci(fold_ape)["mean"])})

    pd.DataFrame([summary_row]).to_csv(RESULTS_DIR / "eval_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(RESULTS_DIR / "fold_metrics.csv", index=False)

    print(f"  Cost            : {ci_cost['mean']:.2f}  [{ci_cost['ci_lower']:.2f}, {ci_cost['ci_upper']:.2f}]")
    print(f"  NN APE vs Cost  : {ci_ape['mean']:.4f}%  [{ci_ape['ci_lower']:.4f}, {ci_ape['ci_upper']:.4f}]")
    print(f"  OVER-pred  : max={over['max_overpred_pct']:.3f}%  q95={over['q_overpred_pct']:.3f}%")
    print(f"  UNDER-pred : max={under['max_overpred_pct']:.3f}%  q95={under['q_overpred_pct']:.3f}%")
    print(f"\nWrote results to {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
