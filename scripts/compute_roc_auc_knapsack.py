"""Robust-knapsack certification ROC/AUC, reported like the other experiments.

The fast heuristic is the value/weight-density greedy solution (_greedy_incumbent);
we certify whether it is optimal. Ground truth is exact: Cost is the Gurobi-certified
global optimum. The NN g = v_hat predicts Cost. Knapsack is a MAXimization, so we
negate (f, g, v) to reuse the min-sense roc_auc_certification helper: a solution is
positive (truly optimal) iff Cost - f_greedy <= delta0, delta0 = 1e-3 * mean(Cost).

Two dataset sizes are treated as separate cases:
  n_train=20000 (test_5000)  and  n_train=80000 (test_20000).

Usage:  python scripts/compute_roc_auc_knapsack.py
"""
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nn.training import load_checkpoint, predict_denorm
from nn.metrics import roc_auc_certification
from problems.robust_knapsack.branch_and_bound import _greedy_incumbent, N_ITEMS

DATA = PROJECT_ROOT / "data" / "robust_knapsack"
MODELS = PROJECT_ROOT / "models" / "robust_knapsack"
ROC_DIR = PROJECT_ROOT / "results" / "roc_curves"
LABEL_COLS = ("Cost",)

CASES = [("knapsack_n20000", 20000, "test_5000"),
         ("knapsack_n80000", 80000, "test_20000")]


def _greedy_values(df, feat_cols):
    mu_cols = [c for c in feat_cols if c.startswith("Mu")]
    s2_cols = [c for c in feat_cols if c.startswith("Sigma2")]
    assert len(mu_cols) == N_ITEMS and len(s2_cols) == N_ITEMS, (len(mu_cols), len(s2_cols))
    mu = df[mu_cols].values.astype(float)
    s2 = df[s2_cols].values.astype(float)
    return np.array([_greedy_incumbent(mu[i], s2[i])[0] for i in range(len(df))])


def run(config, n_train, test_name):
    df = pd.read_csv(DATA / f"{test_name}.csv")
    feat_cols = [c for c in df.columns if c not in LABEL_COLS]
    df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce")
    df = df[np.isfinite(df["Cost"])].reset_index(drop=True)
    X = df[feat_cols].values.astype(np.float64)
    cost = df["Cost"].values.astype(np.float64)              # exact optimum v_true

    f_greedy = _greedy_values(df, feat_cols)                 # heuristic value (MAX)

    models = [load_checkpoint(MODELS / f"dnn_knapsack_n{n_train}_fold{k}.pt")
              for k in range(4)]
    g = np.concatenate([predict_denorm(m, X, s) for m, s, _ in models])
    f = np.tile(f_greedy, 4)
    v = np.tile(cost, 4)

    delta0 = 1e-3 * cost.mean()
    # MAX -> min by negation: gap = Cost - f_greedy >= 0.
    roc = roc_auc_certification(-f, -g, -v, delta0)
    ROC_DIR.mkdir(parents=True, exist_ok=True)
    if np.isfinite(roc["auc"]):
        np.savez(ROC_DIR / f"{config}.npz", fpr=roc["fpr"], tpr=roc["tpr"],
                 tau=roc["tau"], auc=roc["auc"], exp="knapsack")

    err = np.abs(g - v)                                      # NN error vs Cost
    gap = cost - f_greedy                                    # heuristic opt gap
    print(f"{config:16s} n_train={n_train:5d} n_test={len(df):5d}  "
          f"mean_Cost={cost.mean():.3f}  MAE={err.mean():.4f}  "
          f"greedy gap: mean={gap.mean():.4f} frac_suboptimal={np.mean(gap > delta0):.3f}  "
          f"AUC={roc['auc']:.4f}  n_pos={roc['n_pos']} n_neg={roc['n_neg']}")
    return dict(config=config, n_train=n_train, n_test=len(df),
                mean_cost=cost.mean(), mae=err.mean(),
                mean_greedy_gap=gap.mean(), auc=roc["auc"],
                n_pos=roc["n_pos"], n_neg=roc["n_neg"], delta0=delta0)


def main():
    rows = [run(*c) for c in CASES]
    out = PROJECT_ROOT / "results" / "roc_auc_knapsack.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
