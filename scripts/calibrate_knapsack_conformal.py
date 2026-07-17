"""Compute and cache the OOF over-prediction residuals used to calibrate the
NN-based B&B pruning cutoff (see problems/robust_knapsack/conformal.py).

Run once per trained model (per n_train); the residuals are then reused to
compute compute_margin(tau) for any tau cheaply, without reloading models.

Usage:
    python scripts/calibrate_knapsack_conformal.py --n-train 20000
"""

import argparse
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from problems.robust_knapsack.conformal import reconstruct_oof_predictions

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "robust_knapsack"
MODELS_DIR = PROJECT_ROOT / "models" / "robust_knapsack"
RESULTS_DIR = PROJECT_ROOT / "results" / "robust_knapsack"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=20_000)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    args = p.parse_args()

    train_csv = args.data_dir / f"train_{args.n_train}.csv"
    X, cost, oof_pred = reconstruct_oof_predictions(
        train_csv, MODELS_DIR, n_train=args.n_train, folds=args.folds, seed=args.seed)

    e = oof_pred - cost   # over-prediction residual (dangerous direction)
    out_path = RESULTS_DIR / f"knapsack_oof_residuals_n{args.n_train}.npy"
    np.save(out_path, e)
    print(f"n={len(e)}  residual mean={e.mean():.4f} std={e.std():.4f} "
          f"min={e.min():.4f} max={e.max():.4f}")
    print(f"Saved residuals -> {out_path}")


if __name__ == "__main__":
    main()
