"""Retrain ONE fold of one config at the full recipe, capturing the train/val
loss curves for a paper figure.

The production trainer (scripts/train_generic.py, scripts/train_acopf.py)
deliberately discards train_history/test_history after training -- only the
final weights are saved, so the loss curve isn't recoverable from an existing
checkpoint. This script reruns fold 0 of a chosen config identically (same
KFold split via the same random_state, same architecture/hyperparameters) but
keeps the per-epoch losses and does NOT overwrite the real checkpoint -- it's
purely to produce a representative training-history plot, not a new model.

Usage:
    python scripts/plot_training_history.py --key qcqp
"""

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nn.models import DNN
from nn.training import train_model, to_loader
from problems.registry import get_spec, LABEL_COLS, SMALL_PROBLEM_KEYS

FIGURES_DIR = PROJECT_ROOT / "figures"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key", default="qcqp", choices=SMALL_PROBLEM_KEYS)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256] * 6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    spec = get_spec(args.key)
    df = pd.read_csv(spec.train_csv)
    feat_cols = spec.feature_cols or [c for c in df.columns if c not in LABEL_COLS]
    df = df[np.isfinite(pd.to_numeric(df["Cost"], errors="coerce"))].reset_index(drop=True)

    X_all = df[feat_cols].values.astype(np.float64)
    y_all = df["Cost"].values.astype(np.float64)

    x_mean, x_std = X_all.mean(0), X_all.std(0)
    x_std[x_std == 0] = 1.0
    y_mean, y_std = float(y_all.mean()), float(y_all.std()) or 1.0
    Xs = ((X_all - x_mean) / x_std).astype(np.float32)
    ys = ((y_all - y_mean) / y_std).astype(np.float32)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(kf.split(Xs))
    tr_idx, val_idx = splits[args.fold]

    print(f"Retraining {args.key} fold {args.fold}/{args.folds} "
          f"(train={len(tr_idx)}, val={len(val_idx)}) for {args.epochs} epochs "
          f"-- this reproduces the production run's split/architecture but is "
          f"NOT saved as a checkpoint (history-only).")

    model = DNN(input_dim=Xs.shape[1], hidden_dims=args.hidden_dims)
    tl = to_loader(Xs[tr_idx], ys[tr_idx], batch_size=args.batch_size)
    vl = to_loader(Xs[val_idx], ys[val_idx], batch_size=args.batch_size, shuffle=False)
    _, train_hist, val_hist = train_model(
        model, tl, vl, n_epochs=args.epochs, learning_rate=args.lr,
        weight_decay=args.weight_decay, verbose=True,
    )

    hist_df = pd.DataFrame({"epoch": np.arange(1, len(train_hist) + 1),
                            "train_mse": train_hist, "val_mse": val_hist})
    hist_path = FIGURES_DIR / f"training_history_{args.key}_fold{args.fold}.csv"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    hist_df.to_csv(hist_path, index=False)
    print(f"wrote {hist_path}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(hist_df["epoch"], hist_df["train_mse"], alpha=0.25, color="tab:blue")
    ax.plot(hist_df["epoch"], hist_df["val_mse"], alpha=0.25, color="tab:orange")
    # A short rolling mean makes the underlying cosine-annealed decay legible
    # through the per-batch noise, without hiding the raw curves (still plotted
    # faintly underneath).
    win = max(1, len(hist_df) // 100)
    ax.plot(hist_df["epoch"], hist_df["train_mse"].rolling(win, min_periods=1).mean(),
            color="tab:blue", label="train MSE (standardized target)")
    ax.plot(hist_df["epoch"], hist_df["val_mse"].rolling(win, min_periods=1).mean(),
            color="tab:orange", label="val MSE (standardized target)")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (standardized target scale, log)")
    ax.set_title(f"Training history -- {args.key}, fold {args.fold}\n"
                 f"({'x'.join(str(h) for h in args.hidden_dims)} DNN, {args.epochs} epochs,\n"
                 f"single-phase cosine-annealed AdamW)", fontsize=11)
    ax.legend()
    fig.tight_layout()

    fig_path = FIGURES_DIR / f"training_history_{args.key}_fold{args.fold}.png"
    fig.savefig(fig_path, dpi=150)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
