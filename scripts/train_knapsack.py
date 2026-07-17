"""Train a DNN on the robust-knapsack MISOCP optimal-value data.

Single fixed problem instance (N_ITEMS=50 items; no case/relaxation grid like
AC-OPF), so this is a simplified train_acopf.py: load the CSV, standardize
inputs/target, 4-fold single-phase training with the SAME ablation-validated
recipe (depth 6 x width 256, 1000 epochs), save self-contained checkpoints.

Usage:
    python scripts/train_knapsack.py
    python scripts/train_knapsack.py --epochs 20   # quick smoke test
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nn.models import DNN
from nn.training import (train_model, to_loader, train_model_two_phase,
                         save_checkpoint, load_checkpoint)

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"   / "robust_knapsack"
MODELS_DIR       = PROJECT_ROOT / "models" / "robust_knapsack"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS = ("Cost",)


def _train_csv(data_dir, n):
    return data_dir / f"train_{n}.csv"


def _ckpt_path(n, fold):
    return MODELS_DIR / f"dnn_knapsack_n{n}_fold{fold}.pt"


def main():
    p = argparse.ArgumentParser(description="Train a DNN on robust-knapsack data.")
    p.add_argument("--n-train", type=int, default=20_000)
    p.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256] * 6)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--two-phase", action="store_true")
    p.add_argument("--pretrain-epochs", type=int, default=500)
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--pretrain-batch-size", type=int, default=256)
    p.add_argument("--finetune-epochs", type=int, default=200)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--finetune-batch-size", type=int, default=32)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Retrain all folds even if their checkpoints already exist "
                        "(default: skip existing/loadable fold checkpoints so a "
                        "resubmit after a timeout resumes instead of restarting).")
    args = p.parse_args()

    csv_path = _train_csv(args.data_dir, args.n_train)
    if not csv_path.exists():
        print(f"SKIP: missing {csv_path.name}")
        return

    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in LABEL_COLS]

    n_raw = len(df)
    df = df[np.isfinite(pd.to_numeric(df["Cost"], errors="coerce"))].reset_index(drop=True)
    n_drop = n_raw - len(df)
    if n_drop:
        print(f"  dropped {n_drop}/{n_raw} infeasible rows (NaN Cost)")

    X_all = df[feat_cols].values.astype(np.float64)
    y_all = df["Cost"].values.astype(np.float64)
    input_dim = X_all.shape[1]

    x_mean = X_all.mean(axis=0); x_std = X_all.std(axis=0); x_std[x_std == 0] = 1.0
    y_mean = float(y_all.mean()); y_std = float(y_all.std()) or 1.0
    Xs = ((X_all - x_mean) / x_std).astype(np.float32)
    ys = ((y_all - y_mean) / y_std).astype(np.float32)

    print(f"rows={len(df)}  input_dim={input_dim}  folds={args.folds}  "
          f"cost_mean={y_mean:.1f} cost_std={y_std:.1f}", flush=True)

    if args.dry_run:
        for fold in range(args.folds):
            print(f"  [dry-run] would write {_ckpt_path(args.n_train, fold).name}")
        return

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(kf.split(Xs)):
        tf = time.time()
        path = _ckpt_path(args.n_train, fold)
        # Idempotent resume: skip folds whose checkpoint already exists and loads
        # cleanly, so a resubmit after a wall-clock timeout continues instead of
        # restarting. KFold is deterministic (fixed seed), so splits reproduce.
        if path.exists() and not args.force:
            try:
                load_checkpoint(path)
                print(f"  fold {fold+1}/{args.folds} -> {path.name}  (exists, skipped)", flush=True)
                continue
            except Exception:
                print(f"  fold {fold+1}/{args.folds} -> {path.name}  (exists but unreadable, retraining)", flush=True)
        model = DNN(input_dim=input_dim, hidden_dims=args.hidden_dims)
        if args.two_phase:
            model, _, _ = train_model_two_phase(
                model, Xs[tr_idx], Xs[val_idx], ys[tr_idx], ys[val_idx],
                pretrain_epochs=args.pretrain_epochs, pretrain_lr=args.pretrain_lr,
                pretrain_batch_size=args.pretrain_batch_size,
                finetune_epochs=args.finetune_epochs, finetune_lr=args.finetune_lr,
                finetune_batch_size=args.finetune_batch_size,
                weight_decay=args.weight_decay, verbose=False,
            )
        else:
            tl = to_loader(Xs[tr_idx], ys[tr_idx], batch_size=args.batch_size)
            vl = to_loader(Xs[val_idx], ys[val_idx], batch_size=args.batch_size, shuffle=False)
            model, _, _ = train_model(
                model, tl, vl, n_epochs=args.epochs, learning_rate=args.lr,
                weight_decay=args.weight_decay, verbose=False,
            )
        save_checkpoint(
            path, model, x_mean, x_std, y_mean, y_std,
            input_dim, args.hidden_dims, feat_cols,
            extra={"problem": "robust_knapsack", "n_train": args.n_train,
                   "fold": fold, "seed": args.seed},
        )
        print(f"  fold {fold+1}/{args.folds} -> {path.name}  ({time.time()-tf:.0f}s)", flush=True)

    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
