"""Train DNNs on pre-generated relaxation-labeled data, for any small problem
in the registry (QCQP, MIMO, IK-lass1, IK-lass2).

Generalizes scripts/train_acopf.py's recipe (already validated by ablation on
AC-OPF) so every problem trains under the exact same deep/4-fold protocol:
depth-6 width-256 DNN, single-phase 1000-epoch cosine-annealed AdamW, weight
decay 1e-4, KFold(n_splits=4, shuffle=True, random_state=0).

Train-only: consumes the labelled CSV already produced by generate_dataset.py
and writes self-contained per-fold checkpoints (nn/training.save_checkpoint).
It does NOT generate data and does NOT compute certification metrics --
scripts/evaluate_certify.py reloads the checkpoints and does that separately,
so results stay reproducible outside any training session.

AC-OPF is NOT trained here -- its fold checkpoints already exist and are
reused as-is (registry can_generate=False).

Usage:
    python scripts/train_generic.py --key qcqp
    python scripts/train_generic.py --key ik_lass2 --epochs 5   # smoke test
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
from nn.training import (train_model, train_model_two_phase, to_loader,
                         save_checkpoint, load_checkpoint)
from problems.registry import get_spec, LABEL_COLS, SMALL_PROBLEM_KEYS


def train_config(spec, args):
    print(f"\n{'='*70}\n  TRAIN  {spec.key.upper()}\n{'='*70}", flush=True)

    if not spec.train_csv.exists():
        print(f"  SKIP: missing {spec.train_csv}", flush=True)
        return False

    df = pd.read_csv(spec.train_csv)
    feat_cols = spec.feature_cols or [c for c in df.columns if c not in LABEL_COLS]

    # Drop infeasible / failed samples (NaN Cost) -- can't train on a NaN target.
    n_raw = len(df)
    df = df[np.isfinite(pd.to_numeric(df["Cost"], errors="coerce"))].reset_index(drop=True)
    n_drop = n_raw - len(df)
    if n_drop:
        print(f"  dropped {n_drop}/{n_raw} infeasible rows (NaN Cost)", flush=True)
    if len(df) < args.folds:
        print(f"  SKIP: only {len(df)} usable rows", flush=True)
        return False

    X_all = df[feat_cols].values.astype(np.float64)
    y_all = df["Cost"].values.astype(np.float64)
    input_dim = X_all.shape[1]

    # Standardize on the full (post-drop) train set; the same scalers are stored
    # with every fold's checkpoint. Guard zero-variance features.
    x_mean = X_all.mean(axis=0)
    x_std = X_all.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean = float(y_all.mean())
    y_std = float(y_all.std()) or 1.0

    Xs = ((X_all - x_mean) / x_std).astype(np.float32)
    ys = ((y_all - y_mean) / y_std).astype(np.float32)

    print(f"  rows={len(df)}  input_dim={input_dim}  folds={args.folds}  "
          f"cost_mean={y_mean:.4g} cost_std={y_std:.4g}", flush=True)

    spec.models_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for fold in range(args.folds):
            ckpt = spec.models_dir / spec.ckpt_pattern.format(fold=fold)
            print(f"  [dry-run] would write {ckpt.name}")
        return True

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(kf.split(Xs)):
        tf = time.time()
        ckpt_path = spec.models_dir / spec.ckpt_pattern.format(fold=fold)
        # Idempotent resume: if this fold's checkpoint already exists and loads
        # cleanly, skip it. The KFold split is deterministic (fixed random_state),
        # so a resubmit after a wall-clock timeout reproduces the same splits and
        # only re-trains the folds that never finished. --force retrains all.
        if ckpt_path.exists() and not args.force:
            try:
                load_checkpoint(ckpt_path)
                print(f"  fold {fold+1}/{args.folds} -> {ckpt_path.name}  (exists, skipped)", flush=True)
                continue
            except Exception:
                print(f"  fold {fold+1}/{args.folds} -> {ckpt_path.name}  (exists but unreadable, retraining)", flush=True)
        model = DNN(input_dim=input_dim, hidden_dims=args.hidden_dims)
        if args.two_phase:
            model, _, _ = train_model_two_phase(
                model,
                Xs[tr_idx], Xs[val_idx], ys[tr_idx], ys[val_idx],
                pretrain_epochs=args.pretrain_epochs, pretrain_lr=args.pretrain_lr,
                pretrain_batch_size=args.pretrain_batch_size,
                finetune_epochs=args.finetune_epochs, finetune_lr=args.finetune_lr,
                finetune_batch_size=args.finetune_batch_size,
                weight_decay=args.weight_decay, verbose=False,
            )
        else:
            # Single cosine-annealed phase -- the AC-OPF ablation showed this
            # matches the two-phase schedule on accuracy AND the over-prediction
            # tail while training ~1.5x faster; used as the default everywhere.
            tl = to_loader(Xs[tr_idx], ys[tr_idx], batch_size=args.batch_size)
            vl = to_loader(Xs[val_idx], ys[val_idx], batch_size=args.batch_size, shuffle=False)
            model, _, _ = train_model(
                model, tl, vl, n_epochs=args.epochs, learning_rate=args.lr,
                weight_decay=args.weight_decay, verbose=False,
            )
        save_checkpoint(
            ckpt_path, model, x_mean, x_std, y_mean, y_std,
            input_dim, args.hidden_dims, feat_cols,
            extra={"key": spec.key, "n_train": len(df), "fold": fold, "seed": args.seed},
        )
        print(f"  fold {fold+1}/{args.folds} -> {ckpt_path.name}  ({time.time()-tf:.0f}s)", flush=True)

    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--key", required=True, choices=SMALL_PROBLEM_KEYS)
    p.add_argument("--n-train", type=int, default=20_000)
    p.add_argument("--n-test", type=int, default=5_000)
    p.add_argument("--folds", type=int, default=4)
    # depth 6 x width 256 -- the ablation-validated recipe from AC-OPF, applied
    # uniformly here rather than re-tuning per problem.
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256] * 6)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--two-phase", action="store_true",
                   help="Use the two-phase schedule instead of single-phase (default off).")
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
                        "(default: skip existing/loadable fold checkpoints, so a "
                        "resubmit after a timeout resumes instead of restarting).")
    args = p.parse_args()

    spec = get_spec(args.key, n_train=args.n_train, n_test=args.n_test)
    t0 = time.time()
    ok = train_config(spec, args)
    print(f"\n{'Trained' if ok else 'Skipped'} '{args.key}' in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
