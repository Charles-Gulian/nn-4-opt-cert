"""Train DNNs on pre-generated AC-OPF relaxation data.

Train-only: consumes the labelled CSVs already produced by the data-generation
pipeline (train_<N>_<relax>_<case>.csv) and writes self-contained checkpoints.
It does NOT generate data and does NOT report results — evaluate_acopf.py reloads
the checkpoints and computes metrics separately, so results are reproducible
outside any training session.

Per (case, relaxation):
  - load the train CSV, infer feature columns from the header,
  - drop infeasible rows (NaN Cost),
  - standardize inputs and target (fit on train),
  - K-fold two-phase training of a DNN per fold,
  - save each fold as a checkpoint embedding weights + scalers + architecture.

Usage:
    python scripts/train_acopf.py                      # full grid
    python scripts/train_acopf.py --cases case9 --relax socp
    python scripts/train_acopf.py --cases case9 --relax socp \
        --pretrain-epochs 20 --finetune-epochs 10      # quick smoke test
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
                         save_checkpoint)

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"   / "acopf-hpc"   # SAVIO-generated CSVs
MODELS_DIR       = PROJECT_ROOT / "models" / "acopf"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS = ("Cost", "Exact", "LocalCost")

# Full experiment grid. case2869pegase is SOCP-only (chordal SDP intractable).
ALL_CASES = ["case9", "case14", "case39", "case89pegase",
             "case118", "case300", "case1354pegase", "case2869pegase"]
ALL_RELAX = ["socp", "chordal_sdp"]


def _train_csv(data_dir, case, relax, n):
    return data_dir / f"train_{n}_{relax}_{case}.csv"


def _ckpt_path(case, relax, n, fold):
    return MODELS_DIR / f"dnn_{relax}_{case}_n{n}_fold{fold}.pt"


def _feature_columns(df):
    return [c for c in df.columns if c not in LABEL_COLS]


def train_config(case, relax, args):
    print(f"\n{'='*70}\n  TRAIN  {case.upper()}  |  {relax.upper()}\n{'='*70}", flush=True)

    csv_path = _train_csv(args.data_dir, case, relax, args.n_train)
    if not csv_path.exists():
        print(f"  SKIP: missing {csv_path.name}", flush=True)
        return False

    df = pd.read_csv(csv_path)
    feat_cols = _feature_columns(df)

    # Drop infeasible / failed samples (NaN Cost) — can't train on a NaN target.
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
    # with every fold's checkpoint.  Guard zero-variance features.
    x_mean = X_all.mean(axis=0)
    x_std  = X_all.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean = float(y_all.mean())
    y_std  = float(y_all.std()) or 1.0

    Xs = ((X_all - x_mean) / x_std).astype(np.float32)
    ys = ((y_all - y_mean) / y_std).astype(np.float32)

    print(f"  rows={len(df)}  input_dim={input_dim}  folds={args.folds}  "
          f"cost_mean={y_mean:.1f} cost_std={y_std:.1f}", flush=True)

    if args.dry_run:
        for fold in range(args.folds):
            print(f"  [dry-run] would write {_ckpt_path(case, relax, args.n_train, fold).name}")
        return True

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(kf.split(Xs)):
        tf = time.time()
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
            # Single cosine-annealed phase — the ablation showed this matches the
            # two-phase schedule on accuracy AND the over-prediction tail while
            # training ~1.5x faster (no slow small-batch finetune stage).
            tl = to_loader(Xs[tr_idx], ys[tr_idx], batch_size=args.batch_size)
            vl = to_loader(Xs[val_idx], ys[val_idx], batch_size=args.batch_size, shuffle=False)
            model, _, _ = train_model(
                model, tl, vl, n_epochs=args.epochs, learning_rate=args.lr,
                weight_decay=args.weight_decay, verbose=False,
            )
        path = _ckpt_path(case, relax, args.n_train, fold)
        save_checkpoint(
            path, model, x_mean, x_std, y_mean, y_std,
            input_dim, args.hidden_dims, feat_cols,
            extra={"case": case, "relaxation": relax,
                   "n_train": args.n_train, "fold": fold, "seed": args.seed},
        )
        print(f"  fold {fold+1}/{args.folds} -> {path.name}  ({time.time()-tf:.0f}s)", flush=True)

    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    return True


def main():
    p = argparse.ArgumentParser(description="Train DNNs on AC-OPF relaxation data.")
    p.add_argument("--cases", nargs="+", default=ALL_CASES, metavar="CASE")
    p.add_argument("--relax", nargs="+", default=ALL_RELAX,
                   choices=ALL_RELAX, metavar="RELAX")
    p.add_argument("--n-train", type=int, default=20_000)
    p.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR,
                   help="Directory holding the train CSVs (default: data/acopf-hpc)")
    p.add_argument("--folds", type=int, default=4)   # 20000/4 = 5000 val per fold = test size
    # depth 6 x width 256: a genuinely deep network whose certification-critical
    # over-prediction tail (q95) is as good as or better than shallower nets, with
    # no cost penalty worth worrying about.  Width 512 rarely earns its 2-4x params.
    # See results/acopf/ablation_summary.csv.
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256] * 6)
    # Single-phase training (ablation-validated default): one cosine-annealed
    # phase, 1000 epochs, lr 1e-3, batch 256.  Matches two-phase quality, ~1.5x
    # faster; 1000 epochs chosen as the knee of the epoch ablation (the over-
    # prediction tail's gains diminish past ~1000; 1500/2000 confirm the plateau).
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    # Opt-in two-phase schedule (pretrain then small-batch finetune).
    p.add_argument("--two-phase", action="store_true",
                   help="Use the two-phase schedule instead of single-phase.")
    p.add_argument("--pretrain-epochs", type=int, default=500)
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--pretrain-batch-size", type=int, default=256)
    p.add_argument("--finetune-epochs", type=int, default=200)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--finetune-batch-size", type=int, default=32)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    configs = [(c, r) for c in args.cases for r in args.relax
               if not (c == "case2869pegase" and r == "chordal_sdp")]
    print(f"Training {len(configs)} configs: "
          f"{', '.join(f'{r}/{c}' for c, r in configs)}", flush=True)

    t0 = time.time()
    n_ok = sum(train_config(c, r, args) for c, r in configs)
    print(f"\nTrained {n_ok}/{len(configs)} configs in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
