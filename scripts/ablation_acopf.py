"""Architecture ablation for AC-OPF cost prediction.

Sweeps network depth x width and reports test-set accuracy + fold variance +
parameter count + training time per (case, architecture), so we can pick a
default architecture empirically before the full study.

Each architecture is trained with k-fold CV (same two-phase schedule the main
pipeline uses) on standardized inputs/target, evaluated on the held-out test
CSV via the fold ensemble. Throwaway models are NOT persisted — only metrics are
recorded to results/acopf/ablation_summary.csv (idempotent per
case/relax/depth/width).

Usage:
    python scripts/ablation_acopf.py                         # trio, socp, full grid
    python scripts/ablation_acopf.py --cases case118 --depths 2 --widths 256 \
        --pretrain-epochs 20 --finetune-epochs 10            # quick check
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
from nn.training import train_model_two_phase, train_model, to_loader, predict
from nn.metrics import mean_ci, overprediction_summary

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"    / "acopf-hpc"
RESULTS_DIR      = PROJECT_ROOT / "results" / "acopf"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS = ("Cost", "Exact", "LocalCost")
DEFAULT_CASES = ["case9", "case118", "case1354pegase"]


def _load_xy(csv_path, need_local):
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in LABEL_COLS]
    df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce")
    keep = np.isfinite(df["Cost"])
    if need_local:
        df["LocalCost"] = pd.to_numeric(df["LocalCost"], errors="coerce")
        keep &= np.isfinite(df["LocalCost"])
    df = df[keep].reset_index(drop=True)
    X = df[feat_cols].values.astype(np.float64)
    cost = df["Cost"].values.astype(np.float64)
    local = df["LocalCost"].values.astype(np.float64) if need_local else None
    return X, cost, local


def _upsert_csv(path, new_df, key_cols):
    if path.exists():
        old = pd.read_csv(path)
        keys = set(new_df[key_cols].apply(tuple, axis=1))
        mask = ~old[key_cols].apply(tuple, axis=1).isin(keys)
        out = pd.concat([old[mask], new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(path, index=False)


def run_case(case, relax, args):
    train_csv = args.data_dir / f"train_{args.n_train}_{relax}_{case}.csv"
    test_csv  = args.data_dir / f"test_{args.n_test}_{relax}_{case}.csv"
    if not train_csv.exists() or not test_csv.exists():
        print(f"  SKIP {case}/{relax}: missing CSV(s)", flush=True)
        return []

    X_tr, y_tr, _ = _load_xy(train_csv, need_local=False)
    X_te, relax_te, local_te = _load_xy(test_csv, need_local=True)
    input_dim = X_tr.shape[1]

    # standardize on train
    x_mean = X_tr.mean(0); x_std = X_tr.std(0); x_std[x_std == 0] = 1.0
    y_mean = float(y_tr.mean()); y_std = float(y_tr.std()) or 1.0
    Xs = ((X_tr - x_mean) / x_std).astype(np.float32)
    ys = ((y_tr - y_mean) / y_std).astype(np.float32)
    Xte_s = ((X_te - x_mean) / x_std).astype(np.float32)

    print(f"\n{'='*70}\n  ABLATION  {case.upper()}  |  {relax.upper()}  "
          f"(input_dim={input_dim}, train={len(X_tr)}, test={len(X_te)})\n{'='*70}", flush=True)

    rows = []
    for depth in args.depths:
        for width in args.widths:
            hidden = [width] * depth
            t0 = time.time()
            kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
            fold_preds, n_params = [], None
            for tr_idx, val_idx in kf.split(Xs):
                model = DNN(input_dim=input_dim, hidden_dims=hidden)
                if n_params is None:
                    n_params = sum(p.numel() for p in model.parameters())
                if not args.dry_run:
                    if args.single_phase:
                        # Single cosine-annealed phase over the same total epoch
                        # budget (pretrain settings), as the two-phase baseline.
                        n_ep = args.pretrain_epochs + args.finetune_epochs
                        tl = to_loader(Xs[tr_idx], ys[tr_idx], batch_size=args.pretrain_batch_size)
                        vl = to_loader(Xs[val_idx], ys[val_idx],
                                       batch_size=args.pretrain_batch_size, shuffle=False)
                        model, _, _ = train_model(
                            model, tl, vl, n_epochs=n_ep, learning_rate=args.pretrain_lr,
                            weight_decay=args.weight_decay, verbose=False)
                    else:
                        model, _, _ = train_model_two_phase(
                            model, Xs[tr_idx], Xs[val_idx], ys[tr_idx], ys[val_idx],
                            pretrain_epochs=args.pretrain_epochs, pretrain_lr=args.pretrain_lr,
                            pretrain_batch_size=args.pretrain_batch_size,
                            finetune_epochs=args.finetune_epochs, finetune_lr=args.finetune_lr,
                            finetune_batch_size=args.finetune_batch_size,
                            weight_decay=args.weight_decay, verbose=False,
                        )
                    pred = predict(model, Xte_s) * y_std + y_mean
                else:
                    pred = relax_te  # dry-run placeholder
                fold_preds.append(pred)

            ens = np.mean(fold_preds, axis=0)
            ape_rel = 100.0 * np.abs(ens - relax_te) / relax_te
            ape_loc = 100.0 * np.abs(ens - local_te) / local_te
            # per-fold APE-vs-relax means -> fold spread
            fold_ape = [float(np.mean(100.0 * np.abs(fp - relax_te) / relax_te))
                        for fp in fold_preds]
            ci_rel = mean_ci(ape_rel)
            # Worst-case OVER-prediction of the relaxation value (certification-critical).
            over = overprediction_summary(ens, relax_te, q=0.95)
            elapsed = time.time() - t0

            row = {
                "case": case, "relaxation": relax, "depth": depth, "width": width,
                "pretrain_epochs": args.pretrain_epochs,
                "finetune_epochs": args.finetune_epochs,
                "single_phase": int(args.single_phase),
                "n_params": int(n_params), "n_train": args.n_train, "folds": args.folds,
                "ape_relax_mean": ci_rel["mean"],
                "ape_relax_ci_lo": ci_rel["ci_lower"],
                "ape_relax_ci_hi": ci_rel["ci_upper"],
                "ape_relax_fold_std": float(np.std(fold_ape)),
                "ape_local_mean": float(mean_ci(ape_loc)["mean"]),
                "max_overpred_pct": over["max_overpred_pct"],
                "q95_overpred_pct": over["q_overpred_pct"],
                "train_time_s": elapsed,
            }
            rows.append(row)
            print(f"  depth={depth} width={width:4d}  params={n_params:>9,}  "
                  f"APE_relax={ci_rel['mean']:.4f}%  "
                  f"OVERpred max={over['max_overpred_pct']:.3f}% q95={over['q_overpred_pct']:.3f}%  "
                  f"{elapsed:.0f}s", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description="Depth x width ablation for AC-OPF DNNs.")
    p.add_argument("--cases", nargs="+", default=DEFAULT_CASES, metavar="CASE")
    p.add_argument("--relax", default="socp", choices=["socp", "chordal_sdp"])
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--widths", type=int, nargs="+", default=[256, 512])
    p.add_argument("--n-train", type=int, default=20_000)
    p.add_argument("--n-test", type=int, default=5_000)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--pretrain-epochs", type=int, default=150)
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--pretrain-batch-size", type=int, default=256)
    p.add_argument("--finetune-epochs", type=int, default=50)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--finetune-batch-size", type=int, default=32)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--single-phase", action="store_true",
                   help="Train with a single cosine phase (total = pretrain+finetune "
                        "epochs) instead of the two-phase schedule.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=pathlib.Path, default=RESULTS_DIR / "ablation_summary.csv")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    total = 0
    for case in args.cases:
        rows = run_case(case, args.relax, args)
        if rows and not args.dry_run:
            # Write after each case so partial progress persists during a long run.
            # Key includes the training config so epochs/strategy sweeps coexist.
            _upsert_csv(args.out, pd.DataFrame(rows),
                        ["case", "relaxation", "depth", "width",
                         "pretrain_epochs", "finetune_epochs", "single_phase"])
            total += len(rows)
            print(f"  -> wrote {len(rows)} rows for {case} to {args.out}", flush=True)

    if total:
        print(f"\nAblation complete: {total} rows in {args.out}  "
              f"(total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
