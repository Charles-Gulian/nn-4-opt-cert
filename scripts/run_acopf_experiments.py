"""Run AC-OPF pipeline for all (relaxation, case) configurations.

Configurations: (socp, sdp) x (case9, case14, case39)

Key design:
  - X points (demand vectors) are generated once per (case, N, seed) and saved to
    a relaxation-agnostic file, so SOCP and SDP are guaranteed to share the same
    input distribution.
  - Relaxation labels (Cost, Exact) are computed separately per relaxation and
    saved to the standard per-relaxation CSV files that the notebook also uses.
  - Results for all configs are appended to a single summary CSV.

Usage:
    python scripts/run_acopf_experiments.py [--dry-run]

    --dry-run  : Print what would be done without generating data or training.
    --no-regen : Skip data generation if files already exist (default behaviour).
                 Pass --regen to force regeneration.
"""

import argparse
import csv
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# ── project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from problems.acopf.network import load_network
from problems.acopf.problem import solve_relaxation, solve_local
from problems.acopf.generate_data import (
    sample_parameters, _col_names,
    DEFAULT_ALPHA_MIN, DEFAULT_ALPHA_MAX, DEFAULT_ETA_RANGE,
)
from nn.models import DNN
from nn.training import train_model_two_phase, predict, save_model
from nn.metrics import error_summary, optimality_confusion_matrix

# ── experiment grid ───────────────────────────────────────────────────────────
RELAXATIONS = ["socp", "sdp"]
CASES       = ["case9", "case14", "case39"]

# ── shared hyperparameters ────────────────────────────────────────────────────
SEED                = 42
N_TRAIN             = 10_000
N_TEST              = 5_000
N_FOLDS             = 2
HIDDEN_DIMS         = [256, 256]
PRETRAIN_EPOCHS     = 500
PRETRAIN_LR         = 1e-3
PRETRAIN_BATCH_SIZE = 256
FINETUNE_EPOCHS     = 200
FINETUNE_LR         = 1e-4
FINETUNE_BATCH_SIZE = 32
WEIGHT_DECAY        = 1e-4

# Confusion-matrix tolerance per relaxation.  SOCP has a non-trivial gap on
# larger cases; SDP is typically exact (rank-1).  These are starting values —
# inspect the per-config gap statistics printed during the run to tune further.
TOL = {"socp": 50.0, "sdp": 10.0}

DATA_DIR    = PROJECT_ROOT / "data"    / "acopf"
MODELS_DIR  = PROJECT_ROOT / "models"  / "acopf"
RESULTS_DIR = PROJECT_ROOT / "results" / "acopf"

for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = RESULTS_DIR / "summary.csv"

SUMMARY_FIELDS = [
    "case", "relaxation",
    "n_train", "n_test",
    "relax_gap_mean", "relax_gap_median", "relax_gap_p95", "relax_gap_max",
    "exact_frac_train",
    "mae", "mae_ci_lower", "mae_ci_upper", "max_abs_error",
    "tol",
    "tp", "fp", "fn", "tn", "fpr", "fnr",
    "n_certifiable", "certifiable_pct",
    "elapsed_data_s", "elapsed_train_s", "elapsed_eval_s",
]


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _x_path(case_name, n, seed, split):
    """Path for the relaxation-agnostic X matrix (demand vectors only)."""
    return DATA_DIR / f"X_{split}_{n}_{case_name}_seed{seed}.npy"


def _data_path(case_name, relaxation, n, split):
    """Path for the labelled CSV (Cost, Exact, [LocalCost])."""
    return DATA_DIR / f"{split}_{n}_{relaxation}_{case_name}.csv"


def _model_path(case_name, relaxation, fold):
    return MODELS_DIR / f"dnn_{relaxation}_{case_name}_fold{fold}.pt"


def _label_data(X, nd, net, relaxation, include_local, args_base, verbose=True):
    """Solve relaxation (and optionally local OPF) for every row of X.

    Returns a DataFrame with columns [Cost, Exact] or [Cost, Exact, LocalCost].
    """
    prob_cache = {}
    relax_args = dict(args_base, prob_cache=prob_cache, relaxation=relaxation)

    costs, exacts, locals_ = [], [], []
    n = len(X)
    for i, p in enumerate(X):
        if verbose and (i % 500 == 0):
            print(f"    sample {i}/{n} ...", flush=True)
        val, res = solve_relaxation(p, args=relax_args)
        costs.append(val)
        exacts.append(res["exact"])
        if include_local:
            lval, _ = solve_local(p, args=args_base)
            locals_.append(lval)

    df = pd.DataFrame({"Cost": costs, "Exact": exacts})
    if include_local:
        df["LocalCost"] = locals_
    return df


# ─────────────────────────────────────────────────────────────────────────────
# per-config pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_config(case_name, relaxation, force_regen=False, dry_run=False):
    print(f"\n{'='*70}")
    print(f"  {case_name.upper()}  |  {relaxation.upper()}")
    print(f"{'='*70}")

    # ── network ───────────────────────────────────────────────────────────────
    net, nd = load_network(case_name)
    feat_cols = _col_names(nd)
    input_dim = 2 * nd.n_loads

    args_base = {
        "nd": nd, "net": net, "case_name": case_name,
        "alpha_min": DEFAULT_ALPHA_MIN, "alpha_max": DEFAULT_ALPHA_MAX,
        "eta_range": DEFAULT_ETA_RANGE,
    }

    # ── paths ─────────────────────────────────────────────────────────────────
    x_train_path  = _x_path(case_name, N_TRAIN, SEED,     "train")
    x_test_path   = _x_path(case_name, N_TEST,  SEED + 1, "test")
    train_csv     = _data_path(case_name, relaxation, N_TRAIN, "train")
    test_csv      = _data_path(case_name, relaxation, N_TEST,  "test")

    if dry_run:
        print(f"  [dry-run] would write: {train_csv.name}, {test_csv.name}")
        return None

    # ── Stage 1: X generation (shared across relaxations) ────────────────────
    t0_data = time.time()

    if not x_train_path.exists() or force_regen:
        print(f"  Sampling {N_TRAIN} training X points (seed={SEED}) ...")
        X_train = sample_parameters(N_TRAIN, args=dict(args_base, seed=SEED))
        np.save(x_train_path, X_train)
        print(f"    -> {x_train_path.name}")
    else:
        print(f"  Loading training X from {x_train_path.name}")
        X_train = np.load(x_train_path)

    if not x_test_path.exists() or force_regen:
        print(f"  Sampling {N_TEST} test X points (seed={SEED+1}) ...")
        X_test = sample_parameters(N_TEST, args=dict(args_base, seed=SEED + 1))
        np.save(x_test_path, X_test)
        print(f"    -> {x_test_path.name}")
    else:
        print(f"  Loading test X from {x_test_path.name}")
        X_test = np.load(x_test_path)

    # ── Stage 2: relaxation labels ────────────────────────────────────────────
    if not train_csv.exists() or force_regen:
        print(f"  Labelling {N_TRAIN} training samples with {relaxation.upper()} ...")
        df_train_labels = _label_data(
            X_train, nd, net, relaxation, include_local=False, args_base=args_base
        )
        df_train = pd.DataFrame(X_train, columns=feat_cols)
        df_train = pd.concat([df_train, df_train_labels], axis=1)
        df_train.to_csv(train_csv, index=False)
        print(f"    -> {train_csv.name}")
    else:
        print(f"  Loading training CSV from {train_csv.name}")
        df_train = pd.read_csv(train_csv)

    if not test_csv.exists() or force_regen:
        print(f"  Labelling {N_TEST} test samples with {relaxation.upper()} + local solver ...")
        df_test_labels = _label_data(
            X_test, nd, net, relaxation, include_local=True, args_base=args_base
        )
        df_test = pd.DataFrame(X_test, columns=feat_cols)
        df_test = pd.concat([df_test, df_test_labels], axis=1)
        df_test.to_csv(test_csv, index=False)
        print(f"    -> {test_csv.name}")
    else:
        print(f"  Loading test CSV from {test_csv.name}")
        df_test = pd.read_csv(test_csv)

    elapsed_data = time.time() - t0_data

    # ── Stage 3: train DNN ────────────────────────────────────────────────────
    t0_train = time.time()

    X_all = df_train[feat_cols].values.astype(np.float32)
    y_all = df_train["Cost"].values.astype(np.float32)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_models = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all)):
        print(f"  Fold {fold + 1}/{N_FOLDS} ...")
        model = DNN(input_dim=input_dim, hidden_dims=HIDDEN_DIMS)
        model, _, _ = train_model_two_phase(
            model,
            X_all[tr_idx], X_all[val_idx],
            y_all[tr_idx], y_all[val_idx],
            pretrain_epochs=PRETRAIN_EPOCHS,
            pretrain_lr=PRETRAIN_LR,
            pretrain_batch_size=PRETRAIN_BATCH_SIZE,
            finetune_epochs=FINETUNE_EPOCHS,
            finetune_lr=FINETUNE_LR,
            finetune_batch_size=FINETUNE_BATCH_SIZE,
            weight_decay=WEIGHT_DECAY,
            verbose=False,   # suppress tqdm in batch mode
        )
        save_model(model, _model_path(case_name, relaxation, fold))
        fold_models.append(model)

    elapsed_train = time.time() - t0_train

    # ── Stage 4: evaluate ─────────────────────────────────────────────────────
    t0_eval = time.time()

    X_te = df_test[feat_cols].values.astype(np.float32)
    y_relax = df_test["Cost"].values
    y_local = df_test["LocalCost"].values

    preds = np.stack([predict(m, X_te) for m in fold_models]).mean(axis=0)
    errs  = error_summary(y_relax, preds)

    relax_gap = y_local - y_relax
    tol = TOL[relaxation]
    cm  = optimality_confusion_matrix(y_relax, y_local, preds, tol=tol)
    n_certifiable = cm["tp"] + cm["fn"]

    exact_frac = df_train["Exact"].mean()

    elapsed_eval = time.time() - t0_eval

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\n  --- Results ({case_name}, {relaxation.upper()}) ---")
    print(f"  Exact (tight) in train : {exact_frac*100:.1f}%")
    print(f"  Relax gap (test)       : mean={relax_gap.mean():.2f}  "
          f"median={np.median(relax_gap):.2f}  "
          f"p95={np.percentile(relax_gap,95):.2f}  "
          f"max={relax_gap.max():.2f} $/hr")
    print(f"  MAE                    : {errs['mean_abs_error']:.2f} $/hr  "
          f"[{errs['ci_lower']:.2f}, {errs['ci_upper']:.2f}]")
    print(f"  Max abs error          : {errs['max_abs_error']:.2f} $/hr")
    print(f"  Confusion matrix (TOL={tol}):")
    print(f"    TP={cm['tp']}  FP={cm['fp']}  FN={cm['fn']}  TN={cm['tn']}")
    print(f"    FPR={cm['fpr']:.4f}  FNR={cm['fnr']:.4f}")
    print(f"    Certifiable: {n_certifiable}/{cm['n']} ({100*n_certifiable/cm['n']:.1f}%)")
    print(f"  Timing: data={elapsed_data:.0f}s  train={elapsed_train:.0f}s  eval={elapsed_eval:.0f}s")

    row = {
        "case": case_name,
        "relaxation": relaxation,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "relax_gap_mean":   float(relax_gap.mean()),
        "relax_gap_median": float(np.median(relax_gap)),
        "relax_gap_p95":    float(np.percentile(relax_gap, 95)),
        "relax_gap_max":    float(relax_gap.max()),
        "exact_frac_train": float(exact_frac),
        "mae":              float(errs["mean_abs_error"]),
        "mae_ci_lower":     float(errs["ci_lower"]),
        "mae_ci_upper":     float(errs["ci_upper"]),
        "max_abs_error":    float(errs["max_abs_error"]),
        "tol":              tol,
        "tp": cm["tp"], "fp": cm["fp"], "fn": cm["fn"], "tn": cm["tn"],
        "fpr": cm["fpr"], "fnr": cm["fnr"],
        "n_certifiable":    n_certifiable,
        "certifiable_pct":  100.0 * n_certifiable / cm["n"],
        "elapsed_data_s":   elapsed_data,
        "elapsed_train_s":  elapsed_train,
        "elapsed_eval_s":   elapsed_eval,
    }
    return row


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--regen",    action="store_true",
                        help="Force regeneration even if CSV files exist")
    parser.add_argument("--cases",    nargs="+", default=CASES,
                        choices=CASES, metavar="CASE",
                        help="Subset of cases to run (default: all)")
    parser.add_argument("--relax",    nargs="+", default=RELAXATIONS,
                        choices=RELAXATIONS, metavar="RELAX",
                        help="Subset of relaxations to run (default: all)")
    args = parser.parse_args()

    configs = [(r, c) for c in args.cases for r in args.relax]
    print(f"Running {len(configs)} configurations: "
          f"{', '.join(f'{r}/{c}' for r,c in configs)}")

    rows = []
    t_total = time.time()

    for relaxation, case_name in configs:
        row = run_config(
            case_name, relaxation,
            force_regen=args.regen,
            dry_run=args.dry_run,
        )
        if row is not None:
            rows.append(row)

    if rows:
        write_header = not SUMMARY_CSV.exists()
        with open(SUMMARY_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary appended to {SUMMARY_CSV}")

        # pretty-print the table
        df = pd.DataFrame(rows)[["case", "relaxation", "exact_frac_train",
                                  "relax_gap_mean", "mae", "fpr", "fnr",
                                  "certifiable_pct"]]
        df.columns = ["case", "relax", "exact%", "gap_mean", "MAE", "FPR", "FNR", "cert%"]
        df["exact%"]  = (df["exact%"]  * 100).round(1)
        df["gap_mean"] = df["gap_mean"].round(2)
        df["MAE"]      = df["MAE"].round(2)
        df["FPR"]      = df["FPR"].round(4)
        df["FNR"]      = df["FNR"].round(4)
        df["cert%"]    = df["cert%"].round(1)
        print(f"\n{'='*70}")
        print("Summary table")
        print(f"{'='*70}")
        print(df.to_string(index=False))

    print(f"\nTotal elapsed: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
