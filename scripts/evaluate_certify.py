"""Per-fold conformal calibration, test evaluation, and optimality certification
-- steps 4-6 of the unified workflow, applied uniformly to every problem
(QCQP, MIMO, IK-lass1, IK-lass2, AC-OPF) via problems/registry.py.

Unlike scripts/evaluate_acopf.py (which only reports the 4-fold ENSEMBLE and
saves no raw predictions), this script treats each fold model as an
independent estimator:

  Step 4 -- calibrate: reconstruct the exact KFold(4, shuffle, random_state=0)
    split used at training time (pattern from
    problems/robust_knapsack/conformal.py:reconstruct_oof_predictions), predict
    on each fold's own held-out validation rows, and take the split-conformal
    offset t_L = nn.metrics.conformal_offset(g - v, L) for L in {0.90, 0.95, 0.99}.
    This is a proper held-out calibration set for that fold's model (it never
    trained on these rows), unlike the OOF/ensemble reconstruction knapsack
    uses for a different purpose (calibrating one shared cutoff for a 4-model
    ensemble) -- here every fold gets its OWN offsets from its OWN validation
    fold.

  Step 5 -- test: predict on the FULL 5000-row test set with each fold model
    (all folds see the same held-out test set), and derive MAPE / signed-error
    percentiles / min-max via nn.metrics.prediction_error_summary. The raw
    per-instance predictions are persisted in full (fold_test_predictions.csv)
    since we may want box-and-whisker plots of the error distribution later --
    not just the summary statistics.

  Step 6 -- certify: at each fold/level, nn.metrics.certification_confusion
    scores every test row against delta=0.1% (relative optimality gap).

Outputs under <spec.results_dir>/ (spec from problems/registry.py):
  fold_test_predictions.csv   fold, idx, g (nn pred), v (Cost), f (LocalCost)  [large, SAVIO-only]
  val_residuals.csv           fold, idx, residual (g - v) on that fold's own validation rows [large, SAVIO-only]
  conformal_offsets.csv       fold, level, offset                              [small, portable]
  fold_test_metrics.csv       fold, mape, p1, p5, p10, p90, p95, p99, min_error, max_error  [small, portable]
  certification.csv           fold, level, tp, fp, tn, fn, tpr, fpr, tnr, fnr  [small, portable]

Usage:
    python scripts/evaluate_certify.py --key qcqp
    python scripts/evaluate_certify.py --key acopf_socp_case9
    python scripts/evaluate_certify.py --keys-file <list of acopf keys>   # batch
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nn.training import load_checkpoint, predict_denorm
from nn.metrics import conformal_offset, prediction_error_summary, certification_confusion
from problems.registry import get_spec, LABEL_COLS, SMALL_PROBLEM_KEYS, acopf_keys

LEVELS = (0.90, 0.95, 0.99)
DELTA = 0.001   # fixed test optimality-gap target, 0.1%, per the workflow spec


def _load_train_for_calibration(spec, folds, seed):
    """Reconstruct the post-drop, reset-index train dataframe and its KFold
    split exactly as train_generic.py (or train_acopf.py) built it, so each
    fold's validation-index array lines up with the rows that fold's
    checkpoint never trained on. Mirrors
    problems/robust_knapsack/conformal.py:reconstruct_oof_predictions.
    """
    df = pd.read_csv(spec.train_csv)
    df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce")
    df = df[np.isfinite(df["Cost"])].reset_index(drop=True)
    feat_cols = spec.feature_cols or [c for c in df.columns if c not in LABEL_COLS]

    X = df[feat_cols].values.astype(np.float64)
    v = df["Cost"].values.astype(np.float64)

    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(kf.split(X))
    return X, v, feat_cols, splits


def _ckpt_path(spec, fold):
    return spec.models_dir / spec.ckpt_pattern.format(fold=fold)


def calibrate_and_certify(key, n_train, n_test, folds, seed):
    spec = get_spec(key, n_train=n_train, n_test=n_test)
    print(f"\n{'='*70}\n  CERTIFY  {key.upper()}\n{'='*70}", flush=True)

    missing = [f for f in range(folds) if not _ckpt_path(spec, f).exists()]
    if missing:
        print(f"  SKIP: missing checkpoints for folds {missing}", flush=True)
        return False
    if not spec.test_csv.exists():
        print(f"  SKIP: missing test CSV {spec.test_csv}", flush=True)
        return False

    # ── Step 4: per-fold calibration on that fold's own held-out validation rows ──
    X_train, v_train, feat_cols, splits = _load_train_for_calibration(spec, folds, seed)

    offset_rows = []
    val_resid_rows = []
    for fold, (_, val_idx) in enumerate(splits):
        model, scalers, meta = load_checkpoint(_ckpt_path(spec, fold))
        g_val = predict_denorm(model, X_train[val_idx], scalers)
        resid = g_val - v_train[val_idx]   # over-prediction residual, g - v

        for idx, r in zip(val_idx, resid):
            val_resid_rows.append({"fold": fold, "idx": int(idx), "residual": float(r)})
        for L in LEVELS:
            t = conformal_offset(resid, L)
            offset_rows.append({"fold": fold, "level": L, "offset": t})
            print(f"  fold {fold}  L={L:.2f}  n_val={len(val_idx)}  offset={t:.6g}", flush=True)

    offsets_df = pd.DataFrame(offset_rows)
    val_resid_df = pd.DataFrame(val_resid_rows)

    # ── Step 5: per-fold prediction on the FULL test set + error summary ──────────
    test_df = pd.read_csv(spec.test_csv)
    test_feat_cols = feat_cols  # same columns used at train time
    X_test = test_df[test_feat_cols].values.astype(np.float64)
    v_test = pd.to_numeric(test_df["Cost"], errors="coerce").values.astype(np.float64)
    f_test = pd.to_numeric(test_df["LocalCost"], errors="coerce").values.astype(np.float64)
    ok_test = np.isfinite(v_test) & np.isfinite(f_test)
    n_dropped = (~ok_test).sum()
    if n_dropped:
        print(f"  dropping {n_dropped}/{len(test_df)} test rows with NaN Cost/LocalCost", flush=True)
    X_test, v_test, f_test = X_test[ok_test], v_test[ok_test], f_test[ok_test]

    pred_rows = []
    metrics_rows = []
    cert_rows = []
    for fold in range(folds):
        model, scalers, meta = load_checkpoint(_ckpt_path(spec, fold))
        g_test = predict_denorm(model, X_test, scalers)

        for i in range(len(g_test)):
            pred_rows.append({"fold": fold, "idx": i, "g": float(g_test[i]),
                              "v": float(v_test[i]), "f": float(f_test[i])})

        summary = prediction_error_summary(g_test, v_test)
        metrics_rows.append({"fold": fold, **summary})
        print(f"  fold {fold}  test MAPE={summary['mape']:.4g}%  "
              f"p95={summary['p95']:.4g}  p99={summary['p99']:.4g}", flush=True)

        # ── Step 6: certification confusion matrix at each conformal level ────────
        for L in LEVELS:
            t = offsets_df.query("fold == @fold and level == @L")["offset"].iloc[0]
            cm = certification_confusion(v_test, f_test, g_test, t, DELTA)
            cert_rows.append({"fold": fold, "level": L, **cm})
            print(f"    L={L:.2f}  TP={cm['tp']} FP={cm['fp']} TN={cm['tn']} FN={cm['fn']}  "
                  f"TPR={cm['tpr']:.4g} FPR={cm['fpr']:.4g}", flush=True)

    pred_df = pd.DataFrame(pred_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    cert_df = pd.DataFrame(cert_rows)

    # ── write outputs: large intermediates first, then small portable summaries ──
    spec.results_dir.mkdir(parents=True, exist_ok=True)
    pred_path = spec.results_dir / "fold_test_predictions.csv"
    resid_path = spec.results_dir / "val_residuals.csv"
    pred_df.to_csv(pred_path, index=False)
    val_resid_df.to_csv(resid_path, index=False)
    print(f"  wrote {pred_path} ({len(pred_df)} rows)")
    print(f"  wrote {resid_path} ({len(val_resid_df)} rows)")

    offsets_path = spec.results_dir / "conformal_offsets.csv"
    metrics_path = spec.results_dir / "fold_test_metrics.csv"
    cert_path = spec.results_dir / "certification.csv"
    offsets_df.to_csv(offsets_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    cert_df.to_csv(cert_path, index=False)
    print(f"  wrote {offsets_path}")
    print(f"  wrote {metrics_path}")
    print(f"  wrote {cert_path}")

    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--key", help="single config key, e.g. 'qcqp' or 'acopf_socp_case9'")
    group.add_argument("--all-small", action="store_true",
                        help="run all small-problem keys: " + ", ".join(SMALL_PROBLEM_KEYS))
    group.add_argument("--all-acopf", action="store_true",
                        help="run all AC-OPF (case, relaxation) configs")
    p.add_argument("--n-train", type=int, default=20_000)
    p.add_argument("--n-test", type=int, default=5_000)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.key:
        keys = [args.key]
    elif args.all_small:
        keys = SMALL_PROBLEM_KEYS
    else:
        keys = acopf_keys()

    n_ok = 0
    for key in keys:
        ok = calibrate_and_certify(key, args.n_train, args.n_test, args.folds, args.seed)
        n_ok += int(ok)
    print(f"\nCertified {n_ok}/{len(keys)} configs", flush=True)


if __name__ == "__main__":
    main()
