"""Split-conformal calibration of the NN-based B&B pruning cutoff.

Confirmed mechanism (general min-MILP framing): we minimize sum(-v_i)x_i, so
the true optimum is v*_min = -Cost. The NN predicts Cost directly (MAX-sense,
un-negated); its MIN-sense prediction is nn_pred_min = -ensemble_pred.

The risk direction for the unified pruning rule
    cutoff = min(incumbent, nn_pred_min + margin)
    prune if node_lower_bound >= cutoff
is nn_pred_min UNDER-predicting v*_min. We calibrate margin so that
P(nn_pred_min + margin >= v*_min) >= 1 - tau (split conformal, one-sided).

Translating that calibration into the quantities we can directly compute (raw
Cost predictions, no negation): under-prediction of v*_min = -Cost by
nn_pred_min = -ensemble_pred is EXACTLY over-prediction of Cost by
ensemble_pred (negating both sides flips under to over). So:

    margin(tau) = the (1-tau) quantile of the OVER-prediction residuals
                  e_i = ensemble_pred_i - Cost_i
                  (computed OUT-OF-FOLD on the training set -- no retraining,
                   no held-out-from-test cost: cross-validation already
                   gives every training point an honest prediction from the ONE
                   fold model that never saw it)

    nn_cutoff_min(theta) = -ensemble_pred(theta) + margin(tau)
                          = margin(tau) - ensemble_pred(theta)

This nn_cutoff_min is what solve_bnb's `nn_cutoff` argument expects directly.
"""

import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from nn.training import load_checkpoint, predict_denorm

LABEL_COLS = ("Cost",)


def reconstruct_oof_predictions(train_csv_path, models_dir, n_train, folds=4, seed=0):
    """Reproduce train_knapsack.py's exact preprocessing + KFold split, and
    return (X_raw, cost, oof_pred) aligned row-for-row with the post-drop,
    reset-index dataframe train_knapsack.py actually trained on.

    Each row's oof_pred comes from the ONE fold model that did NOT see it
    during training (cross-conformal style) -- this reuses the full training
    set as a calibration set with zero retraining cost.
    """
    df = pd.read_csv(train_csv_path)
    df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce")
    # MUST match train_knapsack.py's drop-then-reset-index exactly, or fold
    # index arrays refer to different rows than what each checkpoint trained on.
    df = df[np.isfinite(df["Cost"])].reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in LABEL_COLS]
    X_raw = df[feat_cols].values.astype(np.float64)
    cost = df["Cost"].values.astype(np.float64)

    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_pred = np.full(len(df), np.nan)
    seen = np.zeros(len(df), dtype=bool)
    for fold, (_, val_idx) in enumerate(kf.split(X_raw)):
        ckpt_path = pathlib.Path(models_dir) / f"dnn_knapsack_n{n_train}_fold{fold}.pt"
        model, scalers, _ = load_checkpoint(ckpt_path)
        oof_pred[val_idx] = predict_denorm(model, X_raw[val_idx], scalers)
        seen[val_idx] = True

    assert seen.all(), "KFold reconstruction did not cover all rows -- split mismatch"
    assert np.isfinite(oof_pred).all(), "some rows never got an OOF prediction"
    return X_raw, cost, oof_pred


def compute_margin(oof_pred, cost, tau):
    """One-sided split-conformal margin from OVER-prediction residuals.

    e_i = oof_pred_i - cost_i  (positive = over-prediction of Cost, which is
    the dangerous direction after translating to the min-native cutoff -- see
    module docstring).

    Uses the exact finite-sample order-statistic quantile (NOT np.quantile's
    interpolated default), since that's what gives the marginal coverage
    guarantee P(margin >= e_new) >= 1 - tau exactly, not approximately.
    """
    e = np.asarray(oof_pred, dtype=float) - np.asarray(cost, dtype=float)
    n = len(e)
    e_sorted = np.sort(e)
    k = int(np.ceil((n + 1) * (1 - tau)))
    k = min(k, n)
    return float(e_sorted[k - 1])   # 1-indexed k-th order statistic


def nn_cutoff_min(ensemble_pred, margin):
    """Convert a raw (MAX-sense) ensemble prediction + margin into the
    MIN-sense cutoff solve_bnb expects: cutoff = margin - ensemble_pred.
    """
    return margin - np.asarray(ensemble_pred, dtype=float)
