"""Error-summary utilities for comparing predicted vs. true optimal values."""

import numpy as np


def error_summary(y_true, y_pred, confidence=0.95):
    """Return a dict with mean absolute error, a (two-sided) confidence interval
    on the absolute error, and the maximum absolute error.
    """
    y_true, y_pred = np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)
    abs_err = np.abs(y_pred - y_true)

    n = len(abs_err)
    mean = abs_err.mean()
    sem = abs_err.std(ddof=1) / np.sqrt(n)
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)

    return dict(
        n=n,
        mean_abs_error=mean,
        ci_lower=mean - z * sem,
        ci_upper=mean + z * sem,
        max_abs_error=abs_err.max(),
    )


def optimality_confusion_matrix(relax_value, local_value, nn_pred, tol=1e-2):
    """Confusion matrix for using the NN to certify optimality of a local solver.

    "Optimal" means the local solution's value matches the relaxation value
    (i.e. the relaxation's lower bound is tight) to within `tol`:
      - Ground truth optimality:  |local_value - relax_value| <= tol
      - Predicted optimality:     |local_value - nn_pred|     <= tol

    A "positive" = certified optimal. Returns counts plus false positive /
    false negative rates:
      - False positive: NN certifies optimal, but local solution is actually
        suboptimal (relaxation is not tight at the local solution's value).
      - False negative: NN fails to certify an actually-optimal local solution.
    """
    relax_value = np.asarray(relax_value).reshape(-1)
    local_value = np.asarray(local_value).reshape(-1)
    nn_pred = np.asarray(nn_pred).reshape(-1)

    actual_optimal = local_value - relax_value <= tol
    predicted_optimal = local_value - nn_pred <= tol

    tp = int(np.sum(actual_optimal & predicted_optimal))
    tn = int(np.sum(~actual_optimal & ~predicted_optimal))
    fp = int(np.sum(~actual_optimal & predicted_optimal))
    fn = int(np.sum(actual_optimal & ~predicted_optimal))

    n_pos = tp + fn  # actually optimal
    n_neg = tn + fp  # actually suboptimal

    return dict(
        tp=tp, tn=tn, fp=fp, fn=fn,
        n=tp + tn + fp + fn,
        fpr=fp / n_neg if n_neg > 0 else float("nan"),
        fnr=fn / n_pos if n_pos > 0 else float("nan"),
    )
