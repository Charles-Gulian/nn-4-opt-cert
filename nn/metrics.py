"""Error-summary utilities for comparing predicted vs. true optimal values."""

import numpy as np

_Z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}


def mean_ci(values, confidence=0.95):
    """Mean and a two-sided normal confidence interval on the mean.

    NaN-safe: non-finite entries are dropped before computing statistics, so
    this can be applied directly to per-sample metric arrays that may contain
    NaNs from infeasible samples.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return dict(n=0, mean=float("nan"),
                    ci_lower=float("nan"), ci_upper=float("nan"))
    mean = float(v.mean())
    sem = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    z = _Z.get(confidence, 1.96)
    return dict(n=n, mean=mean, ci_lower=mean - z * sem, ci_upper=mean + z * sem)


def overprediction_summary(pred, target, q=0.95):
    """Worst-case OVER-prediction of `target` by `pred`, in signed percent.

    For optimality certification, over-prediction (pred > target, where target is
    the relaxation's optimal value / lower bound) is the dangerous direction: it
    inflates the bound and can falsely certify a sub-optimal solution.  If the NN
    never over-predicts by more than e%, then any solution it certifies optimal
    has a true optimality gap below e% + the NN's relative tolerance.

    over_pct = 100 * (pred - target) / target   (positive = over-prediction)

    Returns the maximum over_pct (worst case) and its q-quantile (upper tail),
    both NaN-safe (non-finite and non-positive-target samples are dropped).
    """
    pred = np.asarray(pred, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    ok = np.isfinite(pred) & np.isfinite(target) & (target > 0)
    if not ok.any():
        return dict(max_overpred_pct=float("nan"),
                    q_overpred_pct=float("nan"), q=q, n=0)
    over = 100.0 * (pred[ok] - target[ok]) / target[ok]
    return dict(
        max_overpred_pct=float(np.max(over)),
        q_overpred_pct=float(np.percentile(over, 100.0 * q)),
        q=q, n=int(ok.sum()),
    )


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


def optimality_confusion_matrix(relax_value, local_value, nn_pred, tol=1e-2,
                                relative=False):
    """Confusion matrix for using the NN to certify optimality of a local solver.

    "Optimal" means the local solution's value is close to the relaxation's lower
    bound (the bound is tight). Because the local value is an upper bound and the
    relaxation a lower bound, ``local - bound >= 0``, so we use a one-sided test.

    Two tolerance modes:
      - absolute (default): tight  <=>  local_value - bound <= tol
      - relative (relative=True): tight  <=>  (local_value - bound) / bound <= tol
        i.e. the local value is within ``tol`` (a fraction) above the bound.

    The "bound" is ``relax_value`` for the ground-truth test and ``nn_pred`` for
    the predicted test:
      - Ground truth optimality:  uses relax_value
      - Predicted optimality:     uses nn_pred

    A "positive" = certified optimal. Returns counts plus false positive /
    false negative rates:
      - False positive: NN certifies optimal, but the relaxation is not actually
        tight at the local solution's value (dangerous: would prune the optimum).
      - False negative: NN fails to certify an actually-optimal local solution.
    """
    relax_value = np.asarray(relax_value, dtype=float).reshape(-1)
    local_value = np.asarray(local_value, dtype=float).reshape(-1)
    nn_pred = np.asarray(nn_pred, dtype=float).reshape(-1)

    if relative:
        # Guard non-positive denominators (costs are positive in practice).
        safe_relax = np.where(relax_value > 0, relax_value, np.nan)
        safe_pred = np.where(nn_pred > 0, nn_pred, np.nan)
        actual_optimal = (local_value - relax_value) / safe_relax <= tol
        predicted_optimal = (local_value - nn_pred) / safe_pred <= tol
    else:
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
