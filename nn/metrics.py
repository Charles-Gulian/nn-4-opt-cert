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


def conformal_offset(residuals, level):
    """One-sided split-conformal offset from OVER-prediction residuals.

    residuals = g - v (NN prediction minus relaxation value) on a calibration set.
    Returns t = the exact finite-sample order statistic giving
        Pr(g(theta) - v(theta) <= t) >= level
    for a fresh exchangeable draw, namely the ceil((n+1)*level)-th smallest residual
    (1-indexed, clamped to n). This is NOT np.quantile's interpolated value -- the
    order statistic is what carries the marginal coverage guarantee exactly. Mirrors
    problems/robust_knapsack/conformal.py:compute_margin (there parameterized by
    tau = 1 - level).
    """
    e = np.asarray(residuals, dtype=float).reshape(-1)
    e = e[np.isfinite(e)]
    n = len(e)
    if n == 0:
        return float("nan")
    e_sorted = np.sort(e)
    k = int(np.ceil((n + 1) * level))
    k = min(k, n)
    return float(e_sorted[k - 1])


def prediction_error_summary(pred, target, percentiles=(1, 5, 10, 90, 95, 99)):
    """Distributional summary of the NN's relaxation-value prediction error.

    Works on the SIGNED error e = pred - target (target = relaxation value v, the
    quantity the NN is trained to predict). The sign matters for certification --
    over-prediction (e > 0) is the direction that can inflate the conformal lower
    bound -- so we report signed percentiles rather than absolute ones.

    Returns MAPE (mean absolute percentage error, in %), the requested signed-error
    percentiles, and the signed min/max. NaN-safe: non-finite rows and (for MAPE)
    zero targets are dropped.
    """
    pred = np.asarray(pred, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    ok = np.isfinite(pred) & np.isfinite(target)
    if not ok.any():
        out = dict(n=0, mape=float("nan"),
                   min_error=float("nan"), max_error=float("nan"))
        out.update({f"p{q}": float("nan") for q in percentiles})
        return out

    e = pred[ok] - target[ok]                      # signed error
    nz = target[ok] != 0
    mape = float(np.mean(np.abs(e[nz] / target[ok][nz])) * 100.0) if nz.any() else float("nan")

    out = dict(
        n=int(ok.sum()),
        mape=mape,
        min_error=float(e.min()),
        max_error=float(e.max()),
    )
    for q in percentiles:
        out[f"p{q}"] = float(np.percentile(e, q))
    return out


def certification_confusion(relax_value, local_value, nn_pred, offset, delta):
    """Confusion matrix for the conformal optimality certificate.

    For each instance we hold three numbers (minimization sense):
      relax_value v -- convex relaxation, a LOWER bound on the true optimum,
      local_value f -- local solver, a feasible UPPER bound,
      nn_pred     g -- NN's prediction of v.

    `offset` is a split-conformal offset t such that Pr(g - v <= t) >= level, so
    LB = g - t is a probabilistic LOWER bound on v (hence on the true optimum).

    Certificate (relative-gap form): certify the local solution delta-optimal iff
        f <= (g - t) * (1 + delta),
    i.e. its relative gap to the certified lower bound LB is at most delta. This is
    the safe direction: since LB <= v <= v*, f <= LB(1+delta) implies the true
    relative gap is also <= delta.

    Ground-truth event (what we score against): the solution really is delta-optimal
    iff its relative gap to the relaxation value is at most delta,
        (f - v) / v <= delta.

    A "positive" = certified. Returns counts + the four rates:
      TPR = TP/(TP+FN)  correctly certified among truly delta-optimal
      FPR = FP/(FP+TN)  falsely certified among not-delta-optimal (the risk the
                        conformal level is meant to bound -- should track ~1-level)
      TNR = TN/(TN+FP)
      FNR = FN/(FN+TP)
    NaN-safe: non-finite rows and non-positive relax_value (bad denominator) dropped.
    """
    v = np.asarray(relax_value, dtype=float).reshape(-1)
    f = np.asarray(local_value, dtype=float).reshape(-1)
    g = np.asarray(nn_pred, dtype=float).reshape(-1)
    t = float(offset)

    ok = np.isfinite(v) & np.isfinite(f) & np.isfinite(g) & (v > 0)
    v, f, g = v[ok], f[ok], g[ok]

    lb = g - t
    certified = f <= lb * (1.0 + delta)
    truth = (f - v) / v <= delta

    tp = int(np.sum(certified & truth))
    tn = int(np.sum(~certified & ~truth))
    fp = int(np.sum(certified & ~truth))
    fn = int(np.sum(~certified & truth))

    n_pos = tp + fn   # truly delta-optimal
    n_neg = tn + fp   # truly not delta-optimal
    return dict(
        tp=tp, tn=tn, fp=fp, fn=fn, n=tp + tn + fp + fn,
        tpr=tp / n_pos if n_pos > 0 else float("nan"),
        fpr=fp / n_neg if n_neg > 0 else float("nan"),
        tnr=tn / n_neg if n_neg > 0 else float("nan"),
        fnr=fn / n_pos if n_pos > 0 else float("nan"),
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


def roc_auc_certification(local_value, nn_pred, true_value, delta0):
    """ROC curve + AUC for the NN certification classifier.

    The certificate rule ``f - g < tau`` (with tau = delta - q_alpha) is a binary
    classifier whose decision score is the NN-predicted optimality gap

        s = f(x_hat; theta) - g(theta),   g = v_hat(theta)   (lower s => more optimal)

    swept over the threshold tau. Rather than fix a single (delta, alpha) we
    report the whole ROC and its AUC. The ground-truth positive class is "the
    local solution really is (delta0-)optimal", measured against the exact or
    best-available optimum ``true_value`` v(theta):

        y = 1  iff  f(x_hat; theta) - v(theta) <= delta0.

    Parameters
    ----------
    local_value : array   f, the local/heuristic solver cost (upper bound).
    nn_pred     : array   g = v_hat, the NN's value prediction.
    true_value  : array   v(theta), exact/best-available optimum (ground truth).
    delta0      : float   absolute tolerance defining the positive (optimal) class.

    Returns
    -------
    dict with:
      auc            AUC (P a truly-optimal instance is scored more optimal than a
                     truly-suboptimal one); nan if only one class is present.
      fpr, tpr       ROC arrays.
      tau            threshold on s = f - g corresponding to each (fpr, tpr) point,
                     so a chosen operating point maps back to tau = delta - q_alpha.
      n_pos, n_neg   class sizes; n total scored (finite, dropped otherwise).
    NaN-safe: non-finite rows in any of f/g/v are dropped.
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    f = np.asarray(local_value, dtype=float).reshape(-1)
    g = np.asarray(nn_pred, dtype=float).reshape(-1)
    vt = np.asarray(true_value, dtype=float).reshape(-1)

    ok = np.isfinite(f) & np.isfinite(g) & np.isfinite(vt)
    f, g, vt = f[ok], g[ok], vt[ok]

    s = f - g                       # predicted optimality gap (decision score)
    y = (f - vt <= delta0).astype(int)   # 1 = truly (delta0-)optimal

    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return dict(auc=float("nan"), fpr=np.array([]), tpr=np.array([]),
                    tau=np.array([]), n_pos=n_pos, n_neg=n_neg, n=len(y))

    # Higher score => more likely positive (optimal); s is lower for optimal,
    # so rank by -s. roc_curve thresholds are on the score -s; tau on s is -thr.
    score = -s
    fpr, tpr, thr = roc_curve(y, score)
    auc = roc_auc_score(y, score)
    return dict(auc=float(auc), fpr=fpr, tpr=tpr, tau=-thr,
                n_pos=n_pos, n_neg=n_neg, n=len(y))
