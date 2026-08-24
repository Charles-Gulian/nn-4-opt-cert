"""Compute the three consolidated results tables for the paper (offline / online /
timing) at the final fixed operating point:

    delta = 0.01 * std(v(theta))     (absolute, per config; v = exact truth or v_r)
    alpha = 0.01  =>  conformal offset q_99   (level 0.99 in conformal_offsets.csv)

Certify delta-optimal iff  f(x_hat;theta) <= v_hat(theta) - q_99 + delta ;
ground truth  f - v(theta) <= delta. TP/FP/TN/FN counted per fold then averaged.

Writes results/table_offline.csv, table_online.csv, table_timing.csv, matching the
row order / columns of table mockup.xlsx.

Usage:  python scripts/compute_final_tables.py
"""
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compute_roc_auc import (_ik_truth, _mimo_truth, _acopf_chordal_truth,
                                     ACOPF_CASES)
from nn.metrics import conformal_offset
from problems.robust_knapsack.conformal import compute_margin

RESULTS = PROJECT_ROOT / "results"
DATA = PROJECT_ROOT / "data"

# Fast local/heuristic solver used at deployment, per experiment.
SOLVER = {
    "QCQP": "IPM", "Inverse Kinematics": "IPM", "AC-OPF": "IPM",
    "MIMO Detection": "zero-forcing", "Robust Knapsack": "greedy",
    "Robust Knapsack (corr.)": "greedy",
}

# (Experiment, Relaxation label, Case label, results-dir key, truth-kind)
# truth-kind: 'self' (v=v_r), 'ik1','ik2','mimo', 'acopf_socp','acopf_chordal', 'knap'
ROWS = [
    ("QCQP", "Shor", "", "qcqp", "self"),
    ("Inverse Kinematics", "Shor", "", "ik_lass1", "ik1"),
    ("Inverse Kinematics", "Lasserre-2", "", "ik_lass2", "ik2"),
    ("MIMO Detection", "Shor", "", "mimo", "mimo"),
]
for case in ACOPF_CASES:
    ROWS.append(("AC-OPF", "SOCP", case, f"acopf_socp_{case}", "acopf_socp"))
for case in ACOPF_CASES:
    if case == "case2869pegase":
        continue  # no chordal SDP
    ROWS.append(("AC-OPF", "SDP", case, f"acopf_chordal_sdp_{case}", "acopf_chordal"))
ROWS.append(("Robust Knapsack", "None", "n=20000", "knapsack_n20000", "knap"))
ROWS.append(("Robust Knapsack", "None", "n=80000", "knapsack_n80000", "knap"))
# Correlated-mu robust knapsack (n=25 items, mu ~ N(c, Sigma) with positive
# correlation): the redesigned experiment. Two training targets -- the exact
# MISOCP optimum and the continuous SOCP relaxation v_r -- at two data sizes,
# from scripts/knapsack_corr_kfold.py.
for _n in (20, 80):
    ROWS.append(("Robust Knapsack (corr.)", "Exact", f"n={_n}000",
                 f"knapsack_corr_exact_{_n}k", "knap_corr"))
    ROWS.append(("Robust Knapsack (corr.)", "SOCP", f"n={_n}000",
                 f"knapsack_corr_relax_{_n}k", "knap_corr"))


def _pred_dir(key):
    p = RESULTS / key
    return p if (p / "fold_test_predictions.csv").exists() else RESULTS / "acopf-cert" / key


ACOPF_RAW_DIR = (DATA / "acopf-hpc" if (DATA / "acopf-hpc").exists() else DATA / "acopf")


def _acopf_own_feasible_mask(case, relax_key):
    """The exact per-instance feasibility mask evaluate_certify.py applies
    when building fold_test_predictions.csv (isfinite Cost & LocalCost),
    computed on the RAW test CSV so it is indexed by ORIGINAL row position
    (0..4999, shared theta order across relaxations) rather than the
    post-filter 0..n-1 reindex that fold_test_predictions.csv itself uses."""
    raw = pd.read_csv(ACOPF_RAW_DIR / f"test_5000_{relax_key}_{case}.csv")
    v = pd.to_numeric(raw["Cost"], errors="coerce").values
    f = pd.to_numeric(raw["LocalCost"], errors="coerce").values
    return np.isfinite(v) & np.isfinite(f)


def _acopf_keep_in_own_order(case, own_relax_key, other_relax_key):
    """Boolean array, in THIS relaxation's own post-filter row order (same
    order fold_test_predictions.csv uses), marking which of its own retained
    rows are ALSO feasible for the paired relaxation on the SAME case.

    WHY: SOCP and SDP each independently drop their own infeasible rows and
    reset to a fresh 0..n-1 index (evaluate_certify.py), discarding which
    original theta each row is. When the two relaxations fail on different
    instances (case89pegase/case300/case1354pegase; verified NOT the case for
    case9/14/39/118, where the feasible SETS are identical), "row i" in the
    SOCP file and "row i" in the SDP file are in general different theta --
    so directly comparing/averaging v_r or f across the two relaxation rows
    silently mixes non-comparable populations (this produced the impossible
    SDP-v_r < SOCP-v_r and the SOCP/SDP f mismatch). Restricting both rows to
    the shared-feasible subset (this function) before computing any
    statistic makes the two rows describe the exact same test population.
    Returns None if there's nothing to intersect against (e.g.
    case2869pegase has no chordal-SDP data at all).
    """
    try:
        own_mask = _acopf_own_feasible_mask(case, own_relax_key)
        other_mask = _acopf_own_feasible_mask(case, other_relax_key)
    except FileNotFoundError:
        return None
    shared = own_mask & other_mask
    return shared[own_mask]   # length == own_mask.sum(), in own post-filter order


def _truth(kind, key, d):
    """v(theta) aligned & pooled to match the (fold-pooled) predictions frame d."""
    if kind in ("self", "acopf_chordal"):
        return d["v"].values
    if kind == "ik1":   return np.tile(_ik_truth(1), 4)
    if kind == "ik2":   return np.tile(_ik_truth(2), 4)
    if kind == "mimo":  return np.tile(_mimo_truth(), 4)
    if kind == "acopf_socp":
        case = key.replace("acopf_socp_", "")
        if case == "case2869pegase":
            return d["v"].values
        n = int((d["fold"] == 0).sum())
        return np.tile(_acopf_chordal_truth(case, n), 4)
    raise ValueError(kind)


def _offsets_q(key, level):
    """Per-fold conformal offset at the given level from conformal_offsets.csv."""
    p = _pred_dir(key) / "conformal_offsets.csv"
    o = pd.read_csv(p)
    return o[np.isclose(o["level"], level)].set_index("fold")["offset"].to_dict()


def _cert_counts(g_by_fold, f, v_true, q99_by_fold, delta):
    """Average TP/FP/TN/FN over folds. certify iff f <= g - q99 + delta;
    truth iff f - v_true <= delta. Rows without a finite ground truth v(theta)
    (e.g. the chordal reference was infeasible) are dropped -- we cannot score a
    certificate we have no ground truth for."""
    ok = np.isfinite(f) & np.isfinite(v_true)
    f, v_true = f[ok], v_true[ok]
    truth = (f - v_true) <= delta
    tp = fp = tn = fn = 0.0
    folds = sorted(g_by_fold)
    for k in folds:
        g = g_by_fold[k][ok]
        cert = f <= g - q99_by_fold[k] + delta
        tp += np.sum(cert & truth); fp += np.sum(cert & ~truth)
        tn += np.sum(~cert & ~truth); fn += np.sum(~cert & truth)
    n = len(folds)
    return dict(TP=tp/n, FP=fp/n, TN=tn/n, FN=fn/n)


def _standard_row(exp, relax, case, key, kind, delta_frac, level):
    d = pd.read_csv(_pred_dir(key) / "fold_test_predictions.csv")
    d0 = d[d["fold"] == 0]
    v = d0["v"].values; f = d0["f"].values
    v_true = _truth(kind, key, d)   # computed on the UNRESTRICTED data -- _acopf_chordal_truth
    vt0 = v_true[:len(d0)]          # internally checks row counts against d0's own full feasible set
    g_by_fold = {k: d[d["fold"] == k]["g"].values for k in d["fold"].unique()}

    # AC-OPF: restrict SOCP/SDP rows for the same case to the subset feasible
    # for BOTH relaxations (see _acopf_keep_in_own_order), so the two rows
    # describe the exact same test population. Without this, SOCP and SDP
    # silently average different theta populations whenever they fail to
    # converge on different instances (verified for case89pegase/case300/
    # case1354pegase; the feasible SETS are identical -- not just
    # equal-count -- for case9/14/39/118, so this is a no-op there), which
    # produced an impossible-looking SDP-v_r < SOCP-v_r and a SOCP/SDP f
    # mismatch despite f being literally the same local solver both times.
    keep = None
    if kind == "acopf_socp" and case != "case2869pegase":
        keep = _acopf_keep_in_own_order(case, "socp", "chordal_sdp")
    elif kind == "acopf_chordal":
        keep = _acopf_keep_in_own_order(case, "chordal_sdp", "socp")
    if keep is not None:
        assert len(keep) == len(d0), f"{key}: keep len {len(keep)} != {len(d0)} rows/fold"
        v, f, vt0 = v[keep], f[keep], vt0[keep]
        g_by_fold = {k: g[keep] for k, g in g_by_fold.items()}

    q = _offsets_q(key, level)
    delta = delta_frac * np.nanstd(vt0)
    # |g - v_r| pooled over folds, from the (possibly keep-restricted) arrays
    # above rather than d directly, so MAE reflects the same restriction.
    err = np.abs(np.concatenate([g_by_fold[k] for k in sorted(g_by_fold)])
                 - np.tile(v, len(g_by_fold)))
    off = dict(
        experiment=exp, relaxation=relax, case=case,
        mean_v=np.nanmean(v), std_v=np.nanstd(v),
        pct_infeasible=100.0 * (1 - len(d0) / 5000.0),   # this relaxation's OWN solve success rate
        mean_relax_gap=np.nanmean(vt0 - v),
        mae=np.nanmean(err), q_offset=np.mean(list(q.values())),
        # unambiguous fields for the v_r / v / f side-by-side table: mean_v
        # above is v_r EXCEPT in _knapsack_corr_row, where it is v_true --
        # these three are always the same quantity regardless of row kind.
        mean_vr=np.nanmean(v), mean_vtrue=np.nanmean(vt0), mean_f=np.nanmean(f),
    )
    opt_gap = f - vt0
    cc = _cert_counts(g_by_fold, f, vt0, q, delta)
    onl = dict(experiment=exp, relaxation=relax, case=case, solver=SOLVER[exp],
               mean_opt_gap=np.nanmean(opt_gap), worst_opt_gap=np.nanmax(opt_gap),
               delta=delta, **{k: round(v) for k, v in cc.items()})
    return off, onl


def _knapsack_corr_row(exp, relax, case, key, kind, delta_frac, level):
    """Correlated-mu robust knapsack (MAX-sense), 4-fold OOF, from
    scripts/knapsack_corr_kfold.py outputs.

    This is the only MAXIMIZATION problem in the suite, so the certificate signs
    are flipped relative to the min problems: vhat + q is a (1-alpha) UPPER bound
    on the target (q = the (1-alpha) quantile of the out-of-fold residual
    target - prediction), and we certify iff  vhat + q - f <= delta, which
    implies v* - f <= delta. For the SOCP-relaxation target the learned quantity
    is v_r >= v*, so the same bound covers v* and the relaxation gap simply adds
    to the effective slack.
    """
    d = pd.read_csv(RESULTS / key / "fold_test_predictions.csv")
    oof = np.load(RESULTS / key / "oof_residuals.npy")
    oof = oof[np.isfinite(oof)]
    q = conformal_offset(oof, level)          # Pr(target <= vhat + q) >= level

    folds = sorted(d["fold"].unique())
    one = d[d["fold"] == folds[0]]
    vstar = one["vstar"].values               # exact optimum (ground truth)
    f_g = one["f"].values                     # greedy local-solver value
    y = one["v"].values                       # the target the NN learned
    delta = delta_frac * np.nanstd(vstar)

    off = dict(experiment=exp, relaxation=relax, case=case,
               mean_v=float(np.mean(vstar)), std_v=float(np.std(vstar)),
               pct_infeasible=0.0,
               mean_relax_gap=float(np.mean(y - vstar)),
               mae=float(np.mean(np.abs(d["g"].values - d["v"].values))),
               q_offset=float(q),
               # unambiguous fields (see _standard_row): here mean_v above is
               # v_true (not v_r), so mean_vr/mean_vtrue must NOT reuse it blindly.
               mean_vr=float(np.mean(y)), mean_vtrue=float(np.mean(vstar)),
               mean_f=float(np.mean(f_g)))

    truth = (vstar - f_g) <= delta
    tp = fp = tn = fn = 0.0
    for k in folds:
        g = d[d["fold"] == k]["g"].values
        cert = (g + q - f_g) <= delta
        tp += np.sum(cert & truth); fp += np.sum(cert & ~truth)
        tn += np.sum(~cert & ~truth); fn += np.sum(~cert & truth)
    nf = len(folds)
    opt_gap = vstar - f_g
    onl = dict(experiment=exp, relaxation=relax, case=case, solver=SOLVER[exp],
               mean_opt_gap=float(opt_gap.mean()), worst_opt_gap=float(opt_gap.max()),
               delta=delta, TP=round(tp/nf), FP=round(fp/nf),
               TN=round(tn/nf), FN=round(fn/nf))
    return off, onl


def _knapsack_row(exp, relax, case, key, kind, delta_frac, level):
    n_train = 20000 if "20000" in case else 80000
    test_name = "test_5000" if n_train == 20000 else "test_20000"
    from scripts.compute_roc_auc_knapsack import _greedy_values
    df = pd.read_csv(DATA / "robust_knapsack" / f"{test_name}.csv")
    feat = [c for c in df.columns if c != "Cost"]
    df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce")
    df = df[np.isfinite(df["Cost"])].reset_index(drop=True)
    cost = df["Cost"].values
    f_g = _greedy_values(df, feat)
    # per-fold NN predictions
    from nn.training import load_checkpoint, predict_denorm
    X = df[feat].values.astype(float)
    models = [load_checkpoint(PROJECT_ROOT / "models" / "robust_knapsack" /
                              f"dnn_knapsack_n{n_train}_fold{k}.pt") for k in range(4)]
    g_by_fold = {k: predict_denorm(m, X, s) for k, (m, s, _) in enumerate(models)}
    # q_99 (MAX upper-bound offset) from OOF residuals: Pr(Cost - vhat <= q) >= 0.99
    oof = np.load(RESULTS / "robust_knapsack" / f"knapsack_oof_residuals_n{n_train}.npy")
    # oof = pred - cost ; residual for upper bound = cost - pred = -oof
    q = conformal_offset(-oof, level)
    delta = delta_frac * np.nanstd(cost)
    err = np.abs(np.concatenate([g_by_fold[k] for k in range(4)]) - np.tile(cost, 4))
    off = dict(experiment=exp, relaxation=relax, case=case,
               mean_v=cost.mean(), std_v=cost.std(), pct_infeasible=0.0,
               mean_relax_gap=np.nan, mae=err.mean(), q_offset=q)
    # MAX certification: certify iff vhat + q - f_g <= delta ; truth Cost - f_g <= delta
    truth = (cost - f_g) <= delta
    tp = fp = tn = fn = 0.0
    for k in range(4):
        cert = g_by_fold[k] + q - f_g <= delta
        tp += np.sum(cert & truth); fp += np.sum(cert & ~truth)
        tn += np.sum(~cert & ~truth); fn += np.sum(~cert & truth)
    opt_gap = cost - f_g
    onl = dict(experiment=exp, relaxation=relax, case=case, solver=SOLVER[exp],
               mean_opt_gap=opt_gap.mean(), worst_opt_gap=opt_gap.max(), delta=delta,
               TP=round(tp/4), FP=round(fp/4), TN=round(tn/4), FN=round(fn/4))
    return off, onl


def _timing_row(exp, relax, case, key, kind):
    """Read solve_times.json where present; blanks otherwise (filled by
    scripts/measure_solve_times.py)."""
    p = (RESULTS / key / "solve_times.json") if kind == "knap" else (_pred_dir(key) / "solve_times.json")
    relax_t = local_t = np.nan
    if p.exists():
        t = json.loads(p.read_text())
        relax_t = t.get("mean_relax_solve_s", np.nan)
        local_t = t.get("mean_local_solve_s", np.nan)
    if kind == "knap":
        relax_t = np.nan   # N/A: no convex relaxation
    return dict(experiment=exp, relaxation=relax, case=case,
                relax_solve_s=relax_t, local_solve_s=local_t)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta-frac", type=float, default=0.01,
                    help="delta = delta_frac * std(v(theta))  (default 0.01 = 1%%)")
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="miscoverage; offset taken at level 1-alpha (default 0.01 -> q_99)")
    args = ap.parse_args()
    level = round(1.0 - args.alpha, 4)
    print(f"delta = {args.delta_frac:.0%} * std(v);  alpha = {args.alpha:.0%}  =>  q_{{{int(level*100)}}}")

    offline, online, timing = [], [], []
    for exp, relax, case, key, kind in ROWS:
        row = {"knap": _knapsack_row,
               "knap_corr": _knapsack_corr_row}.get(kind, _standard_row)
        try:
            off, onl = row(exp, relax, case, key, kind, args.delta_frac, level)
            offline.append(off); online.append(onl)
        except Exception as e:
            print(f"  WARN {key}: {e}")
        timing.append(_timing_row(exp, relax, case, key, kind))
    (RESULTS / "table_meta.json").write_text(json.dumps(
        {"delta_frac": args.delta_frac, "alpha": args.alpha, "level": level}))
    for name, rows in [("offline", offline), ("online", online), ("timing", timing)]:
        out = RESULTS / f"table_{name}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"wrote {out} ({len(rows)} rows)")
    pd.set_option("display.width", 240, "display.max_columns", 30)
    print("\n=== OFFLINE ==="); print(pd.DataFrame(offline).to_string(index=False))
    print("\n=== ONLINE ==="); print(pd.DataFrame(online).to_string(index=False))


if __name__ == "__main__":
    main()
