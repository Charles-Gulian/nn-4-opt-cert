"""Recompute paper results in the ROC/AUC framework.

For each experiment config we treat the certificate as a binary classifier with
decision score s = f(x_hat;theta) - v_hat(theta) (the NN-predicted optimality
gap) and report the ROC/AUC against a ground-truth optimality label defined by
the exact or best-available optimum v(theta):

    positive (optimal)  iff  f - v(theta) <= delta0,   delta0 = 1e-3 * mean(v_r).

Ground-truth v(theta):
  - qcqp        : SDP relaxation is exact (S-lemma) => v = v_r (the 'v' column).
  - acopf socp  : chordal-SDP value at the same theta (best available truth).
  - acopf chordal: its own relaxation value (tightest available) => v = v_r.
  - ik / mimo   : handled by compute_roc_auc_small.py (needs analytic / MILP truth).

Writes results/roc_auc_summary.csv with, per config: mean/std v_r, mean relax gap
v - v_r, mean opt gap f - v, % feasible, MAE and (p5,p95) of the NN error g - v_r,
AUC, and pooled ROC arrays saved to results/roc_curves/<config>.npz.

Usage:  python scripts/compute_roc_auc.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nn.metrics import roc_auc_certification

RESULTS = PROJECT_ROOT / "results"
DATA = PROJECT_ROOT / "data"
ACOPF_DATA = DATA / "acopf-hpc"
ROC_DIR = RESULTS / "roc_curves"
LABELS = ("Cost", "Exact", "LocalCost")

ACOPF_CASES = ["case9", "case14", "case39", "case89pegase",
               "case118", "case300", "case1354pegase", "case2869pegase"]


def _feasible_theta(csv_path, feat_cols):
    """Return (theta of feasible rows in order, Cost, LocalCost) matching the
    post-filter prediction indexing (drop rows with NaN Cost or LocalCost)."""
    df = pd.read_csv(csv_path)
    v = pd.to_numeric(df["Cost"], errors="coerce").values
    f = pd.to_numeric(df["LocalCost"], errors="coerce").values
    mask = np.isfinite(v) & np.isfinite(f)
    return df.loc[mask, feat_cols].values.astype(float), v[mask], f[mask]


def _ik_truth(order):
    """IK exact ground truth: the analytic closed-form value function."""
    from problems.ik.problem import ground_truth, DEFAULT_L1, DEFAULT_L2
    key = f"ik_lass{order}"
    theta, v, f = _feasible_theta(DATA / key / "test_5000.csv", ["xd", "yd"])
    args = {"l1": DEFAULT_L1, "l2": DEFAULT_L2}
    v_true = np.array([ground_truth(t, args) for t in theta])
    return v_true


def _mimo_truth():
    """MIMO exact ground truth: brute-force ML detection over x in {-1,1}^n."""
    from problems.mimo_detection.problem import A_REAL, N_TRANSMITTERS
    import itertools
    b_cols = [f"b{i}" for i in range(A_REAL.shape[0])]
    theta, v, f = _feasible_theta(DATA / "mimo" / "test_5000.csv", b_cols)
    cands = np.array(list(itertools.product([-1.0, 1.0], repeat=N_TRANSMITTERS)))  # (4, n)
    Hc = (A_REAL @ cands.T).T                    # (4, 2m): H x for each candidate
    # v_true[i] = min_x ||H x - y_i||^2
    d = theta[:, None, :] - Hc[None, :, :]       # (N, 4, 2m)
    return (d ** 2).sum(axis=2).min(axis=1)


def _acopf_chordal_truth(case, n_socp_rows):
    """Return chordal-SDP value aligned to the SOCP post-filter prediction rows.

    SOCP and chordal share the same sampled thetas (same X seed), in original row
    order. The SOCP predictions are indexed by position among SOCP-feasible rows,
    so we map each back to its original row k and read the chordal Cost at k.
    """
    socp_csv = ACOPF_DATA / f"test_5000_socp_{case}.csv"
    chordal_csv = ACOPF_DATA / f"test_5000_chordal_sdp_{case}.csv"
    if not chordal_csv.exists():
        return None
    sd = pd.read_csv(socp_csv)
    cd = pd.read_csv(chordal_csv)
    v = pd.to_numeric(sd["Cost"], errors="coerce").values
    f = pd.to_numeric(sd["LocalCost"], errors="coerce").values
    mask = np.isfinite(v) & np.isfinite(f)            # SOCP eval's drop rule
    orig_idx = np.where(mask)[0]
    if len(orig_idx) != n_socp_rows:
        # Local raw SOCP CSV is out of sync with the predictions (the 3 regenerated
        # PEGASE cases: their raw CSVs live in data/acopf on SAVIO, not pulled). Can't
        # align the chordal-truth join without the synced CSV.
        print(f"  SKIP chordal-truth for {case}: raw SOCP CSV has {len(orig_idx)} "
              f"feasible rows but predictions have {n_socp_rows} -- needs synced "
              f"data/acopf/test_5000_socp_{case}.csv from SAVIO.", flush=True)
        return None
    chordal_cost = pd.to_numeric(cd["Cost"], errors="coerce").values
    return chordal_cost[orig_idx]                      # nan where chordal infeasible


def _row(config, exp, relax, case, g, v, f, v_true, n_raw):
    """Assemble one summary row from pooled per-instance arrays."""
    delta0 = 1e-3 * np.nanmean(v)
    err = g - v                                        # NN error vs its target v_r
    ae = np.abs(err)
    roc = roc_auc_certification(f, g, v_true, delta0)
    ROC_DIR.mkdir(parents=True, exist_ok=True)
    if np.isfinite(roc["auc"]):
        np.savez(ROC_DIR / f"{config}.npz", fpr=roc["fpr"], tpr=roc["tpr"],
                 tau=roc["tau"], auc=roc["auc"], exp=exp)
    relax_gap = v_true - v                             # v(theta) - v_r(theta)
    opt_gap = f - v_true                               # f - v(theta)
    n_pred = len(g) // 4                               # per-fold count (4 folds pooled)
    return dict(
        config=config, experiment=exp, relaxation=relax, case=case,
        mean_vr=np.nanmean(v), std_vr=np.nanstd(v),
        mean_relax_gap=np.nanmean(relax_gap), mean_opt_gap=np.nanmean(opt_gap),
        pct_feasible=100.0 * n_pred / n_raw,
        mae=ae.mean(), ae_p5=np.percentile(ae, 5), ae_p95=np.percentile(ae, 95),
        delta0=delta0, auc=roc["auc"], n_pos=roc["n_pos"], n_neg=roc["n_neg"],
    )


def main():
    rows = []

    # ---- QCQP (exact relaxation: v_true = v_r) ----
    d = pd.read_csv(RESULTS / "qcqp" / "fold_test_predictions.csv")
    rows.append(_row("qcqp", "QCQP", "SDP", "--",
                     d["g"].values, d["v"].values, d["f"].values,
                     d["v"].values, n_raw=5000))

    # ---- MIMO (exact ML detection by brute force) ----
    d = pd.read_csv(RESULTS / "mimo" / "fold_test_predictions.csv")
    rows.append(_row("mimo", "MIMO", "SDP", "--",
                     d["g"].values, d["v"].values, d["f"].values,
                     np.tile(_mimo_truth(), 4), n_raw=5000))

    # ---- IK (exact analytic value function), both relaxation orders ----
    for order in (1, 2):
        key = f"ik_lass{order}"
        d = pd.read_csv(RESULTS / key / "fold_test_predictions.csv")
        rows.append(_row(key, "IK", f"Lasserre-{order}", "--",
                         d["g"].values, d["v"].values, d["f"].values,
                         np.tile(_ik_truth(order), 4), n_raw=5000))

    # ---- AC-OPF ----
    for relax in ("socp", "chordal_sdp"):
        for case in ACOPF_CASES:
            cfg = f"acopf_{relax}_{case}"
            pred = RESULTS / "acopf-cert" / cfg / "fold_test_predictions.csv"
            if not pred.exists():
                continue
            d = pd.read_csv(pred)
            g, v, f = d["g"].values, d["v"].values, d["f"].values
            n_per_fold = (d["fold"] == 0).sum()
            if relax == "chordal_sdp":
                v_true = v                              # tightest available truth
            elif case == "case2869pegase":
                v_true = v                              # no chordal: SOCP-as-truth (flagged)
            else:
                truth = _acopf_chordal_truth(case, n_per_fold)
                if truth is None:
                    # chordal exists but raw SOCP CSV is stale -> genuinely pending a
                    # data pull; skip rather than report a self-truth AUC.
                    continue
                v_true = np.tile(truth, 4)              # 4 folds pooled, same order
            rows.append(_row(cfg, "AC-OPF", relax.upper().replace("_SDP", " SDP"),
                             case, g, v, f, v_true, n_raw=5000))

    out = pd.DataFrame(rows)
    out_path = RESULTS / "roc_auc_summary.csv"
    out.to_csv(out_path, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(out[["config", "mean_vr", "mean_relax_gap", "mean_opt_gap",
               "pct_feasible", "mae", "auc", "n_pos", "n_neg"]].to_string(index=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
