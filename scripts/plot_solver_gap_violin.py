"""Figure: distribution of local-solver optimality gap, per experiment / solver.

The gap g = f - v >= 0 (min-sense; v - f for knapsack-max) measures how far the
fast local solver's feasible value f sits above the optimum v. Its SHAPE decides
how we choose the certification tolerance delta (Part C of the plan):
  * bimodal -- an atom at g~0 (solver found the global optimum) plus a separated
    positive mode (solver landed in a suboptimal basin), with a valley between --
    => delta goes in the valley;
  * continuous/unimodal from 0 -- no valley => delta is a policy choice.

Two features make a plain violin useless here: (1) a large POINT MASS at g=0
(many instances solved exactly), and (2) a positive tail spanning several orders
of magnitude. So we plot a one-sided (half) violin of log10(gap/std(v)) over the
strictly-positive gaps, and annotate the fraction solved exactly-optimal
separately (it cannot live on a log axis).

Gaps are normalized by std(v) so experiments of vastly different magnitude share
one axis (matching the delta = frac*std(v) scale).

NOTE on v: for QCQP/MIMO/IK we use v = the value the NN predicts (the relaxation
value v_r, == exact optimum where the relaxation is tight). For AC-OPF, v is the
SOCP/chordal relaxation value (Cost), so f - v also carries the relaxation gap,
not pure solver suboptimality -- flagged in the labels.

Usage:
    /opt/anaconda3/envs/nn4opt/bin/python scripts/plot_solver_gap_violin.py
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS = PROJECT_ROOT / "results"

# --- knobs -----------------------------------------------------------------
ZERO_EPS = 1e-4          # gap/std <= ZERO_EPS counts as "solved exactly optimal"
LOG_FLOOR = 1e-4         # clip positive gap/std at this floor before log10
VIOLIN_COLOR = "steelblue"
ATOM_COLOR = "0.35"
# AC-OPF: the stored rank-check `Exact` flag is unusable (all False -- the 1e-4
# per-bag rank tolerance is far below CLARABEL's interior-point precision, whose
# 2nd/1st bag-eigenvalue ratio is ~2e-2). Instead we certify exactness by a
# SANDWICH: v_chordal (lower bound) meets f_IPOPT (feasible upper bound) within
# SANDWICH_TOL relative => v_true is pinned == v_chordal (relaxation exact AND
# IPOPT optimal). Points failing the sandwich are discarded (small, sound bias).
SANDWICH_TOL = 1e-3
ACOPF_DATA = PROJECT_ROOT / "data" / "acopf-hpc"


def _gap_over_std(f, v, maximize=False):
    """Non-negative solver gap normalized by std(v). Returns (gap_norm, std_v)."""
    f = np.asarray(f, float); v = np.asarray(v, float)
    gap = (v - f) if maximize else (f - v)
    ok = np.isfinite(gap) & np.isfinite(v)
    gap, v = gap[ok], v[ok]
    sv = np.nanstd(v)
    gap = np.maximum(gap, 0.0)          # clip tiny negative numerical noise
    return (gap / sv if sv > 0 else gap), sv


def _small_problem_gap(key):
    # dedup: gap = f - v does not depend on fold, so take one fold's rows only
    d = pd.read_csv(RESULTS / key / "fold_test_predictions.csv")
    d = d[d["fold"] == d["fold"].min()]
    return _gap_over_std(d["f"].values, d["v"].values)


def _acopf_gap(case, n_test=5000):
    """AC-OPF gap against the chordal SDP value, restricted to sandwich-certified
    points (v_chordal within SANDWICH_TOL of the IPOPT feasible cost => exact)."""
    d = pd.read_csv(ACOPF_DATA / f"test_{n_test}_chordal_sdp_{case}.csv")
    v = pd.to_numeric(d["Cost"], errors="coerce").values       # chordal lower bound
    f = pd.to_numeric(d["LocalCost"], errors="coerce").values  # IPOPT feasible cost
    ok = np.isfinite(v) & np.isfinite(f) & (np.abs(v) > 1e-9)
    v, f = v[ok], f[ok]
    rel = (f - v) / np.abs(v)
    exact = (rel >= -1e-6) & (rel <= SANDWICH_TOL)             # sandwich certificate
    print(f"    [{case}] sandwich-exact kept {exact.mean()*100:.1f}% of {len(v)} pts", flush=True)
    v, f = v[exact], f[exact]
    return _gap_over_std(f, v)


def _knapsack_gap(n_train=20000):
    # knapsack computes f (greedy) live; v = exact Cost (MAX-sense)
    from scripts.compute_roc_auc_knapsack import _greedy_values
    from problems.registry import LABEL_COLS
    df = pd.read_csv(PROJECT_ROOT / "data" / "robust_knapsack" / "test_20000.csv")
    df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce")
    df = df[np.isfinite(df["Cost"])].reset_index(drop=True)
    feat = [c for c in df.columns if c not in LABEL_COLS]
    f = _greedy_values(df, feat)
    return _gap_over_std(f, df["Cost"].values, maximize=True)


def half_violin(ax, x0, log_gaps, width=0.36, color=VIOLIN_COLOR):
    """Draw a right-facing half-violin (KDE of log_gaps) at categorical x0."""
    if len(log_gaps) < 5:
        return
    from scipy.stats import gaussian_kde
    lo, hi = log_gaps.min(), log_gaps.max()
    ys = np.linspace(lo, hi, 200)
    dens = gaussian_kde(log_gaps)(ys)
    dens = dens / dens.max() * width
    ax.fill_betweenx(ys, x0, x0 + dens, color=color, alpha=0.85, lw=0)
    # median tick
    med = np.median(log_gaps)
    ax.plot([x0, x0 + width * 0.5], [med, med], color="white", lw=1.5)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # (label, callable -> (gap_norm, std_v))
    experiments = [
        ("QCQP\n(IPOPT)", lambda: _small_problem_gap("qcqp")),
        ("MIMO\n(zero-forcing)", lambda: _small_problem_gap("mimo")),
        ("IK Shor\n(IPOPT)", lambda: _small_problem_gap("ik_lass1")),
        ("IK Lass-2\n(IPOPT)", lambda: _small_problem_gap("ik_lass2")),
        ("Knapsack\n(greedy)", _knapsack_gap),
        ("AC-OPF c300\n(IPOPT, chordal)", lambda: _acopf_gap("case300")),
    ]

    fig, ax = plt.subplots(figsize=(11, 6.0))
    labels = []
    # atom (exact-fraction) sits on its own baseline strip below the log axis
    y_base = np.log10(LOG_FLOOR) - 0.6
    y_top = 1.6
    for i, (label, fn) in enumerate(experiments):
        try:
            g, sv = fn()
        except Exception as e:
            print(f"  [skip] {label!r}: {e}", flush=True)
            continue
        labels.append(label)
        frac0 = float(np.mean(g <= ZERO_EPS))
        pos = g[g > ZERO_EPS]
        log_pos = np.log10(np.maximum(pos, LOG_FLOOR))
        half_violin(ax, i, log_pos)
        # atom: a bar at the baseline whose width is proportional to the exact fraction
        ax.add_patch(plt.Rectangle((i, y_base), 0.36 * frac0, 0.16,
                                   color=ATOM_COLOR, alpha=0.9, lw=0))
        ax.annotate(f"{frac0*100:.0f}% exact", (i, y_base - 0.18),
                    ha="left", va="top", fontsize=9, color=ATOM_COLOR)
        # positive-gap fraction above the violin
        ax.annotate(f"{(1-frac0)*100:.0f}% subopt.", (i, y_top - 0.05),
                    ha="left", va="top", fontsize=9, color=VIOLIN_COLOR)
        print(f"  {label.replace(chr(10),' '):28s} n={len(g):6d} std(v)={sv:.4g}  "
              f"frac_exact={frac0:.3f}  median_pos_gap/std={np.median(pos) if len(pos) else float('nan'):.4g}",
              flush=True)

    ax.axhline(np.log10(LOG_FLOOR) - 0.35, color="0.7", lw=0.8)  # separates atom strip from log axis
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(y_base - 0.7, y_top)
    ax.set_ylabel(r"$\log_{10}[\,(f-v)/\mathrm{std}(v)\,]$  (positive gaps)", fontsize=12)
    ax.set_title("Local-solver optimality-gap distribution by experiment\n"
                 "(bar = fraction solved exactly optimal; violin = suboptimal-gap sizes)", fontsize=12)
    ax.axhline(0.0, color="0.8", lw=0.8, ls="--")  # gap == std(v)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    out = FIGURES_DIR / "solver_gap_violin.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
