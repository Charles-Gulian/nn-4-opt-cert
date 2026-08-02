"""Figure: certification operating characteristic (IK, Lasserre-2).

Motivation. At a FIXED certified level tau = delta - q_alpha, the certificate
rule collapses to  f(x_hat;theta) - g(theta) <= tau, so tau alone determines
which instances are certified -- the (alpha, delta) split is a reparametrization
of a single degree of freedom and nothing observable changes along it. The
interesting, data-driven tradeoff appears when we let the RULE move: fix the
risk level alpha and sweep the claimed optimality gap delta.

This sweeps delta (as a fraction of std(v)) at alpha = 1% (per-fold q_99) and
plots, on the held-out test set:
  * POWER  = TPR = fraction of truly delta-optimal solutions we certify,
  * empirical FPR = fraction of NOT-delta-optimal solutions we wrongly certify,
which the split-conformal construction should hold at or below alpha.

Conventions match scripts/compute_final_tables.py exactly (absolute certificate
f <= g - q_99 + delta, ground truth f - v <= delta, delta = frac * std(v),
per-fold offsets, counts pooled across folds).

Usage:
    /opt/anaconda3/envs/nn4opt/bin/python scripts/plot_certification_operating_curve.py
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FIGURES_DIR = PROJECT_ROOT / "figures"

# --- knobs (mirrored in the notebook cell) -------------------------------
CONFIG = "ik_lass2"
CONFIG_LABEL = "IK, Lasserre-2"
ALPHA = 0.01                       # fixed risk level -> conformal level 0.99
LEVEL = 1.0 - ALPHA
DELTA_FRAC_MAX = 0.10              # sweep delta from 0 to this fraction of std(v)
N_DELTA = 300
OPERATING_DELTA_FRAC = 0.03        # the paper's operating point, marked on the plot
POWER_COLOR = "steelblue"
FPR_COLOR = "crimson"
LINEWIDTH = 3.0


def load_fold_data(config=CONFIG, level=LEVEL):
    """Return (preds_df with g,v,f,fold) and {fold: q_offset at `level`}."""
    d = PROJECT_ROOT / "results" / config
    preds = pd.read_csv(d / "fold_test_predictions.csv")
    off = pd.read_csv(d / "conformal_offsets.csv")
    q = off[np.isclose(off["level"], level)].set_index("fold")["offset"].to_dict()
    return preds, q


def operating_curves(preds, q_by_fold, delta_fracs):
    """Pool TP/FP/TN/FN across folds at each delta; return power (TPR) and FPR
    arrays aligned with delta_fracs. delta = frac * std(v) with v pooled, matching
    compute_final_tables._standard_row."""
    v_all = preds["v"].to_numpy(float)
    std_v = np.nanstd(v_all)
    powers, fprs = [], []
    for frac in delta_fracs:
        delta = frac * std_v
        tp = fp = tn = fn = 0
        for fold, sub in preds.groupby("fold"):
            g = sub["g"].to_numpy(float); v = sub["v"].to_numpy(float); f = sub["f"].to_numpy(float)
            ok = np.isfinite(g) & np.isfinite(v) & np.isfinite(f)
            g, v, f = g[ok], v[ok], f[ok]
            cert = f <= g - q_by_fold[fold] + delta
            truth = (f - v) <= delta
            tp += int(np.sum(cert & truth)); fp += int(np.sum(cert & ~truth))
            tn += int(np.sum(~cert & ~truth)); fn += int(np.sum(~cert & truth))
        powers.append(tp / (tp + fn) if (tp + fn) else np.nan)
        fprs.append(fp / (fp + tn) if (fp + tn) else np.nan)
    return np.array(powers), np.array(fprs), std_v


def plot_operating_curve(preds, q_by_fold, alpha=ALPHA,
                         delta_frac_max=DELTA_FRAC_MAX, n_delta=N_DELTA,
                         operating=OPERATING_DELTA_FRAC, ax=None):
    delta_fracs = np.linspace(0.0, delta_frac_max, n_delta)
    power, fpr, _ = operating_curves(preds, q_by_fold, delta_fracs)
    x = delta_fracs * 100.0  # percent of std(v)

    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(x, power, color=POWER_COLOR, lw=LINEWIDTH, label="power (certified fraction)")
    ax.plot(x, fpr, color=FPR_COLOR, lw=LINEWIDTH, label="empirical FPR")
    ax.axhline(alpha, color=FPR_COLOR, ls=":", lw=1.5,
               label=rf"target risk $\alpha={alpha:g}$")
    if operating is not None:
        ax.axvline(operating * 100.0, color="0.4", ls="--", lw=1.2)
        ax.text(operating * 100.0, 0.5, f"  operating\n  point ({operating*100:g}%)",
                fontsize=9, color="0.3", va="center")
    ax.set_xlabel(r"optimality gap $\delta$  (\% of $\mathrm{std}(v)$)"
                  .replace("\\%", "%"), fontsize=13)
    ax.set_ylabel("rate on test set", fontsize=13)
    ax.set_title(rf"Certification operating characteristic at $\alpha={alpha:g}$"
                 f"\n({CONFIG_LABEL})", fontsize=13)
    ax.set_xlim(0, delta_frac_max * 100.0)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=11, loc="center right")
    ax.grid(True, alpha=0.25)
    return ax


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    preds, q = load_fold_data()
    print(f"Loaded {len(preds)} {CONFIG_LABEL} test rows across {preds['fold'].nunique()} "
          f"folds; per-fold q_{int(LEVEL*100)} = "
          f"{ {k: round(v, 6) for k, v in q.items()} }.", flush=True)

    ax = plot_operating_curve(preds, q)
    fig = ax.get_figure()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = FIGURES_DIR / f"certification_operating_curve.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
