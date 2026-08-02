"""Figure: the alpha-delta tradeoff at a fixed certification level tau (QCQP).

The certificate declares delta-optimality at risk alpha when
    f(x_hat; theta) <= v_hat(theta) - q_alpha + delta,
where q_alpha is the one-sided split-conformal offset from the calibration
residuals e = g - v (NN prediction minus relaxation value):
    q_alpha = conformal_offset(e, level = 1 - alpha)
            = the ceil((n+1)(1-alpha))-th smallest residual.

Define the certification level  tau = delta - q_alpha  (the slack in the
binary certificate rule f - g < tau). For a FIXED tau, there is a whole
family of (alpha, delta) pairs that achieve it:
    delta(alpha; tau) = tau + q_alpha.
Sweeping alpha traces one curve per tau. Because the curves differ only by
the additive constant tau, they are vertical translations of the single
q_alpha(alpha) profile -- which is exactly the point: at a fixed certified
level you trade risk (alpha) against the optimality gap (delta), and raising
tau shifts the whole tradeoff curve upward.

Usage:
    /opt/anaconda3/envs/nn4opt/bin/python scripts/plot_alpha_delta_tradeoff.py

The plotting body is intentionally self-contained so it can be dropped into
the paper's plotting notebook for further tweaking.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nn.metrics import conformal_offset

FIGURES_DIR = PROJECT_ROOT / "figures"

# --- knobs (mirrored in the notebook cell) -------------------------------
CONFIG = "ik_lass2"       # results/<CONFIG>/val_residuals.csv
CONFIG_LABEL = "IK, Lasserre-2"
# tau values chosen near this experiment's residual scale so the alpha-elbow
# is visible rather than dwarfed by a large tau offset.
TAUS = [0.0005, 0.001, 0.002, 0.005, 0.01]
N_ALPHA = 400            # density of the alpha sweep
ALPHA_MIN, ALPHA_MAX = 1e-3, 0.2   # plotted risk range
BASE_COLOR = "steelblue"
LINEWIDTH = 3.0
# Opacity DECREASES as tau increases (higher tau -> more transparent).
OPACITY_HI, OPACITY_LO = 1.0, 0.35


def load_calibration_residuals(config=CONFIG):
    """Pool all per-fold validation residuals into one calibration set."""
    df = pd.read_csv(PROJECT_ROOT / "results" / config / "val_residuals.csv")
    e = df["residual"].to_numpy(dtype=float)
    return e[np.isfinite(e)]


def q_alpha_curve(residuals, alphas):
    """q_alpha for each alpha, using the exact order-statistic offset at
    level = 1 - alpha (matches nn.metrics.conformal_offset / the certificate)."""
    return np.array([conformal_offset(residuals, 1.0 - a) for a in alphas])


def alpha_delta_curves(residuals, taus, alphas):
    """Return {tau: delta_array} with delta = tau + q_alpha."""
    q = q_alpha_curve(residuals, alphas)
    return {tau: tau + q for tau in taus}, q


def plot_alpha_delta(residuals, taus=TAUS, n_alpha=N_ALPHA,
                     alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX,
                     base_color=BASE_COLOR, linewidth=LINEWIDTH,
                     opacity_hi=OPACITY_HI, opacity_lo=OPACITY_LO, ax=None):
    """Self-contained plotting body (drop-in for the notebook)."""
    alphas = np.linspace(alpha_min, alpha_max, n_alpha)
    curves, _ = alpha_delta_curves(residuals, taus, alphas)
    opacities = np.linspace(opacity_hi, opacity_lo, len(taus))

    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 5.0))
    for tau, opacity in zip(taus, opacities):
        ax.plot(alphas, curves[tau], color=base_color, linewidth=linewidth,
                alpha=opacity, label=rf"$\tau = {tau:g}$", solid_capstyle="round")

    ax.set_xlabel(r"risk level $\alpha$", fontsize=13)
    ax.set_ylabel(r"optimality gap $\delta$", fontsize=13)
    ax.set_title(r"$\alpha$--$\delta$ tradeoff at fixed certification level $\tau$"
                 f"\n({CONFIG_LABEL} calibration residuals)", fontsize=13)
    ax.set_xlim(alpha_min, alpha_max)
    ax.legend(title="certification level", frameon=False, fontsize=11)
    ax.grid(True, alpha=0.25)
    return ax


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    residuals = load_calibration_residuals()
    print(f"Loaded {len(residuals)} {CONFIG_LABEL} calibration residuals "
          f"(range [{residuals.min():.4g}, {residuals.max():.4g}]).", flush=True)

    ax = plot_alpha_delta(residuals)
    fig = ax.get_figure()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = FIGURES_DIR / f"alpha_delta_tradeoff.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
