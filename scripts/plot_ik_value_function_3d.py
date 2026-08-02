"""Figure: 3D value-function surfaces for the IK problem, side by side.

One combined figure, three 3D panels sharing z-limits and view angle so the
surfaces are directly comparable (no colorbar -- each panel's own z-axis
carries the shared scale; only the leftmost panel has a z-axis label):
  (a) the TRUE value function v(xd, yd) -- for IK this has a closed form
      (radial projection onto the reachable annulus, see
      problems/ik/problem.py:ground_truth), so no solver calls are needed;
  (b) the NN-learned value function for the order-1 Lasserre/Shor relaxation
      (config "ik_lass1");
  (c) the NN-learned value function for the order-2 Lasserre relaxation
      (config "ik_lass2", tighter but 28x28 vs 7x7 SDP).

Grid range matches the training distribution (problems/ik/generate_data.py:
sample_parameters draws (xd,yd) uniformly from [-r_max, r_max]^2 with
r_max = (l1+l2)*1.1), so NN predictions stay in-distribution.

Requires the ik_lass1 / ik_lass2 model checkpoints (models/ik_lass1/,
models/ik_lass2/), which are gitignored and may need to be pulled from SAVIO:
    rsync -avz <savio>:~/nn-4-opt-cert/models/ik_lass1 models/
    rsync -avz <savio>:~/nn-4-opt-cert/models/ik_lass2 models/
If they're absent, the corresponding panel is left blank with a note instead
of failing.

Usage:
    /opt/anaconda3/envs/nn4opt/bin/python scripts/plot_ik_value_function_3d.py
"""

import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from problems.ik.problem import DEFAULT_L1, DEFAULT_L2
from nn.training import load_checkpoint, predict_denorm

FIGURES_DIR = PROJECT_ROOT / "figures"

# --- knobs (mirrored in the notebook cell) -------------------------------
L1, L2 = DEFAULT_L1, DEFAULT_L2
R_MAX = (L1 + L2) * 1.1          # matches sample_parameters' sampling square
N_GRID = 100
FOLD = 0                          # which fold's checkpoint to visualize
CMAP = "viridis"
ELEV, AZIM = 25, -60              # shared view angle across all 3 panels
PANEL_FIGSIZE = (5.0, 4.6)        # per-panel size; total figure is 3x this wide
FONTSIZE_TITLE = 16
FONTSIZE_LABEL = 14
FONTSIZE_TICK = 11
XY_LABELPAD = 12                  # theta_1/theta_2 labels: pull away from tick numbers
Z_LABELPAD = 6                    # z-label: matplotlib's 3D offset runs toward the
                                   # title, not sideways, so keep this small -- the
                                   # colorbar collision is fixed via margin/placement
                                   # below, not by padding the z-label further out


def make_grid(r_max=R_MAX, n_grid=N_GRID):
    xd_vals = np.linspace(-r_max, r_max, n_grid)
    yd_vals = np.linspace(-r_max, r_max, n_grid)
    XD, YD = np.meshgrid(xd_vals, yd_vals)
    return XD, YD


def true_value_surface(XD, YD, l1=L1, l2=L2):
    """Closed-form ground truth, vectorized over the grid (no solver calls)."""
    r = np.sqrt(XD**2 + YD**2)
    r_max, r_min = l1 + l2, abs(l1 - l2)
    return np.maximum(r - r_max, 0.0)**2 + np.maximum(r_min - r, 0.0)**2


def learned_value_surface(XD, YD, config, fold=FOLD):
    """NN-predicted value function over the grid for a given config
    ("ik_lass1" or "ik_lass2"), or None if the checkpoint isn't available."""
    models_dir = PROJECT_ROOT / "models" / config
    # ckpt_pattern from problems/registry.py: dnn_{key}_n{n_train}_fold{fold}.pt
    candidates = sorted(models_dir.glob(f"dnn_{config}_n*_fold{fold}.pt")) if models_dir.exists() else []
    if not candidates:
        print(f"  [skip] no checkpoint found in {models_dir} "
              f"(dnn_{config}_n*_fold{fold}.pt) -- pull it from SAVIO first.", flush=True)
        return None
    ckpt_path = candidates[0]
    model, scalers, meta = load_checkpoint(ckpt_path)
    coords = np.stack([XD.ravel(), YD.ravel()], axis=1)
    preds = predict_denorm(model, coords, scalers)
    print(f"  loaded {ckpt_path.relative_to(PROJECT_ROOT)}", flush=True)
    return preds.reshape(XD.shape)


def plot_combined(panels, elev=ELEV, azim=AZIM, cmap=CMAP,
                  panel_figsize=PANEL_FIGSIZE):
    """panels: list of (title, zlabel, V or None) triples, all on the same
    (XD, YD) grid. Shares z-limits/view angle across all panels (no colorbar --
    the z-axis on each panel already carries the scale, and it's the same
    scale on all three); only the first (leftmost) panel gets a z-axis label,
    since it's shared across all three."""
    finite_vals = np.concatenate([V.ravel() for _, _, V in panels if V is not None])
    vmin, vmax = np.nanmin(finite_vals), np.nanmax(finite_vals)

    n = len(panels)
    fig = plt.figure(figsize=(panel_figsize[0] * n, panel_figsize[1]))
    for i, (title, zlabel, V) in enumerate(panels):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        if V is None:
            ax.text2D(0.5, 0.5, "checkpoint\nnot found", ha="center", va="center",
                      transform=ax.transAxes, fontsize=11, color="0.4")
            ax.set_title(title)
            ax.set_axis_off()
            continue
        ax.plot_surface(XD_GLOBAL, YD_GLOBAL, V, cmap=cmap,
                        vmin=vmin, vmax=vmax, rstride=2, cstride=2,
                        linewidth=0, antialiased=True)
        ax.set_xlabel(r"$\theta_1$", fontsize=FONTSIZE_LABEL, labelpad=XY_LABELPAD)
        ax.set_ylabel(r"$\theta_2$", fontsize=FONTSIZE_LABEL, labelpad=XY_LABELPAD)
        if i == 0:
            ax.set_zlabel(zlabel, fontsize=FONTSIZE_LABEL, labelpad=Z_LABELPAD)
        ax.tick_params(axis="both", which="major", labelsize=FONTSIZE_TICK)
        ax.set_zlim(vmin, vmax)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=FONTSIZE_TITLE, pad=12)

    fig.subplots_adjust(wspace=0.02)
    return fig


def main():
    global XD_GLOBAL, YD_GLOBAL
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    XD_GLOBAL, YD_GLOBAL = make_grid()

    V_true = true_value_surface(XD_GLOBAL, YD_GLOBAL)
    V_lass1 = learned_value_surface(XD_GLOBAL, YD_GLOBAL, "ik_lass1")
    V_lass2 = learned_value_surface(XD_GLOBAL, YD_GLOBAL, "ik_lass2")

    panels = [
        ("true value function", r"$v(\theta_1,\theta_2)$", V_true),
        ("learned (Lasserre-1)", r"$\hat v(\theta_1,\theta_2)$", V_lass1),
        ("learned (Lasserre-2)", r"$\hat v(\theta_1,\theta_2)$", V_lass2),
    ]
    fig = plot_combined(panels)
    out = FIGURES_DIR / "ik_value_function_comparison.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.3)
    print(f"  wrote {out}", flush=True)
    plt.show()


if __name__ == "__main__":
    main()
