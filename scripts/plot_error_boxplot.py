"""Box-and-whisker plot of the NN's signed test-set prediction error (g - v),
pooled across all 4 folds, for every certified config.

Absolute signed error is used (not relative %) because relative error blows up
for QCQP and IK, whose relaxation value v is near zero for many "already
reachable / already tight" instances -- dividing by v there produces errors of
thousands of percent that would swamp the plot. Absolute error also spans many
orders of magnitude across configs (AC-OPF costs are ~1e3-1e5, IK costs are
~1e-1), so the y-axis uses a symmetric-log scale to keep both the near-zero
small-problem errors and the large AC-OPF errors legibly visible on one plot.

Reads results/*/fold_test_predictions.csv (written by evaluate_certify.py).

Usage:
    python scripts/plot_error_boxplot.py
"""

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Display order: small problems first, then AC-OPF grouped by relaxation and
# sorted by case size (roughly ascending network size).
SMALL_ORDER = ["qcqp", "mimo", "ik_lass1", "ik_lass2"]
ACOPF_CASE_ORDER = ["case9", "case14", "case39", "case89pegase",
                    "case118", "case300", "case1354pegase", "case2869pegase"]


def _display_order():
    order = list(SMALL_ORDER)
    for relax in ("socp", "chordal_sdp"):
        for case in ACOPF_CASE_ORDER:
            key = f"acopf_{relax}_{case}"
            if (RESULTS_DIR / "acopf-cert" / key / "fold_test_predictions.csv").exists():
                order.append(key)
    return order


def _pred_path(key):
    small = RESULTS_DIR / key / "fold_test_predictions.csv"
    return small if small.exists() else RESULTS_DIR / "acopf-cert" / key / "fold_test_predictions.csv"


def main():
    keys = _display_order()
    data, labels = [], []
    for key in keys:
        path = _pred_path(key)
        if not path.exists():
            print(f"skip {key}: no fold_test_predictions.csv")
            continue
        df = pd.read_csv(path)
        err = (df["g"] - df["v"]).dropna().values
        data.append(err)
        labels.append(key.replace("acopf_", "").replace("_case", "\ncase"))

    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(labels)), 6))
    ax.boxplot(data, tick_labels=labels, showfliers=True,
               flierprops=dict(marker=".", markersize=2, alpha=0.3),
               medianprops=dict(color="crimson"))
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("signed prediction error  g(theta) - v(theta)\n(symlog scale)")
    ax.set_title("NN prediction error distribution by experiment (all 4 folds pooled, full test set)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "error_distribution_boxplot.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
