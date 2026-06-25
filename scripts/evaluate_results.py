"""Stage 3: compute error statistics from a predictions CSV.

Generic across problems -- expects a CSV with at least `Cost` (true value),
`Pred` (predicted value), and `fold` columns (as produced by
`scripts/train_qcqp.py`).

Usage:
    python scripts/evaluate_results.py results/qcqp_example/cv_predictions_QCQP_example_2d_5000samples.csv
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nn.metrics import error_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_csv", type=pathlib.Path)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv)

    for fold, group in df.groupby("fold"):
        stats = error_summary(group["Cost"], group["Pred"])
        print(
            f"Fold {fold}: "
            f"mean abs error = {stats['mean_abs_error']:.4f} "
            f"(95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]), "
            f"max abs error = {stats['max_abs_error']:.4f} "
            f"(n={stats['n']})"
        )

    overall = error_summary(df["Cost"], df["Pred"])
    print(
        f"Overall: "
        f"mean abs error = {overall['mean_abs_error']:.4f} "
        f"(95% CI: [{overall['ci_lower']:.4f}, {overall['ci_upper']:.4f}]), "
        f"max abs error = {overall['max_abs_error']:.4f} "
        f"(n={overall['n']})"
    )
