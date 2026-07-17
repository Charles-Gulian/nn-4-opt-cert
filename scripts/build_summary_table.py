"""Assemble the paper-facing summary tables from every config's per-fold result
CSVs (written by scripts/evaluate_certify.py, optionally scripts/generate_dataset.py).

Merges, per (config, fold, level): certification confusion/rates, test-error
summary (MAPE/percentiles), the conformal offset, and (where available -- AC-OPF
skips data generation, so it has none) mean relaxation/local solve times.

Writes two CSVs under results/:
  summary_per_fold.csv   one row per (config, fold, level) -- the reporting
                          unit specified by the workflow (per-fold-model).
  summary_by_config.csv  one row per (config, level), mean +/- std across the
                          4 folds -- for a compact paper table / quick scan.

Usage:
    python scripts/build_summary_table.py
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"

# Every config directory that has a certification.csv is a completed config --
# discover them directly rather than hardcoding the registry's key list, so
# this table-builder doesn't need updating if new configs are added.
CONFIG_DIRS = (
    sorted(RESULTS_DIR.glob("*/certification.csv"))
    + sorted(RESULTS_DIR.glob("acopf-cert/*/certification.csv"))
)


def _config_name(cert_csv_path):
    return cert_csv_path.parent.name


def load_one(cert_csv_path):
    cfg = _config_name(cert_csv_path)
    d = cert_csv_path.parent

    cert = pd.read_csv(cert_csv_path)
    metrics = pd.read_csv(d / "fold_test_metrics.csv")
    offsets = pd.read_csv(d / "conformal_offsets.csv")

    df = cert.merge(offsets, on=["fold", "level"], how="left")
    df = df.merge(metrics, on="fold", how="left", suffixes=("", "_test"))
    df.insert(0, "config", cfg)

    timing_path = d / "solve_times.json"
    if timing_path.exists():
        timing = json.loads(timing_path.read_text())
        df["mean_relax_solve_s"] = timing.get("mean_relax_solve_s", np.nan)
        df["mean_local_solve_s"] = timing.get("mean_local_solve_s", np.nan)
    else:
        # AC-OPF: data (and its solve-time record) predates this workflow and
        # was not regenerated, per the plan -- leave NaN rather than guessing.
        df["mean_relax_solve_s"] = np.nan
        df["mean_local_solve_s"] = np.nan

    return df


def main():
    if not CONFIG_DIRS:
        print("No certification.csv files found under results/ -- nothing to summarize.")
        return

    per_fold = pd.concat([load_one(p) for p in CONFIG_DIRS], ignore_index=True)

    cols = ["config", "fold", "level", "offset",
            "tp", "fp", "tn", "fn", "tpr", "fpr", "tnr", "fnr",
            "mape", "p1", "p5", "p10", "p90", "p95", "p99", "min_error", "max_error",
            "mean_relax_solve_s", "mean_local_solve_s"]
    per_fold = per_fold[cols]

    per_fold_path = RESULTS_DIR / "summary_per_fold.csv"
    per_fold.to_csv(per_fold_path, index=False)
    print(f"wrote {per_fold_path} ({len(per_fold)} rows, {per_fold['config'].nunique()} configs)")

    agg_cols = ["offset", "tpr", "fpr", "tnr", "fnr", "mape",
                "p1", "p5", "p10", "p90", "p95", "p99", "min_error", "max_error"]
    by_config = (per_fold.groupby(["config", "level"])[agg_cols]
                 .agg(["mean", "std"]))
    by_config.columns = [f"{c}_{stat}" for c, stat in by_config.columns]
    by_config = by_config.reset_index()

    # Solve times don't vary by fold -- carry them through as a single column.
    times = per_fold.groupby("config")[["mean_relax_solve_s", "mean_local_solve_s"]].first()
    by_config = by_config.merge(times, on="config", how="left")

    by_config_path = RESULTS_DIR / "summary_by_config.csv"
    by_config.to_csv(by_config_path, index=False)
    print(f"wrote {by_config_path} ({len(by_config)} rows)")


if __name__ == "__main__":
    main()
