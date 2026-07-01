"""Assemble the master AC-OPF result CSVs from per-config part files.

evaluate_acopf.py writes one small part-file per (relaxation, case) under
results/acopf/parts/ (race-free for parallel SLURM jobs).  Run this once after
all evaluation jobs finish to concatenate them into the three master CSVs:
  eval_summary.csv, fold_metrics.csv, confusion_sweep.csv.

Usage:
    python scripts/merge_acopf_results.py
"""

import glob
import pathlib

import pandas as pd

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results" / "acopf"
PARTS_DIR = RESULTS_DIR / "parts"

# part-file prefix -> master filename
_GROUPS = [
    ("eval",      "eval_summary.csv"),
    ("fold",      "fold_metrics.csv"),
    ("confusion", "confusion_sweep.csv"),
]


def main():
    if not PARTS_DIR.exists():
        print(f"No parts directory at {PARTS_DIR} — nothing to merge.")
        return

    for prefix, out_name in _GROUPS:
        files = sorted(glob.glob(str(PARTS_DIR / f"{prefix}__*.csv")))
        if not files:
            print(f"  {out_name}: no parts found (skipped)")
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        # Stable ordering for readability.
        sort_cols = [c for c in ("case", "relaxation", "mode", "threshold", "fold")
                     if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        out_path = RESULTS_DIR / out_name
        df.to_csv(out_path, index=False)
        print(f"  {out_name}: {len(df)} rows from {len(files)} parts -> {out_path}")


if __name__ == "__main__":
    main()
