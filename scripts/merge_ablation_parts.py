"""Merge the per-task ablation CSVs written by the SAVIO job array into the
single results/acopf/ablation_summary.csv that emit_ablation_tables.py reads.

The array runs tasks concurrently, so each writes its own part file
(results/acopf/ablation_parts/part_<i>.csv); ablation_acopf.py's in-place upsert
is not concurrency safe, hence the split-then-merge.

Rows are keyed by (case, relaxation, depth, width, pretrain_epochs,
finetune_epochs, single_phase) -- the same key ablation_acopf.py upserts on -- so
re-running a task and re-merging replaces that row rather than duplicating it.
Newer part files win on conflict.

Usage:
    python scripts/merge_ablation_parts.py
    python scripts/merge_ablation_parts.py --parts results/acopf/ablation_parts \
        --out results/acopf/ablation_summary.csv
"""

import argparse
import pathlib
import sys

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
KEY = ["case", "relaxation", "depth", "width",
       "pretrain_epochs", "finetune_epochs", "single_phase"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=pathlib.Path,
                    default=PROJECT_ROOT / "results" / "acopf" / "ablation_parts")
    ap.add_argument("--out", type=pathlib.Path,
                    default=PROJECT_ROOT / "results" / "acopf" / "ablation_summary.csv")
    args = ap.parse_args()

    files = sorted(args.parts.glob("part_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        print(f"no part files in {args.parts}", file=sys.stderr)
        return 1

    frames = []
    if args.out.exists():
        frames.append(pd.read_csv(args.out))     # existing rows first (lowest priority)
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"  skip {f.name}: {e}", file=sys.stderr)

    df = pd.concat(frames, ignore_index=True)
    for c in KEY:
        if c not in df.columns:
            df[c] = 0
    before = len(df)
    df = df.drop_duplicates(subset=KEY, keep="last").reset_index(drop=True)
    df.to_csv(args.out, index=False)
    print(f"merged {len(files)} part files: {before} rows -> {len(df)} unique "
          f"-> {args.out}")
    print(df.groupby(["case"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
