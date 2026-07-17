"""Generate labeled data for the robust 0-1 knapsack problem.

Each instance theta = (mu, sigma) is labeled with the certified global-optimal
value of the MISOCP (solved exactly via Gurobi) -- see
problems/robust_knapsack/problem.py. Unlike AC-OPF there is no relaxation/
local split, so there is a single "Cost" column and no "Exact"/"LocalCost".

Solves are fast (~30ms each), so this runs serially with periodic
checkpointing (not a multiprocessing pool like the AC-OPF data-gen).

Usage:
    python scripts/generate_knapsack_data.py --n-train 20000 --n-test 5000
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from problems.robust_knapsack.problem import _build_misocp, solve_exact
from problems.robust_knapsack.generate_data import sample_parameters, _col_names

DATA_DIR = PROJECT_ROOT / "data" / "robust_knapsack"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_EVERY = 500


def _x_path(n, seed, split):
    return DATA_DIR / f"X_{split}_{n}_seed{seed}.npy"


def _data_path(n, split):
    return DATA_DIR / f"{split}_{n}.csv"


def _count_completed(csv_path):
    if not csv_path.exists():
        return 0
    with open(csv_path) as fh:
        n_lines = sum(1 for _ in fh)
    return max(0, n_lines - 1)


def _label_checkpointed(X, feat_cols, csv_path, checkpoint_every, desc, solver_opts=None):
    n_total = len(X)
    n_done = _count_completed(csv_path)
    n_remaining = n_total - n_done
    if n_remaining <= 0:
        print(f"  {csv_path.name}: already complete ({n_done} rows). Skipping.", flush=True)
        return
    if n_done > 0:
        print(f"  {csv_path.name}: resuming from row {n_done} ({n_remaining} remaining).", flush=True)

    prob, x, t, mu_param, sigma2_param, sigma_param = _build_misocp()
    args = {"prob": prob, "x": x, "t": t, "mu_param": mu_param,
            "sigma2_param": sigma2_param, "sigma_param": sigma_param,
            "solver_opts": solver_opts or {}}

    write_header = (n_done == 0)
    batch = []
    t0 = time.time()
    for i in tqdm(range(n_done, n_total), desc=desc, initial=0, total=n_remaining):
        try:
            val, _ = solve_exact(X[i], args=args)
        except Exception:
            val = np.nan
        batch.append({**{col: X[i, j] for j, col in enumerate(feat_cols)}, "Cost": val})

        if len(batch) >= checkpoint_every:
            pd.DataFrame(batch).to_csv(csv_path, mode="a", header=write_header, index=False)
            write_header = False
            batch = []
            print(f"  [checkpoint] {_count_completed(csv_path)}/{n_total} rows saved "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)

    if batch:
        pd.DataFrame(batch).to_csv(csv_path, mode="a", header=write_header, index=False)

    n_written = _count_completed(csv_path)
    elapsed = time.time() - t0
    print(f"  {csv_path.name}: {n_written}/{n_total} rows ({elapsed:.0f}s, "
          f"{elapsed/max(n_remaining,1):.3f}s/sample)", flush=True)


def main():
    p = argparse.ArgumentParser(description="Generate robust-knapsack training/test data.")
    p.add_argument("--n-train", type=int, default=20_000)
    p.add_argument("--n-test", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=343)
    p.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    p.add_argument("--regen", action="store_true")
    p.add_argument("--time-limit", type=float, default=None,
                   help="Optional Gurobi TimeLimit (s) per solve, as a safety net.")
    args = p.parse_args()

    seed_train, seed_test = args.seed, args.seed + 1
    feat_cols = _col_names()
    solver_opts = {"TimeLimit": args.time_limit} if args.time_limit else {}

    for n, seed, split in [(args.n_train, seed_train, "train"),
                            (args.n_test, seed_test, "test")]:
        x_path = _x_path(n, seed, split)
        if not x_path.exists() or args.regen:
            print(f"  Sampling {n} {split} instances (seed={seed}) ...", flush=True)
            X = sample_parameters(n, args={"seed": seed})
            np.save(x_path, X)
        else:
            print(f"  Loading {split} X from {x_path.name}", flush=True)
            X = np.load(x_path)

        csv_path = _data_path(n, split)
        if args.regen and csv_path.exists():
            csv_path.unlink()
            print(f"  Deleted {csv_path.name} for regeneration.", flush=True)

        _label_checkpointed(X, feat_cols, csv_path, args.checkpoint_every,
                             desc=f"{split.capitalize()} [robust knapsack]",
                             solver_opts=solver_opts)


if __name__ == "__main__":
    main()
