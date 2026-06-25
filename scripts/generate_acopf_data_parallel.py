"""Parallel AC-OPF data generation for HPC clusters (e.g. SAVIO / SLURM).

Each worker process builds its own CVXPY problem and network once at startup,
then labels a shard of the X matrix independently.  The main process assembles
results and writes the CSV in the same format used by run_acopf_experiments.py.

Typical SLURM usage
-------------------
    #!/bin/bash
    #SBATCH --job-name=acopf_data
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=32
    #SBATCH --time=04:00:00
    #SBATCH --partition=savio3

    module load python
    conda activate nn4opt

    python scripts/generate_acopf_data_parallel.py \\
        --case case39 --relaxation sdp \\
        --n-train 10000 --n-test 5000 \\
        --n-workers 32

Solver notes
------------
- MOSEK_THREADS=1 per worker: total threads = n_workers, matching allocated CPUs.
  Letting MOSEK use multiple threads per solve while also running multiple
  processes leads to oversubscription and is slower overall.
- MOSEK tolerance: MSK_DPAR_INTPNT_CO_TOL_REL_GAP is relaxed from 1e-8 to 1e-6.
  This is still tight enough for our purposes (cost gap < 0.01%) and cuts
  iteration count noticeably on larger problems like case39.
- SCS fallback is kept for infeasible / poorly-conditioned instances.
"""

import argparse
import multiprocessing as mp
import pathlib
import sys
import time
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from problems.acopf.network import load_network
from problems.acopf.problem import solve_relaxation, solve_local
from problems.acopf.generate_data import (
    sample_parameters, _col_names,
    DEFAULT_ALPHA_MIN, DEFAULT_ALPHA_MAX, DEFAULT_ETA_RANGE,
)

DATA_DIR = PROJECT_ROOT / "data" / "acopf"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# MOSEK options passed via solver_opts in args.
# MSK_IPAR_NUM_THREADS = 1  → one thread per worker, no oversubscription.
# MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1e-6 → slightly looser than default 1e-8;
#   still far tighter than the ~0.1% cost errors we care about.
_MOSEK_PARAMS = {
    "mosek_params": {
        "MSK_IPAR_NUM_THREADS":          1,
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-6,
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS":   1e-6,
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS":   1e-6,
    }
}

# ── per-worker state ──────────────────────────────────────────────────────────
# Each worker process initialises these once via _worker_init.
_W_nd    = None
_W_net   = None
_W_cache = None
_W_args  = None


def _worker_init(case_name: str, relaxation: str):
    """Initialise worker: build network + pre-solve once to warm CVXPY cache."""
    global _W_nd, _W_net, _W_cache, _W_args
    net, nd = load_network(case_name)
    _W_nd    = nd
    _W_net   = net
    _W_cache = {}
    _W_args  = {
        "nd": nd, "net": net, "case_name": case_name,
        "relaxation": relaxation,
        "prob_cache": _W_cache,
        "solver_opts": _MOSEK_PARAMS,
    }
    # Pre-solve at nominal demand so CVXPY builds and caches the problem object.
    p_nom = np.hstack([nd.pd_nominal, nd.qd_nominal])
    solve_relaxation(p_nom, args=_W_args)


def _label_one(task):
    """Label a single demand vector.

    Parameters
    ----------
    task : (idx, p, include_local)

    Returns
    -------
    (idx, cost, exact, local_cost)   — local_cost is nan when include_local=False
    """
    idx, p, include_local = task
    val, res = solve_relaxation(p, args=_W_args)

    local_val = np.nan
    if include_local:
        local_val, _ = solve_local(p, args=_W_args)

    return idx, float(val), bool(res["exact"]), float(local_val)


# ── helpers ───────────────────────────────────────────────────────────────────

def _x_path(case_name, n, seed, split):
    return DATA_DIR / f"X_{split}_{n}_{case_name}_seed{seed}.npy"


def _data_path(case_name, relaxation, n, split):
    return DATA_DIR / f"{split}_{n}_{relaxation}_{case_name}.csv"


def _generate_or_load_x(case_name, n, seed, split, args_base, force):
    path = _x_path(case_name, n, seed, split)
    if not path.exists() or force:
        print(f"  Sampling {n} {split} X points (seed={seed}) ...", flush=True)
        X = sample_parameters(n, args=dict(args_base, seed=seed))
        np.save(path, X)
        print(f"    -> {path.name}")
    else:
        print(f"  Loading {split} X from {path.name}")
        X = np.load(path)
    return X


def _label_parallel(X, include_local, n_workers, case_name, relaxation, desc):
    tasks = [(i, X[i], include_local) for i in range(len(X))]

    results = [None] * len(X)
    with mp.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(case_name, relaxation),
    ) as pool:
        for idx, cost, exact, local_cost in tqdm(
            pool.imap_unordered(_label_one, tasks, chunksize=max(1, len(X) // (n_workers * 8))),
            total=len(X),
            desc=desc,
        ):
            results[idx] = (cost, exact, local_cost)

    costs      = [r[0] for r in results]
    exacts     = [r[1] for r in results]
    local_costs = [r[2] for r in results]
    return costs, exacts, local_costs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel AC-OPF data generation (SOCP or SDP)."
    )
    parser.add_argument("--case",        default="case14",
                        choices=["case9", "case14", "case39"])
    parser.add_argument("--relaxation",  default="socp", choices=["socp", "sdp"])
    parser.add_argument("--n-train",     type=int, default=10_000)
    parser.add_argument("--n-test",      type=int, default=5_000)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--n-workers",   type=int, default=mp.cpu_count(),
                        help="Number of parallel worker processes "
                             "(default: all CPUs on this node)")
    parser.add_argument("--regen",       action="store_true",
                        help="Force regeneration even if output files exist")
    parser.add_argument("--train-only",  action="store_true",
                        help="Skip test set generation (no local solver)")
    parser.add_argument("--test-only",   action="store_true",
                        help="Skip training set generation")
    args = parser.parse_args()

    case_name  = args.case
    relaxation = args.relaxation
    n_workers  = args.n_workers
    seed_train = args.seed
    seed_test  = args.seed + 1

    print(f"Case        : {case_name}")
    print(f"Relaxation  : {relaxation.upper()}")
    print(f"N train/test: {args.n_train} / {args.n_test}")
    print(f"Workers     : {n_workers}")
    print(f"Seeds       : train={seed_train}, test={seed_test}")
    print()

    net, nd = load_network(case_name)
    feat_cols = _col_names(nd)
    args_base = {
        "nd": nd, "net": net, "case_name": case_name,
        "alpha_min": DEFAULT_ALPHA_MIN, "alpha_max": DEFAULT_ALPHA_MAX,
        "eta_range": DEFAULT_ETA_RANGE,
    }

    # ── training set ─────────────────────────────────────────────────────────
    if not args.test_only:
        train_csv = _data_path(case_name, relaxation, args.n_train, "train")

        if train_csv.exists() and not args.regen:
            print(f"Train CSV already exists: {train_csv.name}  (use --regen to overwrite)")
        else:
            X_train = _generate_or_load_x(
                case_name, args.n_train, seed_train, "train", args_base, args.regen
            )
            t0 = time.time()
            costs, exacts, _ = _label_parallel(
                X_train, include_local=False,
                n_workers=n_workers, case_name=case_name, relaxation=relaxation,
                desc=f"Train [{relaxation.upper()}]",
            )
            elapsed = time.time() - t0

            df = pd.DataFrame(X_train, columns=feat_cols)
            df["Cost"]  = costs
            df["Exact"] = exacts
            df.to_csv(train_csv, index=False)

            n_exact = sum(exacts)
            print(f"\n  Saved {train_csv.name}  ({elapsed:.0f}s,  "
                  f"{elapsed/args.n_train:.2f}s/sample,  "
                  f"exact={n_exact}/{args.n_train}={100*n_exact/args.n_train:.1f}%)")

    # ── test set ──────────────────────────────────────────────────────────────
    if not args.train_only:
        test_csv = _data_path(case_name, relaxation, args.n_test, "test")

        if test_csv.exists() and not args.regen:
            print(f"Test CSV already exists: {test_csv.name}  (use --regen to overwrite)")
        else:
            X_test = _generate_or_load_x(
                case_name, args.n_test, seed_test, "test", args_base, args.regen
            )
            t0 = time.time()
            costs, exacts, local_costs = _label_parallel(
                X_test, include_local=True,
                n_workers=n_workers, case_name=case_name, relaxation=relaxation,
                desc=f"Test  [{relaxation.upper()} + IPOPT]",
            )
            elapsed = time.time() - t0

            df = pd.DataFrame(X_test, columns=feat_cols)
            df["Cost"]       = costs
            df["Exact"]      = exacts
            df["LocalCost"]  = local_costs
            df.to_csv(test_csv, index=False)

            gap = np.array(local_costs) - np.array(costs)
            gap = gap[np.isfinite(gap)]
            print(f"\n  Saved {test_csv.name}  ({elapsed:.0f}s,  "
                  f"{elapsed/args.n_test:.2f}s/sample)")
            if len(gap):
                print(f"  Gap (LocalCost - {relaxation.upper()}): "
                      f"mean={gap.mean():.2f}  median={np.median(gap):.2f}  "
                      f"max={gap.max():.2f} $/hr")


if __name__ == "__main__":
    # Required on macOS / some Linux configs to avoid fork-related deadlocks with
    # MOSEK's internal thread pool.  "spawn" starts fresh interpreter per worker.
    mp.set_start_method("spawn", force=True)
    main()
