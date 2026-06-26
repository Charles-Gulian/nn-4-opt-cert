"""Parallel AC-OPF data generation for HPC clusters (e.g. SAVIO / SLURM).

Each worker process builds its own CVXPY problem and network once at startup,
then labels a shard of the X matrix independently.  The main process assembles
results and writes the CSV in the same format used by run_acopf_experiments.py.

Checkpointing
-------------
Results are flushed to CSV in batches (--checkpoint-every rows).  On restart,
already-completed rows are detected from the existing CSV and skipped, so a
timed-out job loses at most one batch of work.

Typical SLURM usage
-------------------
    #!/bin/bash
    #SBATCH --job-name=acopf_data
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=56
    #SBATCH --time=08:00:00
    #SBATCH --partition=savio4_htc

    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate nn4opt

    python scripts/generate_acopf_data_parallel.py \\
        --case case39 --relaxation sdp \\
        --n-train 10000 --n-test 5000 \\
        --n-workers 56

Solver notes
------------
- MSK_IPAR_NUM_THREADS=1 per worker: total threads = n_workers, matching
  allocated CPUs.  Letting MOSEK use multiple threads per solve while also
  running multiple processes leads to oversubscription and slower throughput.
- MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1e-6: slightly looser than the default 1e-8.
  Still far tighter than the ~0.1% cost gaps we care about, but cuts
  interior-point iterations noticeably on larger problems like case39.
- SCS fallback is kept for infeasible / poorly-conditioned instances.
"""

import argparse
import multiprocessing as mp
import pathlib
import sys
import time

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

CHECKPOINT_EVERY = 500   # rows per flush; override with --checkpoint-every

_MOSEK_PARAMS_BASE = {
    "MSK_IPAR_NUM_THREADS":            1,
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-6,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS":   1e-6,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS":   1e-6,
}

# Per-case time limits (seconds). Only applied for SDP; SOCP is fast regardless.
_SDP_TIME_LIMITS = {
    "case300": 300.0,
}

def _mosek_params(case_name, relaxation):
    params = dict(_MOSEK_PARAMS_BASE)
    # Apply per-case time limits to SDP-based relaxations only (SOCP is fast regardless).
    if relaxation in ("sdp", "chordal_sdp") and case_name in _SDP_TIME_LIMITS:
        params["MSK_DPAR_OPTIMIZER_MAX_TIME"] = _SDP_TIME_LIMITS[case_name]
    return {"mosek_params": params}

# ── per-worker state (initialised once per process) ───────────────────────────
_W_args = None


def _worker_init(case_name: str, relaxation: str, v_min=None, v_max=None):
    """Build network + CVXPY problem once per worker process."""
    global _W_args
    net, nd = load_network(case_name, v_min=v_min, v_max=v_max)
    cache = {}
    _W_args = {
        "nd": nd, "net": net, "case_name": case_name,
        "relaxation": relaxation,
        "prob_cache": cache,
        "solver_opts": _mosek_params(case_name, relaxation),
    }
    # Warm the CVXPY problem cache at nominal demand.
    p_nom = np.hstack([nd.pd_nominal, nd.qd_nominal])
    solve_relaxation(p_nom, args=_W_args)


def _label_one(task):
    """Label a single demand vector.  Returns (idx, cost, exact, local_cost)."""
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
        print(f"    -> {path.name}", flush=True)
    else:
        print(f"  Loading {split} X from {path.name}", flush=True)
        X = np.load(path)
    return X


def _count_completed(csv_path):
    """Return number of rows already written to csv_path (0 if file absent)."""
    if not csv_path.exists():
        return 0
    try:
        return len(pd.read_csv(csv_path))
    except Exception:
        return 0


def _label_parallel_checkpointed(
    X, feat_cols, include_local,
    n_workers, case_name, relaxation,
    csv_path, checkpoint_every, desc,
    v_min=None, v_max=None,
):
    """Label X in checkpoint batches, appending each batch to csv_path.

    On startup, counts how many rows are already in csv_path and skips them,
    so re-running after a timeout resumes from where it left off.
    """
    n_total    = len(X)
    n_done     = _count_completed(csv_path)
    n_remaining = n_total - n_done

    if n_remaining <= 0:
        print(f"  {csv_path.name}: already complete ({n_done} rows). Skipping.",
              flush=True)
        return

    if n_done > 0:
        print(f"  {csv_path.name}: resuming from row {n_done} "
              f"({n_remaining} remaining).", flush=True)

    # Only submit tasks for rows not yet in the CSV.
    pending_idx = list(range(n_done, n_total))
    tasks = [(i, X[i], include_local) for i in pending_idx]

    write_header = (n_done == 0)
    t0 = time.time()

    with mp.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(case_name, relaxation, v_min, v_max),
    ) as pool:
        chunksize = max(1, len(tasks) // (n_workers * 8))
        result_iter = pool.imap_unordered(_label_one, tasks, chunksize=chunksize)

        # Accumulate into a batch; flush every checkpoint_every results.
        batch = []
        with tqdm(total=n_remaining, desc=desc, initial=0) as pbar:
            for idx, cost, exact, local_cost in result_iter:
                batch.append({
                    **{col: X[idx, j] for j, col in enumerate(feat_cols)},
                    "Cost":      cost,
                    "Exact":     exact,
                    "LocalCost": local_cost if include_local else np.nan,
                })
                pbar.update(1)

                if len(batch) >= checkpoint_every:
                    df_batch = pd.DataFrame(batch)
                    if not include_local:
                        df_batch = df_batch.drop(columns=["LocalCost"])
                    df_batch.to_csv(
                        csv_path, mode="a",
                        header=write_header, index=False,
                    )
                    write_header = False
                    batch = []
                    print(f"  [checkpoint] {_count_completed(csv_path)}/{n_total} rows "
                          f"saved  ({time.time()-t0:.0f}s elapsed)", flush=True)

        # Flush any remaining results after the loop ends.
        if batch:
            df_batch = pd.DataFrame(batch)
            if not include_local:
                df_batch = df_batch.drop(columns=["LocalCost"])
            df_batch.to_csv(
                csv_path, mode="a",
                header=write_header, index=False,
            )

    elapsed = time.time() - t0
    n_written = _count_completed(csv_path)
    print(f"\n  {csv_path.name}: {n_written}/{n_total} rows  "
          f"({elapsed:.0f}s,  {elapsed/n_remaining:.2f}s/sample)", flush=True)

    if n_written < n_total:
        print(f"  WARNING: only {n_written}/{n_total} rows written — "
              f"re-run to resume.", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel AC-OPF data generation with checkpointing."
    )
    parser.add_argument("--case",              default="case14")
    parser.add_argument("--v-min",             type=float, default=None,
                        help="Override voltage lower bound [pu] (default: pandapower case value)")
    parser.add_argument("--v-max",             type=float, default=None,
                        help="Override voltage upper bound [pu] (default: pandapower case value)")
    parser.add_argument("--relaxation",        default="socp",
                        choices=["socp", "sdp", "chordal_sdp"])
    parser.add_argument("--n-train",           type=int, default=10_000)
    parser.add_argument("--n-test",            type=int, default=5_000)
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--n-workers",         type=int, default=mp.cpu_count())
    parser.add_argument("--checkpoint-every",  type=int, default=CHECKPOINT_EVERY,
                        help="Flush results to CSV every N completed samples "
                             f"(default: {CHECKPOINT_EVERY})")
    parser.add_argument("--regen",             action="store_true",
                        help="Delete existing CSVs and regenerate from scratch")
    parser.add_argument("--train-only",        action="store_true")
    parser.add_argument("--test-only",         action="store_true")
    args = parser.parse_args()

    case_name  = args.case
    relaxation = args.relaxation
    n_workers  = args.n_workers
    seed_train = args.seed
    seed_test  = args.seed + 1

    print(f"Case             : {case_name}")
    print(f"Relaxation       : {relaxation.upper()}")
    print(f"N train/test     : {args.n_train} / {args.n_test}")
    print(f"Workers          : {n_workers}")
    print(f"Seeds            : train={seed_train}, test={seed_test}")
    print(f"Checkpoint every : {args.checkpoint_every} rows")
    print(flush=True)

    net, nd = load_network(case_name, v_min=args.v_min, v_max=args.v_max)
    feat_cols = _col_names(nd)
    args_base = {
        "nd": nd, "net": net, "case_name": case_name,
        "alpha_min": DEFAULT_ALPHA_MIN, "alpha_max": DEFAULT_ALPHA_MAX,
        "eta_range": DEFAULT_ETA_RANGE,
    }

    # ── training set ─────────────────────────────────────────────────────────
    if not args.test_only:
        train_csv = _data_path(case_name, relaxation, args.n_train, "train")
        if args.regen and train_csv.exists():
            train_csv.unlink()
            print(f"  Deleted {train_csv.name} for regeneration.")

        X_train = _generate_or_load_x(
            case_name, args.n_train, seed_train, "train", args_base, args.regen
        )
        _label_parallel_checkpointed(
            X_train, feat_cols, include_local=False,
            n_workers=n_workers, case_name=case_name, relaxation=relaxation,
            csv_path=train_csv, checkpoint_every=args.checkpoint_every,
            desc=f"Train [{relaxation.upper()}]",
            v_min=args.v_min, v_max=args.v_max,
        )

    # ── test set ──────────────────────────────────────────────────────────────
    if not args.train_only:
        test_csv = _data_path(case_name, relaxation, args.n_test, "test")
        if args.regen and test_csv.exists():
            test_csv.unlink()
            print(f"  Deleted {test_csv.name} for regeneration.")

        X_test = _generate_or_load_x(
            case_name, args.n_test, seed_test, "test", args_base, args.regen
        )
        _label_parallel_checkpointed(
            X_test, feat_cols, include_local=True,
            n_workers=n_workers, case_name=case_name, relaxation=relaxation,
            csv_path=test_csv, checkpoint_every=args.checkpoint_every,
            desc=f"Test  [{relaxation.upper()} + IPOPT]",
            v_min=args.v_min, v_max=args.v_max,
        )

        # Print gap stats if complete.
        n_done = _count_completed(test_csv)
        if n_done == args.n_test:
            df = pd.read_csv(test_csv)
            gap = (df["LocalCost"] - df["Cost"]).dropna()
            print(f"  Gap (LocalCost - {relaxation.upper()}): "
                  f"mean={gap.mean():.2f}  median={gap.median():.2f}  "
                  f"max={gap.max():.2f} $/hr", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
