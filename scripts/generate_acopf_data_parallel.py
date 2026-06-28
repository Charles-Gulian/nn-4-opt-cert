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
- socp and chordal_sdp both solve with CLARABEL (see problem.py _default_solver).
  CLARABEL is single-threaded by default, so n_workers processes ~= n_workers
  threads, matching the allocated CPUs without oversubscription.
- Per-case CLARABEL options (e.g. a wall-clock time_limit on the largest cases)
  are configured in _CLARABEL_OPTS_BY_CASE below and passed straight through to
  prob.solve() as keyword arguments.
- Any solver that raises (not just returns a non-optimal status) is caught
  per-sample in _label_one and recorded as NaN, so a single bad solve can never
  bring down a multi-day run.
"""

import os

# Pin each process to a single thread BEFORE numpy/cvxpy import.  We run one
# worker per allocated core, so every layer that would otherwise grab all cores
# per process must be capped to 1, or 56 workers x 56 threads = ~3000 threads
# thrash 56 cores and the large cases slow by ~50x.  Two layers matter:
#   - numpy/scipy canonicalization -> OpenBLAS/MKL (OMP_/OPENBLAS_/MKL_/...).
#   - the CLARABEL solve itself is Rust and uses a rayon thread pool, which the
#     BLAS vars do NOT control -> RAYON_NUM_THREADS.
# setdefault lets an explicit env export still win.  Under 'spawn' each worker
# re-imports this module, so this runs (before that worker's numpy import) in
# every process.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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

# Per-case CLARABEL options, passed directly as prob.solve() keyword arguments
# (CLARABEL is the default solver for both socp and chordal_sdp).  Defaults are
# fine for small/medium cases; the largest cases get a wall-clock time_limit as
# a safety net so a single pathological sample can't stall a worker for hours.
# NB: these are CLARABEL kwargs, NOT mosek_params — the previous MOSEK-param
# path had no effect under CLARABEL (and passing an unknown kwarg risks erroring
# on every solve in the real worker path).
_CLARABEL_OPTS_BASE = {}
_CLARABEL_OPTS_BY_CASE = {
    "case1354pegase": {"time_limit": 600.0},
    "case2869pegase": {"time_limit": 600.0},
}

def _solver_opts(case_name, relaxation):
    opts = dict(_CLARABEL_OPTS_BASE)
    opts.update(_CLARABEL_OPTS_BY_CASE.get(case_name, {}))
    return opts

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
        "solver_opts": _solver_opts(case_name, relaxation),
    }
    # Warm the CVXPY problem cache at nominal demand.
    p_nom = np.hstack([nd.pd_nominal, nd.qd_nominal])
    solve_relaxation(p_nom, args=_W_args)


def _label_one(task):
    """Label a single demand vector.  Returns (idx, cost, exact, local_cost).

    Every solve is wrapped in try/except: a solver that *raises* (rather than
    returning a non-optimal status) would otherwise propagate out of the worker
    and kill the entire mp.Pool, taking the whole multi-day job down.  On any
    exception we record NaN and continue — NaN rows are written normally and can
    be dropped/re-run later.
    """
    idx, p, include_local = task
    try:
        val, res = solve_relaxation(p, args=_W_args)
        exact = bool(res["exact"])
    except Exception:
        val, exact = np.nan, False
    local_val = np.nan
    if include_local:
        try:
            local_val, _ = solve_local(p, args=_W_args)
        except Exception:
            local_val = np.nan
    return idx, float(val), exact, float(local_val)


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
    """Number of data rows currently in csv_path (0 if absent).

    Line-based (not pd.read_csv) so it never raises on a malformed file — used
    only for progress/summary display.  Resume decisions use
    _validate_and_repair, which also heals corruption.
    """
    if not csv_path.exists():
        return 0
    with open(csv_path) as fh:
        n_lines = sum(1 for _ in fh)
    return max(0, n_lines - 1)   # minus the header row


def _validate_and_repair(csv_path):
    """Return the number of trustworthy completed rows, repairing the file.

    Results are written in sample order (Pool.imap is ordered), so the longest
    well-formed *leading* run of data rows corresponds to samples 0..count-1.
    A line that repeats the header (from a prior bad append), has the wrong
    field count, or is truncated marks where the data stops being trustworthy;
    everything from there on is dropped and the file is rewritten to exactly
    `header + valid rows`.

    This makes re-running a job idempotent and safe:
      - a clean, complete file  -> count == n_total, the job is skipped;
      - a timed-out partial file -> resumes from the validated count;
      - an append-corrupted file (duplicate header / doubled batch) -> self-heals
        to its valid leading rows instead of compounding the corruption.
    """
    if not csv_path.exists():
        return 0
    with open(csv_path) as fh:
        lines = fh.read().splitlines()
    if not lines:
        return 0
    header = lines[0]
    ncols = header.count(",") + 1
    valid = []
    for ln in lines[1:]:
        if ln == header or ln.count(",") + 1 != ncols:
            break                      # embedded header / wrong width / truncated
        valid.append(ln)
    dropped = (len(lines) - 1) - len(valid)
    if dropped:                        # only rewrite when we actually changed something
        with open(csv_path, "w") as fh:
            fh.write(header + "\n")
            if valid:
                fh.write("\n".join(valid) + "\n")
        print(f"  {csv_path.name}: repaired — kept {len(valid)} valid leading "
              f"row(s), dropped {dropped} suspect line(s).", flush=True)
    return len(valid)


def _label_parallel_checkpointed(
    X, feat_cols, include_local,
    n_workers, case_name, relaxation,
    csv_path, checkpoint_every, desc,
    v_min=None, v_max=None,
):
    """Label X in checkpoint batches, appending each batch to csv_path.

    On startup, validates/repairs csv_path and skips the rows already completed,
    so re-running after a timeout (or accidental re-submission) resumes cleanly
    from where it left off without duplicating or corrupting data.
    """
    n_total    = len(X)
    n_done     = _validate_and_repair(csv_path)
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
        # Ordered imap (not imap_unordered): results come back in task order, so
        # row k of the CSV is sample k.  This is what makes count-based resume
        # correct — otherwise the "first n_done rows" would be a random subset of
        # indices and resuming by range(n_done, n_total) would duplicate some
        # samples and skip others.  Workers still run fully in parallel; only the
        # delivery order is fixed, which costs nothing for our uniform solves.
        result_iter = pool.imap(_label_one, tasks, chunksize=chunksize)

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
