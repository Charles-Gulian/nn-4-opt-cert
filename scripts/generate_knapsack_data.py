"""Generate labeled data for the robust 0-1 knapsack problem.

Each instance theta = (mu, sigma) is labeled with the certified global-optimal
value of the MISOCP (solved exactly via Gurobi) -- see
problems/robust_knapsack/problem.py. Unlike AC-OPF there is no relaxation/
local split, so there is a single "Cost" column and no "Exact"/"LocalCost".

Solves were assumed ~30-40ms each (a serial loop), but measured throughput on
SAVIO under the academic WLS license was actually ~0.15s/solve (per-solve
license-check network overhead), making 640k rows take >24h serially. Pass
--n-workers > 1 to parallelize across processes (each with its own Gurobi
Env); run scripts/probe_gurobi_concurrency.py first to confirm the license
supports concurrent sessions.

Usage:
    python scripts/generate_knapsack_data.py --n-train 20000 --n-test 5000
    python scripts/generate_knapsack_data.py --n-train 640000 --n-test 20000 --n-workers 8
"""

import argparse
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

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

# Populated once per worker process by _init_worker; used by _solve_one.
_WORKER_ARGS = None


def _init_worker(solver_opts):
    global _WORKER_ARGS
    prob, x, t, mu_param, sigma2_param, sigma_param = _build_misocp()
    _WORKER_ARGS = {"prob": prob, "x": x, "t": t, "mu_param": mu_param,
                     "sigma2_param": sigma2_param, "sigma_param": sigma_param,
                     "solver_opts": solver_opts or {}}


def _is_too_many_sessions(exc):
    """WLS license error 10030 ('Too many sessions ...'). This is transient --
    lingering sessions from prior runs expire server-side after a few minutes --
    so it's worth retrying rather than aborting. Any OTHER error is a real
    failure and must propagate."""
    s = repr(exc)
    return "10030" in s or "Too many sessions" in s


def _solve_one(item, max_retries=6, base_backoff=15.0):
    """Run in a worker process. Returns (idx, val, error_or_None).

    A worker's Gurobi session is created on its first solve and reused
    thereafter, so a 'too many sessions' error effectively only occurs at
    worker startup (a phantom session from a prior run hasn't expired yet). We
    retry that specific error with a long backoff -- WLS sessions expire on the
    order of minutes, so backoff grows 15s, 30s, 60s, ... Non-10030 errors
    return immediately so the parent aborts loudly (the original silent-NaN
    failure mode we are guarding against)."""
    idx, x_row = item
    for attempt in range(max_retries + 1):
        try:
            val, _ = solve_exact(x_row, args=_WORKER_ARGS)
            return idx, val, None
        except Exception as e:
            if _is_too_many_sessions(e) and attempt < max_retries:
                time.sleep(base_backoff * (2 ** attempt))
                continue
            return idx, np.nan, repr(e)


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


def _label_checkpointed(X, feat_cols, csv_path, checkpoint_every, desc, solver_opts=None,
                         n_workers=1):
    n_total = len(X)
    n_done = _count_completed(csv_path)
    n_remaining = n_total - n_done
    if n_remaining <= 0:
        print(f"  {csv_path.name}: already complete ({n_done} rows). Skipping.", flush=True)
        return
    if n_done > 0:
        print(f"  {csv_path.name}: resuming from row {n_done} ({n_remaining} remaining).", flush=True)

    solver_opts = solver_opts or {}

    # Fail fast if the environment can't solve at all (e.g. Gurobi missing/
    # unlicensed on a compute node): probe the first instance in this process
    # (no forked workers yet) and raise the real exception rather than
    # silently labelling every row NaN and wasting the whole run.
    _probe_args = None
    _probe_val, _ = solve_exact(X[n_done], args=_probe_args)
    if not np.isfinite(_probe_val):
        raise RuntimeError(
            f"First solve returned {_probe_val!r} (status not OPTIMAL). "
            "Aborting rather than generating all-NaN labels -- check that Gurobi "
            "is installed and licensed in this environment.")

    write_header = (n_done == 0)
    t0 = time.time()
    n_nan = 0
    n_saved = 0

    if n_workers <= 1:
        prob, x, t, mu_param, sigma2_param, sigma_param = _build_misocp()
        args = {"prob": prob, "x": x, "t": t, "mu_param": mu_param,
                "sigma2_param": sigma2_param, "sigma_param": sigma_param,
                "solver_opts": solver_opts}
        pending = []
        for i in tqdm(range(n_done, n_total), desc=desc, initial=0, total=n_remaining):
            try:
                val, _ = solve_exact(X[i], args=args)
            except Exception:
                val = np.nan
            if not np.isfinite(val):
                n_nan += 1
            pending.append({**{col: X[i, j] for j, col in enumerate(feat_cols)}, "Cost": val})
            if len(pending) >= checkpoint_every:
                pd.DataFrame(pending).to_csv(csv_path, mode="a", header=write_header, index=False)
                write_header = False
                pending = []
                print(f"  [checkpoint] {_count_completed(csv_path)}/{n_total} rows saved "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)
        if pending:
            pd.DataFrame(pending).to_csv(csv_path, mode="a", header=write_header, index=False)
    else:
        # Parallel path: each worker process builds its own Gurobi Env/problem
        # once (via _init_worker) and solves assigned indices independently.
        # ProcessPoolExecutor.map preserves input order, so checkpointing logic
        # below is unchanged from the serial path. If ANY worker raises (e.g. a
        # license-concurrency conflict), abort immediately rather than silently
        # writing NaN for that row -- that silent-failure mode is exactly what
        # produced the original all-NaN dataset.
        print(f"  Parallelizing across {n_workers} workers.", flush=True)
        items = ((i, X[i]) for i in range(n_done, n_total))
        pending = []
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                  initargs=(solver_opts,)) as ex:
            for idx, val, err in tqdm(ex.map(_solve_one, items, chunksize=4),
                                       desc=desc, initial=0, total=n_remaining):
                if err is not None:
                    raise RuntimeError(
                        f"Worker failed on row {idx}: {err}. Aborting rather than "
                        "writing a NaN row for what may be a license-concurrency "
                        "conflict -- rerun scripts/probe_gurobi_concurrency.py to "
                        "check the safe worker count, or rerun with --n-workers 1.")
                if not np.isfinite(val):
                    n_nan += 1
                pending.append({**{col: X[idx, j] for j, col in enumerate(feat_cols)}, "Cost": val})
                if len(pending) >= checkpoint_every:
                    pd.DataFrame(pending).to_csv(csv_path, mode="a", header=write_header, index=False)
                    write_header = False
                    pending = []
                    print(f"  [checkpoint] {_count_completed(csv_path)}/{n_total} rows saved "
                          f"({time.time()-t0:.0f}s elapsed)", flush=True)
        if pending:
            pd.DataFrame(pending).to_csv(csv_path, mode="a", header=write_header, index=False)

    n_written = _count_completed(csv_path)
    elapsed = time.time() - t0
    print(f"  {csv_path.name}: {n_written}/{n_total} rows ({elapsed:.0f}s, "
          f"{elapsed/max(n_remaining,1):.3f}s/sample)", flush=True)
    if n_nan:
        print(f"  WARNING: {n_nan}/{n_remaining} solves returned NaN (non-OPTIMAL "
              "status) -- these rows will be dropped downstream.", flush=True)


def main():
    p = argparse.ArgumentParser(description="Generate robust-knapsack training/test data.")
    p.add_argument("--n-train", type=int, default=20_000)
    p.add_argument("--n-test", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=343)
    p.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    p.add_argument("--regen", action="store_true")
    p.add_argument("--time-limit", type=float, default=None,
                   help="Optional Gurobi TimeLimit (s) per solve, as a safety net.")
    p.add_argument("--n-workers", type=int, default=1,
                   help="Parallel worker processes for Gurobi solves (each with its "
                        "own Env). Confirm the license supports this many concurrent "
                        "sessions first via scripts/probe_gurobi_concurrency.py.")
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
                             solver_opts=solver_opts, n_workers=args.n_workers)


if __name__ == "__main__":
    main()
