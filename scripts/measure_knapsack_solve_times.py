"""Time the correlated-mu robust knapsack's two offline solves: the exact
MISOCP (Gurobi) and its continuous SOCP relaxation (CLARABEL), completing the
knapsack row in the timing table (previously '--' for Relaxation Solve, since
no relaxation timing had been measured).

Reuses the exact problem construction from generate_knapsack_corr_data.py
(same Sigma, budget, values) and the shared-mu test pool, so instances are
drawn from the real evaluation distribution. Reports solver-internal time
(cvxpy prob.solver_stats.solve_time) for both, same convention as
measure_solve_times.py's AC-OPF measurements.

Usage:
    /opt/anaconda3/envs/nn4opt/bin/python scripts/measure_knapsack_solve_times.py --n 50
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import cvxpy as cp

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_knapsack_corr_data import N, BUDGET, RHO_ROB, _C, _L

RESULTS = PROJECT_ROOT / "results"
DATA = PROJECT_ROOT / "data" / "knapsack_corr"


def build_problem(relax):
    x = cp.Variable(N, boolean=not relax)
    t = cp.Variable(nonneg=True)
    mu_p = cp.Parameter(N, nonneg=True)
    cons = [mu_p @ x + RHO_ROB * t <= BUDGET, cp.SOC(t, _L.T @ x)]
    if relax:
        cons += [x >= 0, x <= 1]
    return cp.Problem(cp.Maximize(_C @ x), cons), mu_p


def time_solver(mu_samples, relax, solver, n):
    prob, mu_p = build_problem(relax)
    # warm up (first-call construction not charged to a timed sample)
    mu_p.value = mu_samples[0]
    prob.solve(solver=solver, verbose=False)

    solver_ts, total_ts, n_failed = [], [], 0
    for _ in range(n):
        mu = mu_samples[np.random.randint(len(mu_samples))]
        mu_p.value = mu
        t0 = time.perf_counter()
        try:
            prob.solve(solver=solver, verbose=False)
        except Exception:
            n_failed += 1
            continue
        total_ts.append(time.perf_counter() - t0)
        try:
            solver_ts.append(prob.solver_stats.solve_time)
        except Exception:
            pass
    return (float(np.mean(solver_ts)) if solver_ts else float("nan"),
            float(np.mean(total_ts)) if total_ts else float("nan"), n_failed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--only", choices=["exact", "relax"], default=None,
                    help="time only this solver (run each in its own process --"
                         "timing both Gurobi and CLARABEL in one process was "
                         "observed to corrupt CLARABEL's solver_stats.solve_time, "
                         "reporting values inconsistent with, and larger than, "
                         "wall-clock time around the same call -- isolated runs "
                         "don't show this).")
    args = ap.parse_args()

    mu_samples = np.load(DATA / "test_5000.npz")["mu"]
    results = {}
    if args.only in (None, "exact"):
        results["exact"] = time_solver(mu_samples, relax=False, solver=cp.GUROBI, n=args.n)
    if args.only in (None, "relax"):
        results["relax"] = time_solver(mu_samples, relax=True, solver=cp.CLARABEL, n=args.n)

    reported = {}
    for label, (solve_t, total_t, failed) in results.items():
        print(f"{label}: solve={solve_t:.6f}s  total={total_t:.6f}s  ({failed}/{args.n} failed)")
        if solve_t <= total_t * 1.05:
            reported[label] = solve_t
        else:
            # Observed specifically for CLARABEL when this module is imported
            # alongside generate_knapsack_corr_data (whose Gurobi-related setup
            # at import time appears to disturb CLARABEL's internal clock
            # source): prob.solver_stats.solve_time then systematically exceeds
            # the wall-clock time of the enclosing prob.solve() call, which is
            # impossible for a genuine per-call internal time. Falls back to the
            # (always-valid, if slightly conservative) wall-clock time instead
            # of reporting a number known to be wrong.
            print(f"  WARNING: {label} solver_stats.solve_time ({solve_t:.6f}s) exceeds "
                  f"wall time ({total_t:.6f}s) -- falling back to wall-clock time")
            reported[label] = total_t

    for label, relax_t in reported.items():
        out = RESULTS / f"knapsack_corr_{label}_20k" / "solve_times.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"key": f"knapsack_corr_{label}_20k", "n_timed": args.n,
                   "mean_relax_solve_s": relax_t,
                   "mean_local_solve_s": 6.872667069546879e-05,  # greedy, unchanged
                   "note": ("mean_relax_solve_s is the OFFLINE training-data-generation "
                            "solve: exact MISOCP (Gurobi) for the 'exact' row, continuous "
                            "SOCP relaxation (CLARABEL) for the 'relax' row -- not used at "
                            "deployment (mean_local_solve_s, greedy, is the deployed solver).")}
        out.write_text(json.dumps(payload, indent=2))
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
