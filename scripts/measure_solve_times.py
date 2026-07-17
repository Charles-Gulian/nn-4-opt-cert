"""Measure mean convex-relaxation and local (IPOPT) solve times per AC-OPF config
(and the knapsack greedy heuristic), writing solve_times.json in the same schema
the small problems already use, so scripts/compute_final_tables.py picks them up.

Times the SOLVE only: the cvxpy problem is built once (cache warmed) before timing,
matching how data generation reuses the cached problem across instances.

Usage (local, fast configs):
    python scripts/measure_solve_times.py --n 10
    python scripts/measure_solve_times.py --n 10 --cases case9 case14 --relax socp
    python scripts/measure_solve_times.py --knapsack
SAVIO (slow ones):
    python scripts/measure_solve_times.py --n 5 --cases case1354pegase --relax chordal_sdp
    python scripts/measure_solve_times.py --n 5 --cases case2869pegase --relax socp
"""
import argparse
import json
import pathlib
import sys
import time
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("pyomo").setLevel(logging.ERROR)   # silence W1002 init spam
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from problems.acopf.network import load_network
from problems.acopf.problem import solve_relaxation, solve_local

# SAVIO writes X samples to data/acopf; the laptop pull-back is data/acopf-hpc.
DATA = (PROJECT_ROOT / "data" / "acopf-hpc"
        if (PROJECT_ROOT / "data" / "acopf-hpc").exists()
        else PROJECT_ROOT / "data" / "acopf")
RESULTS = PROJECT_ROOT / "results" / "acopf-cert"
VBOUNDS = {"case300": (0.90, 1.10), "case1354pegase": (0.90, 1.10),
           "case2869pegase": (0.90, 1.10)}
ALL_CASES = ["case9", "case14", "case39", "case89pegase",
             "case118", "case300", "case1354pegase", "case2869pegase"]


def _mean_time(fn, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.mean(ts))


def time_acopf(case, relax, n):
    vmin, vmax = VBOUNDS.get(case, (None, None))
    net, nd = load_network(case, v_min=vmin, v_max=vmax)
    X = np.load(DATA / f"X_test_5000_{case}_seed344.npy")
    args = {"nd": nd, "net": net, "case_name": case, "relaxation": relax,
            "prob_cache": {}}
    solve_relaxation(X[0], args=args)   # warm cache (build once), not timed
    relax_t = _mean_time(lambda: solve_relaxation(X[np.random.randint(len(X))], args=args), n)
    local_t = _mean_time(lambda: solve_local(X[np.random.randint(len(X))], args=args), n)
    cfg = f"acopf_{relax}_{case}"
    out = RESULTS / cfg / "solve_times.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"key": cfg, "n_timed": n,
                               "mean_relax_solve_s": relax_t,
                               "mean_local_solve_s": local_t}, indent=2))
    print(f"{cfg:34s} relax={relax_t:.3f}s  local={local_t:.3f}s  -> {out.name}")


def time_knapsack(n):
    import pandas as pd
    from problems.robust_knapsack.branch_and_bound import _greedy_incumbent, N_ITEMS
    df = pd.read_csv(PROJECT_ROOT / "data" / "robust_knapsack" / "test_20000.csv")
    mu = df[[f"Mu{i}" for i in range(N_ITEMS)]].values.astype(float)
    s2 = df[[f"Sigma2{i}" for i in range(N_ITEMS)]].values.astype(float)
    t = _mean_time(lambda: _greedy_incumbent(mu[np.random.randint(len(mu))],
                                             s2[np.random.randint(len(s2))]), n)
    for ntr in (20000, 80000):
        out = PROJECT_ROOT / "results" / f"knapsack_n{ntr}" / "solve_times.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"key": f"knapsack_n{ntr}", "n_timed": n,
                                   "mean_relax_solve_s": None,   # N/A: no relaxation
                                   "mean_local_solve_s": t}, indent=2))
    print(f"knapsack greedy local={t*1e3:.3f}ms  (relaxation = N/A)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--cases", nargs="+", default=ALL_CASES)
    p.add_argument("--relax", nargs="+", default=["socp", "chordal_sdp"])
    p.add_argument("--knapsack", action="store_true")
    args = p.parse_args()
    if args.knapsack:
        time_knapsack(args.n); return
    for case in args.cases:
        for relax in args.relax:
            if relax == "chordal_sdp" and case == "case2869pegase":
                continue
            try:
                time_acopf(case, relax, args.n)
            except Exception as e:
                print(f"  SKIP acopf_{relax}_{case}: {e}")


if __name__ == "__main__":
    main()
