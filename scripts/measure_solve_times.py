"""Measure per-instance solve cost of the convex relaxation vs the fast local
solver, on an EQUAL FOOTING, for every AC-OPF config (and the knapsack heuristic).

WHY THIS WAS REWRITTEN
----------------------
The previous version compared apples to oranges and made the relaxations look
absurdly (up to ~60x) faster than IPOPT:

  * the relaxation was called ONCE before timing to warm `prob_cache`, so the
    timed calls were a DPP parameter update + conic solve -- model construction
    and canonicalization excluded;
  * every timed `solve_local` call, by contrast, paid the FULL cold pipeline:
    a `copy.deepcopy` of the pandapower net, an entire Newton-Raphson AC power
    flow (`pp.runpp`) for the warm start, construction of a fresh
    `pyo.ConcreteModel` (thousands of constraints built in Python), NL-file
    writing, an `ipopt` SUBPROCESS launch, and result parsing.

So the numbers contrasted a steady-state conic solve with a cold Python
modelling pipeline plus process spawn -- an artifact of the harness, not a
property of the algorithms.

WHAT WE REPORT NOW
------------------
For both sides, two consistently-defined quantities:

  solve_s  (HEADLINE, solver-only): time attributable to the numerical solver,
           excluding model construction on both sides.
           - relaxation: cvxpy `prob.solver_stats.solve_time` (CLARABEL/MOSEK)
             -- the solver's own internal time.
           - local     : `res.solver.time` from Pyomo (surfaced by solve_local
             as result["solver_time_s"]). CAVEAT: for the ASL/`ipopt`
             interface this is the wall time of the solver INVOCATION, so it
             still contains NL-file write/read and process launch; it is an
             upper bound on IPOPT's internal iteration time. We report it as-is
             rather than parsing the IPOPT log, and note the caveat wherever
             these numbers are used -- the file-based interface is a real cost
             of the Pyomo/ASL toolchain, but it is not algorithmic work.
           Neither side includes modelling/canonicalization.

  total_s  (end-to-end, warm process): full wall-clock of one `solve_*` call as
           the pipeline actually invokes it, including (for the relaxation) any
           re-canonicalization -- note `ignore_dpp=True` is used for the SDP
           relaxations and for large SOCP, so those DO re-canonicalize every
           solve -- and (for the local solver) deepcopy + power-flow warm start
           + Pyomo build + NL write + subprocess.

We additionally break the local pipeline's overhead out into its power-flow
warm-start component (`local_warmstart_s`), since that is itself a full AC
power-flow solve, so the paper can explain the gap between solve_s and total_s.

Timing is sensitive to CPU contention -- run this on an otherwise idle machine.

Usage:
    python scripts/measure_solve_times.py --n 10
    python scripts/measure_solve_times.py --n 5 --cases case1354pegase --relax chordal_sdp
    python scripts/measure_solve_times.py --knapsack
"""
import argparse
import copy
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

DATA = (PROJECT_ROOT / "data" / "acopf-hpc"
        if (PROJECT_ROOT / "data" / "acopf-hpc").exists()
        else PROJECT_ROOT / "data" / "acopf")
RESULTS = PROJECT_ROOT / "results" / "acopf-cert"
VBOUNDS = {"case300": (0.90, 1.10), "case1354pegase": (0.90, 1.10),
           "case2869pegase": (0.90, 1.10)}
ALL_CASES = ["case9", "case14", "case39", "case89pegase",
             "case118", "case300", "case1354pegase", "case2869pegase"]


def _agg(vals):
    """Mean of the finite entries, or NaN if none reported."""
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def _time_relaxation(X, args, cache_key, n):
    """Return (mean solver-only s, mean end-to-end s) for the relaxation."""
    solver_ts, total_ts = [], []
    for _ in range(n):
        p = X[np.random.randint(len(X))]
        t0 = time.perf_counter()
        solve_relaxation(p, args=args)
        total_ts.append(time.perf_counter() - t0)
        # cvxpy records the solver's internal time on the cached problem object
        built = args["prob_cache"].get(cache_key)
        prob = built[0] if isinstance(built, (tuple, list)) else built
        st = None
        try:
            st = prob.solver_stats.solve_time
        except Exception:
            pass
        solver_ts.append(st)
    return _agg(solver_ts), _agg(total_ts)


def _time_local(X, args, net, n):
    """Return (mean IPOPT-only s, mean end-to-end s, mean warm-start s)."""
    import pandapower as pp
    solver_ts, total_ts, warm_ts = [], [], []
    n_load = len(net.load)
    for _ in range(n):
        p = X[np.random.randint(len(X))]
        t0 = time.perf_counter()
        _, res = solve_local(p, args=args)
        total_ts.append(time.perf_counter() - t0)
        solver_ts.append(res.get("solver_time_s") if isinstance(res, dict) else None)
        # isolate the pandapower power-flow warm start (part of the overhead)
        try:
            netc = copy.deepcopy(net)
            netc.load["p_mw"] = p[:n_load]
            netc.load["q_mvar"] = p[n_load:2 * n_load]
            t1 = time.perf_counter()
            pp.runpp(netc, numba=False, verbose=False)
            warm_ts.append(time.perf_counter() - t1)
        except Exception:
            warm_ts.append(None)
    return _agg(solver_ts), _agg(total_ts), _agg(warm_ts)


def time_acopf(case, relax, n):
    vmin, vmax = VBOUNDS.get(case, (None, None))
    net, nd = load_network(case, v_min=vmin, v_max=vmax)
    X = np.load(DATA / f"X_test_5000_{case}_seed344.npy")
    args = {"nd": nd, "net": net, "case_name": case, "relaxation": relax,
            "prob_cache": {}}

    # Build once so the FIRST-call construction cost is not charged to a timed
    # sample; note this only removes cvxpy problem CONSTRUCTION -- with
    # ignore_dpp (SDP, and SOCP on large nets) each solve still re-canonicalizes,
    # and that cost stays inside total_s.
    t0 = time.perf_counter()
    solve_relaxation(X[0], args=args)
    build_s = time.perf_counter() - t0

    relax_solve, relax_total = _time_relaxation(X, args, relax, n)
    local_solve, local_total, warm_s = _time_local(X, args, net, n)

    cfg = f"acopf_{relax}_{case}"
    out = RESULTS / cfg / "solve_times.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": cfg, "n_timed": n,
        # headline: solver-internal time on both sides
        "mean_relax_solve_s": relax_solve,
        "mean_local_solve_s": local_solve,
        # context
        "mean_relax_total_s": relax_total,
        "mean_local_total_s": local_total,
        "mean_local_warmstart_s": warm_s,
        "relax_first_build_s": build_s,
        "note": ("solve_s = solver-only: cvxpy solver_stats.solve_time vs Pyomo "
                 "res.solver.time (the latter includes NL I/O + process launch, so it "
                 "upper-bounds IPOPT's internal time); total_s = end-to-end including "
                 "modelling/canonicalization, power-flow warm start and subprocess"),
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"{cfg:32s} SOLVER relax={relax_solve:8.4f}s local={local_solve:8.4f}s  |  "
          f"TOTAL relax={relax_total:7.3f}s local={local_total:7.3f}s "
          f"(pf warm start {warm_s:.3f}s)", flush=True)


def time_knapsack(n):
    import pandas as pd
    from problems.robust_knapsack.branch_and_bound import _greedy_incumbent, N_ITEMS
    df = pd.read_csv(PROJECT_ROOT / "data" / "robust_knapsack" / "test_20000.csv")
    mu = df[[f"Mu{i}" for i in range(N_ITEMS)]].values.astype(float)
    s2 = df[[f"Sigma2{i}" for i in range(N_ITEMS)]].values.astype(float)
    ts = []
    for _ in range(n):
        i = np.random.randint(len(mu))
        t0 = time.perf_counter(); _greedy_incumbent(mu[i], s2[i])
        ts.append(time.perf_counter() - t0)
    t = float(np.mean(ts))
    for ntr in (20000, 80000):
        out = PROJECT_ROOT / "results" / f"knapsack_n{ntr}" / "solve_times.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"key": f"knapsack_n{ntr}", "n_timed": n,
                                   "mean_relax_solve_s": None,   # N/A: no relaxation
                                   "mean_local_solve_s": t,
                                   "mean_local_total_s": t}, indent=2))
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
