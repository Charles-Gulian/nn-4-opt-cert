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
import os
import pathlib
import re
import sys
import time
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("pyomo").setLevel(logging.ERROR)   # silence W1002 init spam
import numpy as np
import pyomo.environ as pyo

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from problems.acopf.network import load_network
from problems.acopf.problem import solve_relaxation, solve_local
import problems.acopf.problem as _problem_mod

# Pyomo's res.solver.time (surfaced as result["solver_time_s"]) is the wall
# time of the WHOLE `ipopt` invocation via the ASL file interface: it includes
# writing the NL file, launching/tearing down the subprocess, and reading
# results back -- overhead that is roughly FIXED per call and dominates on
# small problems (measured ~10ms on case9, vs ~3ms of actual IPOPT iteration).
# To get IPOPT's true internal solve time, ask it to print its own timing
# breakdown (only possible at print_level>=1; production runs use
# print_level=0, so this must be enabled here) and parse it out of a captured
# copy of its stdout. This is used ONLY by this benchmarking script -- it does
# not touch problems/acopf/problem.py's solve_local, which stays at
# print_level=0 (quiet) for actual data generation and training.
_IPOPT_TIME_RE = re.compile(
    r"Total seconds in (?:IPOPT \(w/o function evaluations\)|NLP function evaluations)"
    r"\s*=\s*([\d.]+)")


def _solve_local_capture_ipopt_time(p, args):
    """Call solve_local with IPOPT's timing statistics enabled, returning
    (cost, res, ipopt_internal_s) where the third value is None if the log
    could not be parsed (e.g. the 'weak' solver path, which skips IPOPT)."""
    orig_factory = pyo.SolverFactory

    class _TimedWrap:
        def __init__(self, inner):
            self._inner = inner
            self.options = inner.options

        def solve(self, m, **kw):
            self.options["print_timing_statistics"] = "yes"
            self.options["print_level"] = 5
            kw["tee"] = True
            return self._inner.solve(m, **kw)

    def _patched(name, *a, **k):
        s = orig_factory(name, *a, **k)
        return _TimedWrap(s) if name == "ipopt" else s

    _problem_mod.pyo.SolverFactory = _patched
    r_fd, w_fd = os.pipe()
    saved = os.dup(1)
    os.dup2(w_fd, 1)
    try:
        cost, res = solve_local(p, args=args)
    finally:
        os.dup2(saved, 1)
        os.close(w_fd)
        os.close(saved)
        _problem_mod.pyo.SolverFactory = orig_factory
    out = os.read(r_fd, 10 ** 7).decode(errors="replace")
    os.close(r_fd)
    total = sum(float(m.group(1)) for m in _IPOPT_TIME_RE.finditer(out))
    return cost, res, (total if total > 0 else None)

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
    """Return (mean IPOPT-internal s, mean ASL-invocation s, mean end-to-end s,
    mean warm-start s). IPOPT-internal is the true algorithmic time (parsed
    from IPOPT's own timing log); ASL-invocation is Pyomo's res.solver.time,
    which additionally includes NL-file I/O and subprocess launch overhead --
    reported alongside so the gap between the two is visible, not hidden."""
    import pandapower as pp
    ipopt_ts, asl_ts, total_ts, warm_ts = [], [], [], []
    n_load = len(net.load)
    for _ in range(n):
        p = X[np.random.randint(len(X))]
        t0 = time.perf_counter()
        _, res, ipopt_s = _solve_local_capture_ipopt_time(p, args)
        total_ts.append(time.perf_counter() - t0)
        ipopt_ts.append(ipopt_s)
        asl_ts.append(res.get("solver_time_s") if isinstance(res, dict) else None)
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
    return _agg(ipopt_ts), _agg(asl_ts), _agg(total_ts), _agg(warm_ts)


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
    local_ipopt, local_asl, local_total, warm_s = _time_local(X, args, net, n)

    cfg = f"acopf_{relax}_{case}"
    out = RESULTS / cfg / "solve_times.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": cfg, "n_timed": n,
        # headline: TRUE solver-internal time on both sides (cvxpy solver_stats
        # vs IPOPT's own timing-statistics log, not Pyomo's ASL-invocation time)
        "mean_relax_solve_s": relax_solve,
        "mean_local_solve_s": local_ipopt,
        # context
        "mean_local_asl_invoke_s": local_asl,
        "mean_relax_total_s": relax_total,
        "mean_local_total_s": local_total,
        "mean_local_warmstart_s": warm_s,
        "relax_first_build_s": build_s,
        "note": ("solve_s = TRUE solver-internal time: cvxpy solver_stats.solve_time vs "
                 "IPOPT's own 'Total seconds in IPOPT' timing log (print_level bumped for "
                 "this measurement only). asl_invoke_s = Pyomo's res.solver.time, which "
                 "additionally includes NL-file I/O and subprocess launch -- a roughly "
                 "fixed per-call overhead that dominates on small problems (e.g. ~10ms "
                 "extra on case9's ~3ms solve) and is diluted on large ones; reported "
                 "separately rather than folded into solve_s. total_s = end-to-end "
                 "including modelling/canonicalization, power-flow warm start and subprocess."),
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"{cfg:32s} SOLVER relax={relax_solve:8.4f}s local={local_ipopt:8.4f}s "
          f"(asl-invoke={local_asl:.4f}s)  |  TOTAL relax={relax_total:7.3f}s "
          f"local={local_total:7.3f}s (pf warm start {warm_s:.3f}s)", flush=True)


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
