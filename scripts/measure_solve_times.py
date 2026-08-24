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
import tempfile
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
    could not be parsed (e.g. the 'weak' solver path, which skips IPOPT).

    Captures fd 1 into a TEMP FILE, not a pipe: an os.pipe() has a small
    (~64KB on macOS/Linux) kernel buffer, and on the larger PEGASE cases
    IPOPT's per-iteration log at print_level=5 exceeds it well before the
    solve finishes -- since we only os.read() AFTER solve_local() returns,
    the ipopt subprocess blocks on write() waiting for a reader that itself
    is blocked waiting for the subprocess: a deadlock (confirmed: both
    processes sat at 0% CPU indefinitely). A file has no such blocking
    write, so it can't deadlock regardless of log size.
    """
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
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="ipopt_timing_")
    saved = os.dup(1)
    os.dup2(tmp_fd, 1)
    try:
        cost, res = solve_local(p, args=args)
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(tmp_fd)
        _problem_mod.pyo.SolverFactory = orig_factory
    with open(tmp_path, "r", errors="replace") as f:
        out = f.read()
    os.remove(tmp_path)
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
    """Return (mean solver-only s, mean end-to-end s, n_failed).

    A single numerically-hard random instance can make CLARABEL raise (seen
    on case300/case1354pegase's chordal SDP); catch per-sample so one bad
    draw doesn't abort the whole case's timing, matching how _time_local
    already handles IPOPT failures.
    """
    solver_ts, total_ts = [], []
    n_failed = 0
    for _ in range(n):
        p = X[np.random.randint(len(X))]
        t0 = time.perf_counter()
        try:
            solve_relaxation(p, args=args)
        except Exception:
            n_failed += 1
            continue
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
    return _agg(solver_ts), _agg(total_ts), n_failed


def _time_local(X, args, net, n):
    """Return (mean IPOPT-internal s, mean ASL-invocation s, mean end-to-end s,
    mean warm-start s, n_failed). IPOPT-internal is the true algorithmic time
    (parsed from IPOPT's own timing log); ASL-invocation is Pyomo's
    res.solver.time, which additionally includes NL-file I/O and subprocess
    launch overhead -- reported alongside so the gap between the two is
    visible, not hidden.

    BUG THIS GUARDS AGAINST: on a failed/non-converged solve, solve_local()
    returns res={"success": False} with NO 'solver_time_s' key, so the
    ASL-invocation sample is dropped by _agg() -- but the ipopt subprocess
    still ran (often LONGER than a successful solve, e.g. grinding through
    max_iter before giving up) and its timing log still parses fine. Averaging
    the two lists independently then mixes mismatched sample sets: hard,
    slow, failed instances inflate the "internal" mean while being silently
    excluded from the "invocation" mean, which can make internal > invocation
    -- a logical impossibility for any single solve. Fix: only count a sample
    on EITHER side when the solve actually succeeded, so both means are always
    over the identical set of calls.
    """
    import pandapower as pp
    ipopt_ts, asl_ts, total_ts, warm_ts = [], [], [], []
    n_load = len(net.load)
    n_failed = 0
    for _ in range(n):
        p = X[np.random.randint(len(X))]
        t0 = time.perf_counter()
        _, res, ipopt_s = _solve_local_capture_ipopt_time(p, args)
        total_ts.append(time.perf_counter() - t0)
        if isinstance(res, dict) and res.get("success") and res.get("solver_time_s") is not None:
            ipopt_ts.append(ipopt_s)
            asl_ts.append(res["solver_time_s"])
        else:
            n_failed += 1
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
    return _agg(ipopt_ts), _agg(asl_ts), _agg(total_ts), _agg(warm_ts), n_failed


def time_case(case, relaxations, n_local, n_relax=None):
    """Time one AC-OPF case across one or more relaxations, computing the
    LOCAL (IPOPT) solve time ONCE and reusing it for every relaxation's row.

    The local solver is IPOPT regardless of which relaxation it's being
    compared against -- SOCP and chordal-SDP rows for the same case
    previously each drew their own independent random sample for the local
    timing, so the "Local Solve" column could differ slightly between two
    rows that are, algorithmically, the exact same computation. Sharing one
    measurement removes that spurious difference.

    n_relax : optional {relaxation_name: n} override, so e.g. a slow chordal
    SDP can be timed with fewer samples than the (shared) local measurement
    without recomputing local timing at that smaller n.
    """
    n_relax = n_relax or {}
    vmin, vmax = VBOUNDS.get(case, (None, None))
    net, nd = load_network(case, v_min=vmin, v_max=vmax)
    X = np.load(DATA / f"X_test_5000_{case}_seed344.npy")

    # local (IPOPT) timing: computed ONCE per case, shared across relaxations.
    # relaxation choice is irrelevant to args here since solve_local doesn't
    # read args["relaxation"]; a throwaway prob_cache is fine.
    local_args = {"nd": nd, "net": net, "case_name": case, "prob_cache": {}}
    local_ipopt, local_asl, local_total, warm_s, n_failed = \
        _time_local(X, local_args, net, n_local)
    print(f"{'[local, shared]':32s} case={case:16s} IPOPT={local_ipopt:.4f}s "
          f"(asl-invoke={local_asl:.4f}s, n={n_local}"
          + (f", {n_failed} failed" if n_failed else "") + ")", flush=True)

    for relax in relaxations:
        n = n_relax.get(relax, n_local)
        args = {"nd": nd, "net": net, "case_name": case, "relaxation": relax,
                "prob_cache": {}}
        # Build once so the FIRST-call construction cost is not charged to a
        # timed sample; note this only removes cvxpy problem CONSTRUCTION --
        # with ignore_dpp (SDP, and SOCP on large nets) each solve still
        # re-canonicalizes, and that cost stays inside total_s. Try a few
        # samples: some instances are numerically hard for CLARABEL and raise
        # (seen reproducibly on case300's chordal SDP).
        t0 = time.perf_counter()
        for i in range(5):
            try:
                solve_relaxation(X[i], args=args)
                break
            except Exception:
                if i == 4:
                    raise
        build_s = time.perf_counter() - t0

        try:
            relax_solve, relax_total, n_relax_failed = _time_relaxation(X, args, relax, n)
        except Exception as e:
            print(f"  SKIP acopf_{relax}_{case}: {e}")
            continue

        cfg = f"acopf_{relax}_{case}"
        out = RESULTS / cfg / "solve_times.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": cfg, "n_timed_relax": n, "n_timed_local": n_local,
            "n_local_failed": n_failed, "n_relax_failed": n_relax_failed,
            # headline: TRUE solver-internal time on both sides (cvxpy
            # solver_stats vs IPOPT's own timing-statistics log, not Pyomo's
            # ASL-invocation time)
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
                     "this measurement only). local_solve_s is computed ONCE per case (n_timed_"
                     "local samples) and REUSED across every relaxation row for that case, since "
                     "the local solver (IPOPT) does not depend on which relaxation it's compared "
                     "against. asl_invoke_s = Pyomo's res.solver.time, which additionally "
                     "includes NL-file I/O and subprocess launch -- a roughly fixed per-call "
                     "overhead that dominates on small problems (e.g. ~10ms extra on case9's "
                     "~3ms solve) and is diluted on large ones; reported separately rather than "
                     "folded into solve_s. total_s = end-to-end including modelling/"
                     "canonicalization, power-flow warm start and subprocess. n_*_failed samples "
                     "(non-converged/infeasible) are EXCLUDED from both solve_s and asl_invoke_s "
                     "so the two are always averaged over the identical set of calls -- mixing "
                     "sample sets previously let internal time appear to exceed invocation time, "
                     "which is impossible for any single call."),
        }
        out.write_text(json.dumps(payload, indent=2))
        fail_note = f"  [{n_failed}/{n_local} local, {n_relax_failed}/{n} relax failed]" \
            if (n_failed or n_relax_failed) else ""
        print(f"{cfg:32s} SOLVER relax={relax_solve:8.4f}s (n={n}) local={local_ipopt:8.4f}s"
              f"{fail_note} (asl-invoke={local_asl:.4f}s)  |  TOTAL relax={relax_total:7.3f}s "
              f"local={local_total:7.3f}s (pf warm start {warm_s:.3f}s)", flush=True)


def _solve_ik_local_capture_ipopt_time(p, args, ik_problem_mod):
    """Same true-internal-IPOPT-time capture as _solve_local_capture_ipopt_time,
    generalized for problems.ik.problem.solve_local, which imports IPOPT as a
    bare `SolverFactory` name (`from pyomo.environ import SolverFactory`)
    rather than AC-OPF's `pyo.SolverFactory` -- so the monkeypatch target is
    that module-level name instead."""
    orig_factory = ik_problem_mod.SolverFactory

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

    ik_problem_mod.SolverFactory = _patched
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="ipopt_timing_")
    saved = os.dup(1)
    os.dup2(tmp_fd, 1)
    try:
        cost, res = ik_problem_mod.solve_local(p, args=args)
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(tmp_fd)
        ik_problem_mod.SolverFactory = orig_factory
    with open(tmp_path, "r", errors="replace") as f:
        out = f.read()
    os.remove(tmp_path)
    total = sum(float(m.group(1)) for m in _IPOPT_TIME_RE.finditer(out))
    return cost, res, (total if total > 0 else None)


def time_ik(n):
    """Time IK's two relaxations (Shor / Lasserre-2), sharing ONE local
    (IPOPT) timing measurement across both -- same fix as AC-OPF's SOCP/SDP,
    since problems.ik.problem.solve_local doesn't depend on the relaxation."""
    import pandas as pd
    import cvxpy as cp
    import problems.ik.problem as ik

    df = pd.read_csv(PROJECT_ROOT / "data" / "ik_lass1" / "test_5000.csv")
    P = df[["xd", "yd"]].values.astype(float)

    ipopt_ts, asl_ts, n_failed = [], [], 0
    for _ in range(n):
        p = P[np.random.randint(len(P))]
        cost, res, ipopt_s = _solve_ik_local_capture_ipopt_time(p, {}, ik)
        asl_t = None
        if isinstance(res, dict):
            for key in ("solver_time_s", "solver_time"):
                if key in res:
                    asl_t = res[key]
        # solve_local doesn't currently surface solver.time the way AC-OPF's
        # does; fall back to None (asl_invoke context field will be blank)
        # rather than fabricate a number.
        if ipopt_s is not None:
            ipopt_ts.append(ipopt_s)
            asl_ts.append(asl_t)
        else:
            n_failed += 1
    local_ipopt = _agg(ipopt_ts)
    local_asl = _agg(asl_ts)
    print(f"{'[local, shared]':32s} IK  IPOPT={local_ipopt:.4f}s  n={n}"
          + (f"  {n_failed} failed" if n_failed else ""), flush=True)

    # NOTE: MOSEK license expired in this environment (mosek.Error:
    # err_license_expired, uncaught by cvxpy's SolverError fallback in
    # problem.py) -- use CLARABEL instead for this timing measurement only.
    configs = [("ik_lass1", ik.solve_relaxation, cp.CLARABEL),
               ("ik_lass2", ik.solve_lasserre2, cp.CLARABEL)]
    for key, relax_fn, solver in configs:
        prob_cache = {}
        args = {"prob_cache": prob_cache, "solver": solver}
        relax_fn(P[0], args=args)   # warm the cache (build) before timing
        ts = []
        for _ in range(n):
            p = P[np.random.randint(len(P))]
            relax_fn(p, args=args)
            # cache key differs by relaxation (solve_relaxation: (l1,l2);
            # solve_lasserre2: ("lasserre2",l1,l2)) -- after warm-up above,
            # prob_cache has exactly one entry, so just take it rather than
            # hardcode either convention.
            prob = next(iter(prob_cache.values()))[0]
            try:
                ts.append(prob.solver_stats.solve_time)
            except Exception:
                pass
        relax_solve = _agg(ts)

        out = PROJECT_ROOT / "results" / key / "solve_times.json"   # NOT under acopf-cert
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "key": key, "n_timed_relax": n, "n_timed_local": n,
            "mean_relax_solve_s": relax_solve,
            "mean_local_solve_s": local_ipopt,
            "mean_local_asl_invoke_s": local_asl,
            "note": ("mean_local_solve_s is IPOPT's true internal solve time "
                     "(same measurement methodology as AC-OPF), computed ONCE "
                     "and shared across ik_lass1/ik_lass2 since both compare "
                     "against the same local solver."),
        }, indent=2))
        print(f"{key:32s} SOLVER relax={relax_solve:.4f}s (n={n})  local={local_ipopt:.4f}s "
              f"[shared]", flush=True)


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
    p.add_argument("--ik", action="store_true")
    p.add_argument("--n-chordal-large", type=int, default=None,
                   help="override sample count for chordal_sdp specifically on "
                        "case1354pegase/case2869pegase (slow: seconds/solve)")
    args = p.parse_args()
    if args.knapsack:
        time_knapsack(args.n); return
    if args.ik:
        time_ik(args.n); return
    for case in args.cases:
        relaxations = [r for r in args.relax
                       if not (r == "chordal_sdp" and case == "case2869pegase"
                               and args.n_chordal_large is None)]
        if not relaxations:
            continue
        n_relax = {}
        if args.n_chordal_large is not None and case in ("case1354pegase", "case2869pegase"):
            n_relax["chordal_sdp"] = args.n_chordal_large
        try:
            time_case(case, relaxations, args.n, n_relax=n_relax)
        except Exception as e:
            print(f"  SKIP {case}: {e}")


if __name__ == "__main__":
    main()
