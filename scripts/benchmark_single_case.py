"""Single-instance benchmark: solve one AC-OPF at nominal demand and report
peak memory, solve time, optimal cost, and relaxation gap for each method.

Intended as a pre-flight check before launching a full parallel data-gen job,
especially for large cases where per-worker memory usage is unknown.

Usage
-----
    python scripts/benchmark_single_case.py --case case1354pegase
    python scripts/benchmark_single_case.py --case case300 --v-min 0.90 --v-max 1.10
"""

import argparse
import pathlib
import sys
import time
import tracemalloc

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from problems.acopf.network import load_network
from problems.acopf.problem import solve_local, solve_relaxation


def _rss_mb():
    """Resident set size of this process in MB (includes native solver allocations)."""
    try:
        import resource, sys
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is bytes on macOS, kilobytes on Linux.
        return ru / 1024 ** 2 if sys.platform == "darwin" else ru / 1024
    except Exception:
        return float("nan")


def _solve_tracked(label, fn, *args, **kwargs):
    """Run fn(*args, **kwargs), returning (value, result, elapsed_s, peak_mb)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    value, result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1024 ** 2
    rss = _rss_mb()
    print(f"  {label}: {value:.4f}  [{result.get('status', '?')}, "
          f"exact={result.get('exact')}, time={elapsed:.2f}s, "
          f"tracemalloc_peak={peak_mb:.0f} MB, rss={rss:.0f} MB]",
          flush=True)
    return value, result, elapsed, peak_mb


def _checkpoint(tag):
    print(f"  [mem] {tag}: rss={_rss_mb():.0f} MB", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Single-instance AC-OPF benchmark.")
    parser.add_argument("--case",  default="case1354pegase")
    parser.add_argument("--v-min", type=float, default=None)
    parser.add_argument("--v-max", type=float, default=None)
    args = parser.parse_args()

    print(f"Case: {args.case}", flush=True)
    net, nd = load_network(args.case, v_min=args.v_min, v_max=args.v_max)
    print(f"Buses: {nd.n_buses}  Branches: {len(nd.branch_from)}  "
          f"Gens: {nd.n_gens}  Loads: {nd.n_loads}")
    print(f"Voltage bounds: [{nd.v_min.min():.2f}, {nd.v_max.max():.2f}] pu\n",
          flush=True)

    p = np.concatenate([nd.pd_nominal, nd.qd_nominal])
    shared = {"nd": nd}

    results = {}

    print("Solving...", flush=True)
    _checkpoint("before local")
    results["local"] = _solve_tracked(
        "Local (IPOPT)", solve_local, p, args=shared)

    # SOCP: instrument build, canonicalize, and solve separately.
    import cvxpy as cp
    from problems.acopf.problem import _build_socp_problem
    _checkpoint("before socp build")
    socp_built = _build_socp_problem(nd)
    _checkpoint("after socp build")
    prob_socp = socp_built[0]
    Pd_param_socp, Qd_param_socp = socp_built[1], socp_built[2]
    Pd_param_socp.value = np.asarray(p[:nd.n_loads])
    Qd_param_socp.value = np.asarray(p[nd.n_loads:])
    print(f"  SOCP problem: {prob_socp.size_metrics}", flush=True)
    _checkpoint("before socp canonicalize")
    data, chain, inverse_data = prob_socp.get_problem_data(solver=cp.CLARABEL)
    _checkpoint("after socp canonicalize")
    print(f"  Cone data shapes: "
          f"A={data['A'].shape}, b={data['b'].shape}, "
          f"c={data['c'].shape}", flush=True)
    t0 = time.perf_counter()
    prob_socp.solve(solver=cp.CLARABEL, verbose=True)
    socp_time = time.perf_counter() - t0
    _checkpoint("after socp solve")
    socp_val = float(prob_socp.value) if prob_socp.status in ("optimal", "optimal_inaccurate") else float("nan")
    results["socp"] = (socp_val, {"status": prob_socp.status, "exact": False}, socp_time, float("nan"))
    print(f"  SOCP: {socp_val:.4f}  [status={prob_socp.status}, time={socp_time:.2f}s]", flush=True)

    # Free all SOCP objects before building chordal SDP.
    del socp_built, prob_socp, data, chain, inverse_data
    import gc; gc.collect()
    _checkpoint("after socp cleanup")

    from problems.acopf.problem import _build_chordal_sdp_problem
    _checkpoint("before chordal_sdp build")
    chordal_built = _build_chordal_sdp_problem(nd)
    _checkpoint("after chordal_sdp build")
    prob_chordal, Pd_c, Qd_c = chordal_built[0], chordal_built[1], chordal_built[2]
    Pd_c.value = np.asarray(p[:nd.n_loads])
    Qd_c.value = np.asarray(p[nd.n_loads:])
    _checkpoint("before chordal_sdp canonicalize")
    # ignore_dpp=True: skip CVXPY's parametrized (DPP) canonicalization, which
    # materializes a parameter-affine tensor sized by the canonical problem and
    # OOMs once the per-bag PSD cones make that large (see problem.py solve path).
    data_c, chain_c, inv_c = prob_chordal.get_problem_data(
        solver=cp.CLARABEL, ignore_dpp=True)
    _checkpoint("after chordal_sdp canonicalize")
    t0 = time.perf_counter()
    prob_chordal.solve(solver=cp.CLARABEL, verbose=False, ignore_dpp=True)
    chordal_time = time.perf_counter() - t0
    _checkpoint("after chordal_sdp solve")
    chordal_val = float(prob_chordal.value) if prob_chordal.status in ("optimal", "optimal_inaccurate") else float("nan")
    results["chordal_sdp"] = (chordal_val, {"status": prob_chordal.status, "exact": False}, chordal_time, float("nan"))
    print(f"  Chordal SDP: {chordal_val:.4f}  [status={prob_chordal.status}, time={chordal_time:.2f}s]", flush=True)

    # ── summary table ─────────────────────────────────────────────────────────
    local_val = results["local"][0]
    print(f"\n{'Method':<16} {'Cost ($/hr)':>14} {'Gap vs local':>14} "
          f"{'Exact':>7} {'Time (s)':>10} {'Peak (MB)':>11}")
    print("-" * 78)
    for key, label in [
        ("local",       "Local (IPOPT)"),
        ("socp",        "SOCP"),
        ("chordal_sdp", "Chordal SDP"),
    ]:
        val, res, elapsed, peak_mb = results[key]
        gap = 100 * (local_val - val) / local_val if (local_val and val) else float("nan")
        exact_str = "—" if res.get("exact") is None else str(res["exact"])
        print(f"{label:<16} {val:>14.4f} {gap:>13.4f}%  "
              f"{exact_str:>7} {elapsed:>10.2f} {peak_mb:>10.0f}")


if __name__ == "__main__":
    main()
