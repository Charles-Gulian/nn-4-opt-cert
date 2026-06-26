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


def _solve_tracked(label, fn, *args, **kwargs):
    """Run fn(*args, **kwargs), returning (value, result, elapsed_s, peak_mb)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    value, result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1024 ** 2
    print(f"  {label}: {value:.4f}  [{result.get('status', '?')}, "
          f"exact={result.get('exact')}, time={elapsed:.2f}s, peak={peak_mb:.0f} MB]",
          flush=True)
    return value, result, elapsed, peak_mb


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
    results["local"] = _solve_tracked(
        "Local (IPOPT)", solve_local, p, args=shared)

    results["socp"] = _solve_tracked(
        "SOCP         ", solve_relaxation, p,
        args={**shared, "relaxation": "socp"})

    results["chordal_sdp"] = _solve_tracked(
        "Chordal SDP  ", solve_relaxation, p,
        args={**shared, "relaxation": "chordal_sdp"})

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
