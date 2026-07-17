"""Empirically test how many concurrent Gurobi solves the current license
supports, rather than trusting docs.

The knapsack data-gen job (scripts/generate_knapsack_data.py) currently solves
serially (~39 ms/solve, ~7h for 640k). MIMO's generic pipeline parallelizes
generation across N_WORKERS processes. Before doing the same for knapsack, we
need to know whether the WLS academic license here permits multiple
simultaneous Gurobi environments/solves, or is capped (e.g. at 1-2 concurrent
tokens), in which case extra workers would just queue/fail rather than help.

This spawns `--workers` processes, each acquiring its own Gurobi Env and
solving a handful of tiny/real knapsack-shaped MISOCPs at roughly the same
time, and reports per-worker success/failure and timing so we can see:
  - all succeed quickly -> concurrency is fine, safe to parallelize generation
  - some fail/hang/error on env acquisition -> license caps concurrent use

Usage (run on a SAVIO compute node, same env as the real job):
    python scripts/probe_gurobi_concurrency.py --workers 8 --solves 5
"""

import argparse
import multiprocessing as mp
import pathlib
import sys
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _worker(worker_id, n_solves, seed_base, out_queue, threads=None):
    t_start = time.time()
    try:
        from problems.robust_knapsack.problem import _build_misocp, solve_exact
        from problems.robust_knapsack.generate_data import sample_parameters

        t_env0 = time.time()
        prob, x, t, mu_param, sigma2_param, sigma_param = _build_misocp()
        t_env = time.time() - t_env0

        X = sample_parameters(n_solves, args={"seed": seed_base + worker_id})
        # Cap Gurobi threads per worker so K concurrent B&B solves don't
        # oversubscribe the node's cores (the prime suspect for the per-solve
        # slowdown -- MISOCP branch-and-bound defaults to Threads=0 = all cores).
        solver_opts = {"Threads": threads} if threads else {}
        args = {"prob": prob, "x": x, "t": t, "mu_param": mu_param,
                "sigma2_param": sigma2_param, "sigma_param": sigma_param,
                "solver_opts": solver_opts}

        # Time the FIRST solve separately: the Gurobi session is acquired lazily
        # on the first solve, so under license-burst contention this call eats
        # the one-time session-acquisition cost. Subsequent solves reuse the
        # session and should run at full speed -- this split is exactly what
        # distinguishes "one-time startup throttle" from "per-solve throttle".
        values = []
        t_first0 = time.time()
        val, info = solve_exact(X[0], args=args)
        t_first = time.time() - t_first0
        values.append((val, info["status"]))

        t_rest0 = time.time()
        for i in range(1, n_solves):
            val, info = solve_exact(X[i], args=args)
            values.append((val, info["status"]))
        n_rest = max(n_solves - 1, 0)
        t_rest = time.time() - t_rest0
        steady_per_solve = (t_rest / n_rest) if n_rest > 0 else float("nan")

        n_ok = sum(1 for v, s in values if s == "optimal")
        out_queue.put({
            "worker_id": worker_id, "ok": True, "n_ok": n_ok, "n_total": n_solves,
            "env_setup_s": t_env, "first_solve_s": t_first,
            "steady_per_solve_s": steady_per_solve, "solve_s": t_first + t_rest,
            "wall_s": time.time() - t_start, "error": None,
        })
    except Exception as e:
        out_queue.put({
            "worker_id": worker_id, "ok": False, "n_ok": 0, "n_total": n_solves,
            "env_setup_s": None, "first_solve_s": None,
            "steady_per_solve_s": None, "solve_s": None,
            "wall_s": time.time() - t_start, "error": repr(e),
        })


def main():
    p = argparse.ArgumentParser(description="Probe Gurobi license concurrency.")
    p.add_argument("--workers", type=int, default=4,
                    help="Number of concurrent processes, each acquiring its own Gurobi env.")
    p.add_argument("--solves", type=int, default=30,
                    help="Solves per worker. Use enough (>=20) to amortize the one-time "
                         "session-acquisition cost and reveal steady-state per-solve time.")
    p.add_argument("--seed-base", type=int, default=90000,
                    help="Seed offset so workers sample distinct instances.")
    p.add_argument("--threads", type=int, default=None,
                    help="Gurobi Threads per worker (default: unset = all cores). Set to "
                         "~floor(allocated_cores / workers) to avoid CPU oversubscription.")
    args = p.parse_args()

    tinfo = f", {args.threads} Gurobi threads each" if args.threads else ", default (all-core) threads"
    print(f"Launching {args.workers} concurrent workers, {args.solves} solves each{tinfo} ...", flush=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    t0 = time.time()
    for wid in range(args.workers):
        proc = ctx.Process(target=_worker, args=(wid, args.solves, args.seed_base, q, args.threads))
        proc.start()
        procs.append(proc)

    results = [q.get() for _ in procs]
    for proc in procs:
        proc.join()
    total_wall = time.time() - t0

    results.sort(key=lambda r: r["worker_id"])
    print(f"\n{'worker':>6} {'ok':>5} {'solved':>8} {'first_s':>8} {'steady/solve':>13} "
          f"{'wall_s':>8}  error", flush=True)
    for r in results:
        first_s = f"{r['first_solve_s']:.2f}" if r.get("first_solve_s") is not None else "-"
        steady = (f"{r['steady_per_solve_s']:.3f}"
                  if r.get("steady_per_solve_s") is not None else "-")
        print(f"{r['worker_id']:>6} {str(r['ok']):>5} {r['n_ok']}/{r['n_total']:<6} "
              f"{first_s:>8} {steady:>13} {r['wall_s']:>8.2f}  {r['error'] or ''}", flush=True)

    n_workers_ok = sum(1 for r in results if r["ok"] and r["n_ok"] == r["n_total"])
    ok_results = [r for r in results if r["ok"] and r.get("steady_per_solve_s") is not None]
    print(f"\n{n_workers_ok}/{args.workers} workers fully succeeded. "
          f"Total wall time: {total_wall:.1f}s.", flush=True)

    if ok_results:
        import statistics
        steady_vals = [r["steady_per_solve_s"] for r in ok_results]
        first_vals = [r["first_solve_s"] for r in ok_results]
        mean_steady = statistics.mean(steady_vals)
        max_first = max(first_vals)
        print(f"  Among succeeders: max first-solve (session acquisition) = "
              f"{max_first:.1f}s; mean STEADY-STATE per-solve = {mean_steady:.3f}s.",
              flush=True)
        # Steady-state near the serial ~0.15s means the throttle is a one-time
        # per-worker startup cost, not per-solve -- safe to parallelize.
        if mean_steady < 1.0:
            eff = mean_steady / max(n_workers_ok, 1)
            print(f"  => Steady-state solves run at full speed. The startup cost is "
                  f"paid ONCE per worker and is negligible over a long run. "
                  f"Effective throughput at {n_workers_ok} workers ~= "
                  f"{eff:.3f}s/row; 640k rows ~= {640000*eff/3600:.1f}h.", flush=True)
        else:
            print("  => Steady-state per-solve is still elevated (>1s) -- parallelism "
                  "is genuinely throttled here, not just at startup. Reconsider "
                  "worker count or fall back to serial.", flush=True)

    if n_workers_ok == args.workers:
        print(f"All {args.workers} workers succeeded -- safe at this count. "
              "Increase --workers to find the burst ceiling if you want more.", flush=True)
    elif n_workers_ok == 0:
        print("No workers succeeded concurrently -- try fewer, or wait for lingering "
              "sessions to expire (check the license dashboard 'Active sessions').",
              flush=True)
    else:
        print(f"Partial success ({n_workers_ok}/{args.workers}) -- burst ceiling is "
              f"around {n_workers_ok} concurrent sessions on this license. Set "
              f"KNAPSACK_WORKERS at or below {n_workers_ok} (with headroom).",
              flush=True)


if __name__ == "__main__":
    main()
