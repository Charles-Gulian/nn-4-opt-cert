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


def _worker(worker_id, n_solves, seed_base, out_queue):
    t_start = time.time()
    try:
        from problems.robust_knapsack.problem import _build_misocp, solve_exact
        from problems.robust_knapsack.generate_data import sample_parameters

        t_env0 = time.time()
        prob, x, t, mu_param, sigma2_param, sigma_param = _build_misocp()
        t_env = time.time() - t_env0

        X = sample_parameters(n_solves, args={"seed": seed_base + worker_id})
        args = {"prob": prob, "x": x, "t": t, "mu_param": mu_param,
                "sigma2_param": sigma2_param, "sigma_param": sigma_param}

        values = []
        t_solve0 = time.time()
        for i in range(n_solves):
            val, info = solve_exact(X[i], args=args)
            values.append((val, info["status"]))
        t_solve = time.time() - t_solve0

        n_ok = sum(1 for v, s in values if s == "optimal")
        out_queue.put({
            "worker_id": worker_id, "ok": True, "n_ok": n_ok, "n_total": n_solves,
            "env_setup_s": t_env, "solve_s": t_solve,
            "wall_s": time.time() - t_start, "error": None,
        })
    except Exception as e:
        out_queue.put({
            "worker_id": worker_id, "ok": False, "n_ok": 0, "n_total": n_solves,
            "env_setup_s": None, "solve_s": None,
            "wall_s": time.time() - t_start, "error": repr(e),
        })


def main():
    p = argparse.ArgumentParser(description="Probe Gurobi license concurrency.")
    p.add_argument("--workers", type=int, default=8,
                    help="Number of concurrent processes, each acquiring its own Gurobi env.")
    p.add_argument("--solves", type=int, default=5,
                    help="Number of solves each worker performs.")
    p.add_argument("--seed-base", type=int, default=90000,
                    help="Seed offset so workers sample distinct instances.")
    args = p.parse_args()

    print(f"Launching {args.workers} concurrent workers, {args.solves} solves each ...", flush=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    t0 = time.time()
    for wid in range(args.workers):
        proc = ctx.Process(target=_worker, args=(wid, args.solves, args.seed_base, q))
        proc.start()
        procs.append(proc)

    results = [q.get() for _ in procs]
    for proc in procs:
        proc.join()
    total_wall = time.time() - t0

    results.sort(key=lambda r: r["worker_id"])
    print(f"\n{'worker':>6} {'ok':>5} {'solved':>8} {'env_s':>8} {'solve_s':>8} {'wall_s':>8}  error", flush=True)
    for r in results:
        env_s = f"{r['env_setup_s']:.2f}" if r["env_setup_s"] is not None else "-"
        solve_s = f"{r['solve_s']:.2f}" if r["solve_s"] is not None else "-"
        print(f"{r['worker_id']:>6} {str(r['ok']):>5} {r['n_ok']}/{r['n_total']:<6} "
              f"{env_s:>8} {solve_s:>8} {r['wall_s']:>8.2f}  {r['error'] or ''}", flush=True)

    n_workers_ok = sum(1 for r in results if r["ok"] and r["n_ok"] == r["n_total"])
    print(f"\n{n_workers_ok}/{args.workers} workers fully succeeded. "
          f"Total wall time: {total_wall:.1f}s.", flush=True)

    if n_workers_ok == args.workers:
        print("All concurrent workers succeeded -- the license appears to support "
              f"at least {args.workers}-way concurrency. Safe to parallelize "
              "knapsack generation at this worker count (try higher to find the cap "
              "if you want more throughput).", flush=True)
    elif n_workers_ok == 0:
        print("No workers succeeded concurrently -- this license likely allows only "
              "1 concurrent Gurobi session. Keep knapsack generation serial.", flush=True)
    else:
        print(f"Partial success ({n_workers_ok}/{args.workers}) -- the license caps "
              f"concurrency somewhere below {args.workers}. Re-run with fewer "
              "--workers to find the real limit before parallelizing generation.",
              flush=True)


if __name__ == "__main__":
    main()
