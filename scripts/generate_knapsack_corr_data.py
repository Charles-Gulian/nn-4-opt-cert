"""Generate the correlated-mu robust-knapsack dataset (labels only; no training).

Writes to data/knapsack_corr/ INSIDE THE REPO -- an earlier version of this data
lived in the session scratchpad under /tmp and was purged by the OS tmp cleaner,
so it is deliberately persisted with the project now.

Model (n=25): values c ~ Uniform(20,30) FIXED (the "case"); the only per-instance
input is the ellipsoid center mu ~ N(c, Sigma) with
Sigma = tau^2[(1-rho)I + rho*11^T] (equal variance tau^2, positive correlation rho).
The uncertainty covariance is FIXED, not sampled. Robust constraint:
    mu^T x + RHO_ROB * sqrt(x^T Sigma x) <= B,   B = BUDGET_FRAC * sum(c).
Positive correlation defeats the CLT self-averaging that makes an
independent-per-item knapsack's optimal value nearly constant.

Two labels per instance:
    vstar : exact MISOCP optimum (Gurobi)      -- ground truth, and the "exact" target
    vr    : continuous SOCP relaxation (CLARABEL) -- the smoother "relax" target
plus the greedy local-solver value f on the test split.

A SINGLE pool is generated; the n=20k configs train on pool[:20000] and the n=80k
configs on the whole pool, so the data-scaling comparison is a clean nested subset
rather than two independent samples.

Parallelised with a process pool (Gurobi Threads=1 per worker): these are tiny
25-binary-variable MISOCPs, so throughput comes from processes, not from B&B
threads -- one all-core B&B per worker would oversubscribe the machine.

Usage:
    .../python scripts/generate_knapsack_corr_data.py --n-pool 80000 --n-test 5000 --workers 3
"""

import argparse
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "knapsack_corr"

N = 25
BUDGET_FRAC = 0.5
RHO_ROB = 1.0
MU_FLOOR = 0.1
TAU = 5.0
RHO_CORR = 0.8
CASE_SEED = 2024

_C = np.random.default_rng(CASE_SEED).uniform(20.0, 30.0, size=N)
BUDGET = BUDGET_FRAC * _C.sum()
SIGMA = TAU ** 2 * ((1 - RHO_CORR) * np.eye(N) + RHO_CORR * np.ones((N, N)))
_L = np.linalg.cholesky(SIGMA)

_W = {}          # per-worker cvxpy problem cache


def _init_worker():
    import cvxpy as cp
    x_e = cp.Variable(N, boolean=True)
    x_r = cp.Variable(N)
    out = {}
    for name, x, extra in (("exact", x_e, []), ("relax", x_r, None)):
        t = cp.Variable(nonneg=True)
        mu_p = cp.Parameter(N, nonneg=True)
        cons = [mu_p @ x + RHO_ROB * t <= BUDGET, cp.SOC(t, _L.T @ x)]
        if name == "relax":
            cons += [x >= 0, x <= 1]
        out[name] = (cp.Problem(cp.Maximize(_C @ x), cons), mu_p)
    _W["probs"] = out
    _W["cp"] = cp


def _solve_one(mu):
    cp = _W["cp"]
    probs = _W["probs"]
    p_e, mu_e = probs["exact"]
    mu_e.value = mu
    try:
        p_e.solve(solver=cp.GUROBI, verbose=False, Threads=1)
        vstar = float(p_e.value)
    except Exception:
        vstar = float("nan")
    p_r, mu_r = probs["relax"]
    mu_r.value = mu
    try:
        p_r.solve(solver=cp.CLARABEL, verbose=False)
        vr = float(p_r.value)
    except Exception:
        vr = float("nan")
    return vstar, vr


def greedy(mu):
    order = np.argsort(-_C / np.maximum(mu, 1e-12))
    x = np.zeros(N)
    for i in order:
        x[i] = 1
        if mu @ x + RHO_ROB * np.sqrt(x @ SIGMA @ x) > BUDGET:
            x[i] = 0
    return float(_C @ x)


def sample_mu(n, seed):
    r = np.random.default_rng(seed)
    return np.maximum(r.multivariate_normal(_C, SIGMA, size=n), MU_FLOOR)


def run(mu, workers, label):
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        res = list(ex.map(_solve_one, mu, chunksize=64))
    vstar = np.array([r[0] for r in res])
    vr = np.array([r[1] for r in res])
    bad = int(np.sum(~np.isfinite(vstar)))
    print(f"[{label}] {len(mu)} solved in {time.time()-t0:.0f}s "
          f"({bad} failed)", flush=True)
    return vstar, vr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pool", type=int, default=80000)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # test split first: it is small and unblocks everything downstream
    test_p = OUT_DIR / f"test_{args.n_test}.npz"
    if not test_p.exists():
        mu_te = sample_mu(args.n_test, 200)
        vstar, vr = run(mu_te, args.workers, "test")
        f = np.array([greedy(m) for m in mu_te])
        np.savez(test_p, mu=mu_te, vstar=vstar, vr=vr, f=f)
        print(f"  wrote {test_p.name}: std(v*)={np.nanstd(vstar):.3f} "
              f"mean gap v_r-v*={np.nanmean(vr-vstar):.4f} "
              f"greedy exact={100*np.mean(np.abs(vstar-f)<1e-6):.1f}%", flush=True)
    else:
        print(f"  {test_p.name} exists, skipping", flush=True)

    pool_p = OUT_DIR / f"pool_{args.n_pool}.npz"
    if not pool_p.exists():
        mu = sample_mu(args.n_pool, 100)
        vstar, vr = run(mu, args.workers, "pool")
        np.savez(pool_p, mu=mu, vstar=vstar, vr=vr)
        print(f"  wrote {pool_p.name}", flush=True)
    else:
        print(f"  {pool_p.name} exists, skipping", flush=True)
    print("KNAPSACK_CORR_DATA_DONE", flush=True)


if __name__ == "__main__":
    main()
