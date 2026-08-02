"""Gate prototype for the CORRELATED-mu knapsack redesign (user's model).

Model (n=25):
  * values c_i ~ Uniform(20,30), FIXED (the "case"); objective max c^T x.
  * the only per-instance input is the ellipsoid center mu (25-dim). The
    uncertainty covariance is now FIXED (no longer a sampled parameter).
  * per instance mu ~ N(c, Sigma), Sigma = tau^2 [ (1-rho) I + rho 11^T ]:
    equal per-item variance tau^2, positive pairwise correlation rho. Centering
    mu at c makes heavier items more valuable (a genuine knapsack).
  * ONE fixed Sigma plays both roles: the robust uncertainty set in the
    constraint AND the instance-sampling distribution.

Robust constraint (ellipsoidal uncertainty, fixed Sigma):
    mu^T x + RHO_ROB * sqrt(x^T Sigma x) <= B,   B = BUDGET_FRAC * sum(c).
sqrt(x^T Sigma x) = || L^T x ||_2 with Sigma = L L^T (Cholesky), an exact SOC.

Why this should raise std(v*): independent per-item variation self-averages
(CLT) so v* is intrinsically concentrated. Positive correlation rho injects a
COMMON FACTOR (all weights heavy together -> tight budget -> low v*; all light
-> high v*), so v* swings a lot across instances. The idiosyncratic part keeps
the within-budget selection combinatorial (greedy stays imperfect).

GATE (this script): sweep (tau, rho); for each, sample mu, solve exact v*
(Gurobi) + greedy f; report std(v*), CV(v*), greedy exact-fraction, gap/std.
We want CV(v*) well above the ~0.03-0.04 independent-per-item baseline while
greedy_exact stays clearly below 100%.

Usage:
    /opt/anaconda3/envs/nn4opt/bin/python scripts/prototype_knapsack_corr.py
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import cvxpy as cp

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

N = 25
BUDGET_FRAC = 0.5
RHO_ROB = 1.0            # robustness multiplier on sqrt(x^T Sigma x)
ZERO_EPS = 1e-6         # |v* - f| <= ZERO_EPS*std(v*) counts as greedy-optimal
MU_FLOOR = 0.1          # clip sampled mu to stay a positive weight

_C = np.random.default_rng(2024).uniform(20.0, 30.0, size=N)   # fixed value vector
BUDGET = BUDGET_FRAC * _C.sum()


def sigma_matrix(tau, rho):
    """Sigma = tau^2 [ (1-rho) I + rho 11^T ]: equal variance, corr rho."""
    return tau ** 2 * ((1 - rho) * np.eye(N) + rho * np.ones((N, N)))


def build_misocp(Sigma):
    L = np.linalg.cholesky(Sigma)          # Sigma = L L^T
    x = cp.Variable(N, boolean=True)
    t = cp.Variable(nonneg=True)
    mu_p = cp.Parameter(N, nonneg=True)
    prob = cp.Problem(
        cp.Maximize(_C @ x),
        [mu_p @ x + RHO_ROB * t <= BUDGET, cp.SOC(t, L.T @ x)],
    )
    return prob, x, mu_p


def greedy(Sigma, mu):
    """Density greedy: rank by value/mean-weight c_i/mu_i, add while robust-feasible."""
    order = np.argsort(-_C / np.maximum(mu, 1e-12))
    x = np.zeros(N)
    for i in order:
        x[i] = 1
        robust = mu @ x + RHO_ROB * np.sqrt(x @ Sigma @ x)
        if robust > BUDGET:
            x[i] = 0
    return float(_C @ x)


def sample_mu(n_inst, tau, rho, seed):
    Sigma = sigma_matrix(tau, rho)
    r = np.random.default_rng(seed)
    mu = r.multivariate_normal(_C, Sigma, size=n_inst)
    return np.maximum(mu, MU_FLOOR)


def evaluate(tau, rho, n_inst, seed):
    Sigma = sigma_matrix(tau, rho)
    prob, x, mu_p = build_misocp(Sigma)
    mu = sample_mu(n_inst, tau, rho, seed)

    vstar = np.empty(n_inst)
    f = np.empty(n_inst)
    n_items = np.empty(n_inst)
    for i in range(n_inst):
        mu_p.value = mu[i]
        prob.solve(solver=cp.GUROBI, verbose=False)
        vstar[i] = float(prob.value)
        n_items[i] = int(round(np.sum(np.round(x.value)))) if x.value is not None else np.nan
        f[i] = greedy(Sigma, mu[i])

    gap = np.maximum(vstar - f, 0.0)
    sv = vstar.std()
    exact = float(np.mean(gap <= ZERO_EPS * sv))
    pos = gap[gap > ZERO_EPS * sv]
    return dict(tau=tau, rho=rho, mean_vstar=vstar.mean(), std_vstar=sv,
                cv_vstar=sv / vstar.mean(), mean_k=np.nanmean(n_items),
                greedy_exact=exact, mean_gap_std=gap.mean() / sv,
                med_pos_gap_std=(np.median(pos) / sv) if len(pos) else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-inst", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--taus", type=float, nargs="+", default=[2.0, 5.0])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.4, 0.8])
    args = ap.parse_args()

    print(f"n={N}  budget={BUDGET:.1f}  sum(c)={_C.sum():.1f}  RHO_ROB={RHO_ROB}  "
          f"n_inst={args.n_inst}\n", flush=True)
    header = ("  tau   rho   mean(v*)  std(v*)  CV(v*)  mean_k  greedy_exact  "
              "mean_gap/std  med_pos_gap/std")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for tau in args.taus:
        for rho in args.rhos:
            t0 = time.time()
            r = evaluate(tau, rho, args.n_inst, args.seed)
            print(f"  {r['tau']:.1f}   {r['rho']:.1f}   {r['mean_vstar']:7.2f}  "
                  f"{r['std_vstar']:6.3f}  {r['cv_vstar']:.4f}  {r['mean_k']:4.1f}   "
                  f"{r['greedy_exact']*100:5.1f}%       {r['mean_gap_std']:.4f}"
                  f"        {r['med_pos_gap_std']:.4f}   ({time.time()-t0:.0f}s)",
                  flush=True)

    print("\nGate: want CV(v*) >> 0.04 (the independent-per-item baseline) with\n"
          "greedy_exact clearly < 100%. rho drives CV(v*) up (common factor);\n"
          "compare rho=0 vs rho>0 at fixed tau to isolate the correlation effect.",
          flush=True)


if __name__ == "__main__":
    main()
