"""Prototype: does higher item-value DISPERSION raise the spread of the optimal
cost v*(theta) while keeping the greedy local solver imperfect?

Hypothesis (user): the current knapsack case has nearly-interchangeable items
(all identical weight profiles mu0=1, sigma0^2=0.1; values clustered in
[0.5,1.5]), so v*(theta) is a fine-grained staircase of tiny near-random steps
-- the worst case for an NN. Increasing the DISPERSION of the fixed value vector
should (a) let greedy do a little better (clearer value ranking) but NOT perfectly,
and (b) let the NN do relatively much better, because v* is then dominated by
"how many HIGH-value items fit", a more structured function of theta. Net: the
certification ratio q99/std(v*) should improve.

IMPORTANT: dispersion, not scale. Multiplying all values by a constant scales v*
and std(v*) equally -- ratio unchanged (the monotone-transform trap). We vary the
COEFFICIENT OF VARIATION of the (fixed) value vector via a lognormal with spread s:
    v_i = exp(s * z_i),  z_i ~ N(0,1) fixed per level,  then rescaled to mean 1.
    CV(values) = sqrt(exp(s^2) - 1);  s=0 => all items identical.
The current case corresponds to CV ~ 0.29 (Uniform(0.5,1.5)); we sweep s upward.

This is the GATE before the full 20k/5k NN run: we only measure v* spread and the
greedy gap here (Gurobi exact v* + density-greedy f over a modest sample). If a
dispersion level gives higher std(v*) AND greedy still leaves a real gap, we then
run the NN experiment at that level.

Usage:
    /opt/anaconda3/envs/nn4opt/bin/python scripts/prototype_knapsack_variance.py
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import cvxpy as cp

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

N = 50
RHO = 2.0
MU0 = 1.0
SIGMA2_0 = 0.1
BUDGET = 0.5 * N * MU0
MU_RANGE = 0.5          # mu_i ~ Uniform((1-r)mu0, (1+r)mu0), matches generate_data default
SIGMA2_RANGE = 0.5
ZERO_EPS = 1e-6         # |v* - f| <= ZERO_EPS * std(v*) counts as greedy-optimal


def make_values(s, seed=2024):
    """Fixed value vector with lognormal dispersion s, rescaled to mean 1."""
    z = np.random.default_rng(seed).standard_normal(N)
    v = np.exp(s * z)
    return v / v.mean()


def build_misocp(values):
    x = cp.Variable(N, boolean=True)
    t = cp.Variable(nonneg=True)
    mu_p = cp.Parameter(N, nonneg=True)
    sig_p = cp.Parameter(N, nonneg=True)
    prob = cp.Problem(
        cp.Maximize(values @ x),
        [mu_p @ x + RHO * t <= BUDGET, cp.SOC(t, cp.multiply(sig_p, x))],
    )
    return prob, x, mu_p, sig_p


def greedy(values, mu, sigma2):
    sigma = np.sqrt(sigma2)
    order = np.argsort(-values / np.maximum(mu, 1e-12))
    x = np.zeros(N)
    for i in order:
        x[i] = 1
        if BUDGET - mu @ x - RHO * np.sqrt(np.sum((sigma * x) ** 2)) < 0:
            x[i] = 0
    return float(values @ x)


def sample(n_inst, seed):
    r = np.random.default_rng(seed)
    mu = r.uniform((1 - MU_RANGE) * MU0, (1 + MU_RANGE) * MU0, size=(n_inst, N))
    s2 = r.uniform((1 - SIGMA2_RANGE) * SIGMA2_0, (1 + SIGMA2_RANGE) * SIGMA2_0,
                   size=(n_inst, N))
    return mu, s2


def evaluate_level(s, n_inst, seed):
    values = make_values(s)
    cv_vals = values.std() / values.mean()
    prob, x, mu_p, sig_p = build_misocp(values)
    mu, s2 = sample(n_inst, seed)

    vstar = np.empty(n_inst)
    f = np.empty(n_inst)
    for i in range(n_inst):
        mu_p.value = mu[i]
        sig_p.value = np.sqrt(s2[i])
        prob.solve(solver=cp.GUROBI, verbose=False)
        vstar[i] = float(prob.value)
        f[i] = greedy(values, mu[i], s2[i])

    gap = np.maximum(vstar - f, 0.0)          # max-sense: v* >= f
    sv = vstar.std()
    exact_frac = float(np.mean(gap <= ZERO_EPS * sv))
    pos = gap[gap > ZERO_EPS * sv]
    return {
        "s": s, "cv_vals": cv_vals,
        "mean_vstar": vstar.mean(), "std_vstar": sv, "cv_vstar": sv / vstar.mean(),
        "greedy_exact_frac": exact_frac,
        "mean_gap_over_std": gap.mean() / sv,
        "median_pos_gap_over_std": (np.median(pos) / sv) if len(pos) else float("nan"),
        "max_gap_over_std": gap.max() / sv,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-inst", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--s-grid", type=float, nargs="+",
                    default=[0.0, 0.29, 0.5, 0.8, 1.2])
    args = ap.parse_args()

    print(f"n_inst={args.n_inst}  budget={BUDGET}  mu_range={MU_RANGE}  "
          f"sigma2_range={SIGMA2_RANGE}\n", flush=True)
    header = ("  s     CV(vals)  mean(v*)  std(v*)  CV(v*)  greedy_exact  "
              "mean_gap/std  med_pos_gap/std  max_gap/std")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in args.s_grid:
        t0 = time.time()
        r = evaluate_level(s, args.n_inst, args.seed)
        print(f"  {r['s']:.2f}   {r['cv_vals']:.3f}    {r['mean_vstar']:7.3f}  "
              f"{r['std_vstar']:6.4f}  {r['cv_vstar']:.3f}   "
              f"{r['greedy_exact_frac']*100:5.1f}%       "
              f"{r['mean_gap_over_std']:.4f}       {r['median_pos_gap_over_std']:.4f}"
              f"        {r['max_gap_over_std']:.3f}   ({time.time()-t0:.0f}s)",
              flush=True)

    print("\nRead: we want a level where std(v*) (and CV(v*)) is HIGHER than the\n"
          "current case (CV(vals)~0.29) while greedy_exact stays well below 100%%\n"
          "(a real negative class remains). That is the candidate for the NN run.",
          flush=True)


if __name__ == "__main__":
    main()
