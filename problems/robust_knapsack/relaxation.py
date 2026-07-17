"""Continuous-relaxation SOCP for one branch-and-bound node.

Everything in this module works in the MIN-NATIVE representation confirmed
for the B&B design: we minimize sum (-v_i) x_i (the negation of the knapsack's
value objective), so that standard min-B&B terminology applies without any
sign games at call sites:
    - a node's continuous relaxation value is a valid LOWER bound on the best
      (minimum) achievable value in that node's subtree,
    - an incumbent (achieved feasible value) is a valid UPPER bound on the
      true minimum.

A "node" is a partial assignment of the n binary variables: some fixed to 0,
some fixed to 1, the rest free in [0,1]. This is represented as per-variable
lower/upper bound arrays (lo, hi), with lo=hi=0 or lo=hi=1 for fixed variables
and lo=0,hi=1 for free ones.

The relaxation template is built ONCE per B&B run (mu/sigma fixed for the
whole run -- one instance's tree) with lo/hi as CVXPY Parameters, so each
node's bound is a cheap Parameter update + re-solve, not a problem rebuild.
"""

import numpy as np
import cvxpy as cp

from problems.robust_knapsack.problem import N_ITEMS, VALUES, RHO, BUDGET

COSTS = -VALUES   # minimize sum COSTS_i * x_i  <=>  maximize sum VALUES_i * x_i


def build_relaxation_template():
    """Build the reusable continuous-relaxation SOCP for one B&B run.

    mu_param/sigma_param are fixed for the whole run (one problem instance);
    lo_param/hi_param change per node. Returns a dict of the CVXPY objects
    needed by solve_node().
    """
    n = N_ITEMS
    x = cp.Variable(n)              # continuous (NOT boolean=True -- see note below)
    t = cp.Variable(nonneg=True)
    mu_param = cp.Parameter(n, nonneg=True)
    sigma_param = cp.Parameter(n, nonneg=True)   # sqrt(sigma2), same convention as problem.py
    lo_param = cp.Parameter(n)
    hi_param = cp.Parameter(n)

    constraints = [
        mu_param @ x + RHO * t <= BUDGET,
        cp.SOC(t, cp.multiply(sigma_param, x)),
        x >= lo_param,
        x <= hi_param,
    ]
    # NOTE: x is a plain continuous cp.Variable, NOT boolean=True. A boolean
    # variable forces CVXPY onto the MIP solver path regardless of the bound
    # constraints, and CLARABEL doesn't support boolean variables at all --
    # this must be a genuinely separate problem object from problem.py's
    # exact MISOCP (which is solved once, by Gurobi, purely to generate the
    # ground-truth "Cost" labels; this one is solved many times per node, by
    # a fast continuous conic solver).
    objective = cp.Minimize(COSTS @ x)
    prob = cp.Problem(objective, constraints)
    return {
        "prob": prob, "x": x, "t": t,
        "mu_param": mu_param, "sigma_param": sigma_param,
        "lo_param": lo_param, "hi_param": hi_param,
    }


def set_instance(template, mu, sigma2):
    """Fix the (mu, sigma^2) instance parameters for an entire B&B run.

    sigma2 is the VARIANCE (as sampled/stored); sigma = sqrt(sigma2) is what
    enters the SOC constraint, matching problem.py's convention.
    """
    template["mu_param"].value = np.asarray(mu, dtype=float)
    template["sigma_param"].value = np.sqrt(np.asarray(sigma2, dtype=float))


def solve_node(template, lo, hi, solver=cp.CLARABEL):
    """Solve one node's relaxation. Returns (lower_bound, x_val, status).

    lower_bound is np.inf if infeasible (a pruned/dead branch, never a
    candidate for the incumbent or further branching).
    """
    template["lo_param"].value = lo
    template["hi_param"].value = hi
    prob = template["prob"]
    prob.solve(solver=solver, verbose=False)

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return np.inf, None, prob.status
    return float(prob.value), np.asarray(template["x"].value, dtype=float).copy(), prob.status
