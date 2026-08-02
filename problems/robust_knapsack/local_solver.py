"""Fast (heuristic) local solvers for the robust 0-1 knapsack.

The certification framework certifies the feasible value f produced by a fast
local solver. We provide two, of different quality, to demonstrate that the
certificate adapts to the solver (the point of the experiment, not the solver
itself):

  "greedy"      -- density-greedy: rank items by value/nominal-weight v_i/mu_i,
                   add while the robust capacity holds. Ignores the uncertainty
                   in the ranking (it only enters the feasibility check), so it
                   is comparatively weak.  (== branch_and_bound._greedy_incumbent)

  "socp_round"  -- solve the CONTINUOUS SOCP relaxation (x in [0,1], which does
                   account for the robust SOC constraint), then round: add items
                   in order of the fractional solution while robust-feasible.
                   Returns the BEST of this rounding and the greedy incumbent, so
                   it is feasible and never worse than greedy.

Both return (value, x) in MAX-sense (value = VALUES @ x for a feasible 0-1 x).
All feasibility uses the same robust capacity check as the rest of the module:
    mu @ x + RHO * ||sigma o x||_2 <= BUDGET,  sigma = sqrt(sigma2).
"""

import numpy as np

from problems.robust_knapsack.problem import N_ITEMS, VALUES, RHO, BUDGET
from problems.robust_knapsack.branch_and_bound import _greedy_incumbent
from problems.robust_knapsack.relaxation import (
    build_relaxation_template, set_instance, solve_node,
)


def _robust_feasible(x, mu, sigma):
    """Robust capacity check: mu@x + RHO*||sigma o x|| <= BUDGET."""
    return BUDGET - mu @ x - RHO * np.sqrt(np.sum((sigma * x) ** 2)) >= 0.0


def _round_by_order(order, mu, sigma):
    """Greedily set x_i = 1 in the given item order while robust-feasible."""
    x = np.zeros(N_ITEMS)
    for i in order:
        x[i] = 1.0
        if not _robust_feasible(x, mu, sigma):
            x[i] = 0.0
    return x


def socp_rounding_incumbent(mu, sigma2, template=None):
    """SOCP-relaxation rounding heuristic (>= greedy).

    template : reusable dict from build_relaxation_template(); if None, one is
               built per call (fine for one-offs, but pass a shared template
               when solving many instances -- one build, many cheap re-solves).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.sqrt(np.asarray(sigma2, dtype=float))
    if template is None:
        template = build_relaxation_template()

    set_instance(template, mu, sigma2)
    lb, x_frac, status = solve_node(template, np.zeros(N_ITEMS), np.ones(N_ITEMS))

    # Greedy incumbent is always available as a feasible floor.
    g_val, g_x = _greedy_incumbent(mu, sigma2)

    if x_frac is None:
        return g_val, g_x

    # Round the fractional solution: add items with the largest relaxed x first.
    x_round = _round_by_order(np.argsort(-x_frac), mu, sigma)
    r_val = float(VALUES @ x_round)

    if r_val >= g_val:
        return r_val, x_round
    return g_val, g_x


# Dispatch by name -----------------------------------------------------------

_SOLVERS = {
    "greedy": lambda mu, s2, template=None: _greedy_incumbent(mu, s2),
    "socp_round": socp_rounding_incumbent,
}


def solve_local(mu, sigma2, method="greedy", template=None):
    """Return (value, x) for the named local solver ('greedy' or 'socp_round')."""
    if method not in _SOLVERS:
        raise ValueError(f"unknown local solver {method!r}; choose from {list(_SOLVERS)}")
    if method == "socp_round":
        return socp_rounding_incumbent(mu, sigma2, template=template)
    return _SOLVERS[method](mu, sigma2)
