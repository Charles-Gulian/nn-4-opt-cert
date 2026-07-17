"""Custom branch-and-bound for the robust knapsack, with optional NN-based
pruning AND early stopping, each calibrated (independently) via split
conformal prediction.

Everything here works in the MIN-NATIVE representation (see relaxation.py):
we minimize sum (-v_i) x_i. Standard min-B&B terminology applies directly:
  - a node's continuous-relaxation value is a valid LOWER bound on the best
    achievable value in that subtree,
  - the incumbent (best integer-feasible value found) is a valid UPPER bound
    on the true minimum.

Two independent mechanisms, each with its own cutoff and its own risk profile
(see solve_bnb's docstring for the precise pruning/stopping rules):
  - prune_cutoff: standard fathoming, extended with an NN-derived cutoff.
    Danger direction: prune_cutoff UNDER-shooting the true optimum (i.e. the
    raw NN prediction OVER-shooting Cost) can discard the branch containing
    the true optimum. Calibrated from over-prediction residuals.
  - stop_cutoff: return immediately once a solution reaches this value,
    skipping the "prove no better solution exists" phase entirely -- per the
    oracle experiment, THIS is where essentially all of the speedup comes
    from (pruning alone barely beats standard search, since proving
    optimality requires exploring the same optimistic branches regardless of
    cutoff precision). Its risk profile differs from pruning's (accepting a
    solution found too early, far from optimal) and may need its own
    calibration; pass prune_cutoff=stop_cutoff=np.inf (defaults) to disable
    both -- the "baseline" arm, which must reproduce the exact
    Gurobi-certified optimum.

A cheap greedy initial incumbent is always seeded before search starts
(identically regardless of which cutoffs are active), so the algorithm never
returns "nothing" regardless of how poorly calibrated the cutoffs might be.
"""

import heapq
import itertools
import time

import numpy as np

from problems.robust_knapsack.problem import N_ITEMS, VALUES
from problems.robust_knapsack.relaxation import (
    build_relaxation_template, set_instance, solve_node,
)

_FRAC_TOL = 1e-6


class BnBResult:
    def __init__(self, value, x, nodes_explored, wall_time, status):
        self.value = value              # best MAX-sense objective found (sum v_i x_i)
        self.x = x                      # 0/1 selection vector achieving `value`
        self.nodes_explored = nodes_explored
        self.wall_time = wall_time
        self.status = status            # "solved" | "node_limit" | "time_limit"

    def __repr__(self):
        return (f"BnBResult(value={self.value:.4f}, nodes={self.nodes_explored}, "
                f"time={self.wall_time:.3f}s, status={self.status})")


def _greedy_incumbent(mu, sigma2):
    """Cheap value/weight-density greedy heuristic: sort items by v_i / mu_i
    descending, add while the robust capacity constraint stays feasible.
    Not part of the NN-vs-baseline comparison -- applied identically to both
    arms purely as a guaranteed-feasible safety net.

    sigma2 is the VARIANCE (as sampled/stored); sigma = sqrt(sigma2) is what
    enters the robust capacity bound, matching relaxation.py's convention.

    Returns (value, x) in MAX-sense (value = sum v_i x_i for the chosen x).
    """
    from problems.robust_knapsack.problem import RHO, BUDGET
    n = N_ITEMS
    sigma = np.sqrt(np.asarray(sigma2, dtype=float))
    order = np.argsort(-VALUES / np.maximum(mu, 1e-12))
    x = np.zeros(n)
    for i in order:
        x[i] = 1
        used = mu @ x
        # robust capacity check: mu@x + rho*||sigma o x||_2 <= B
        slack = BUDGET - used - RHO * np.sqrt(np.sum((sigma * x) ** 2))
        if slack < 0:
            x[i] = 0   # doesn't fit, skip
    value = float(VALUES @ x)
    return value, x


def solve_bnb(mu, sigma2, prune_cutoff=np.inf, stop_cutoff=np.inf,
              max_nodes=200_000, max_seconds=120, template=None, early_stop=False):
    """Custom branch-and-bound for one robust-knapsack instance.

    Two INDEPENDENT cutoffs, both in MIN-sense (already negated, directly
    comparable to node lower bounds), since pruning and stopping have
    different risk profiles and may be calibrated differently:

    Parameters
    ----------
    mu, sigma2 : the instance's ellipsoid center / variance (shape (N_ITEMS,)).
    prune_cutoff : float
        Used for standard fathoming: prune a node if its lower bound exceeds
        min(incumbent, prune_cutoff). Guards against discarding the branch
        containing the true optimum -- danger direction is prune_cutoff
        UNDER-shooting the true optimum (in min-sense), i.e. the raw NN
        prediction OVER-shooting Cost (max-sense). Pass np.inf (default) to
        disable NN-based pruning -- standard fathoming only.
    stop_cutoff : float
        Used for early termination: return immediately once ANY feasible
        integral solution reaches stop_cutoff, instead of continuing to
        search/prune until the tree is exhausted (only checked if
        early_stop=True). This is a DIFFERENT lever than pruning: pruning
        only discards branches whose bound has already fallen below the
        cutoff (which, per the oracle experiment, buys little speedup, since
        proving no better solution exists elsewhere requires exploring the
        same optimistic-looking branches regardless). Early stopping instead
        skips that proof-of-optimality phase entirely once a "good enough"
        solution is found -- the real source of speedup beyond pruning
        alone. May be calibrated independently from prune_cutoff (e.g. a
        different margin/tau), since its risk profile (accepting a
        found-too-early, far-from-optimal solution) differs from pruning's
        (discarding the true-optimal branch).
    early_stop : bool
        Enable the stop_cutoff check (see above). Default False.
    max_nodes, max_seconds : safety-valve budgets; if exceeded, return the
        best incumbent found so far with status reflecting the limit hit.
    template : optional pre-built relaxation template (from
        build_relaxation_template()) to reuse across multiple instances in a
        loop -- set_instance() is still called per instance.

    Returns
    -------
    BnBResult
    """
    n = N_ITEMS
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)

    tpl = template if template is not None else build_relaxation_template()
    set_instance(tpl, mu, sigma2)

    # Greedy initial incumbent (MAX-sense value/x), converted to MIN-sense.
    inc_value_max, inc_x = _greedy_incumbent(mu, sigma2)
    incumbent = -inc_value_max   # MIN-sense incumbent (upper bound on true min)

    lo0, hi0 = np.zeros(n), np.ones(n)
    t0 = time.time()
    root_bound, root_x, root_status = solve_node(tpl, lo0, hi0)

    nodes_explored = 0
    status = "solved"

    if root_status not in ("optimal", "optimal_inaccurate"):
        # Root infeasible -- shouldn't happen for a well-posed instance, but
        # fall back to the greedy incumbent rather than crash.
        return BnBResult(inc_value_max, inc_x, 0, time.time() - t0, "root_infeasible")

    counter = itertools.count()
    heap = [(root_bound, next(counter), lo0, hi0)]

    while heap:
        if nodes_explored >= max_nodes:
            status = "node_limit"
            break
        if time.time() - t0 > max_seconds:
            status = "time_limit"
            break

        bound, _, lo, hi = heapq.heappop(heap)
        # incumbent is a VERIFIED, already-achieved value: a tie (bound ==
        # incumbent) genuinely can't improve on what we already have, so >=
        # is correct and safe to prune. prune_cutoff is only a TARGET/belief,
        # not yet confirmed reachable -- a tie there might BE the true
        # optimum, not yet found, so it must survive (strict >) or we could
        # prune away the only path that discovers and records it. Conflating
        # the two into a single non-strict "bound >= min(incumbent,
        # prune_cutoff)" check caused exactly this: with an exact (oracle)
        # cutoff, the node containing the true optimum has bound ==
        # prune_cutoff and was being discarded before ever being confirmed as
        # the optimal solution.
        if bound >= incumbent - _FRAC_TOL or bound > prune_cutoff + _FRAC_TOL:
            continue

        nodes_explored += 1

        free = np.where((hi - lo) > _FRAC_TOL)[0]
        # Recompute x_val fresh (heap only stores the bound; re-solve is cheap
        # given DPP caching, and avoids storing large arrays in the heap).
        bound, x_val, node_status = solve_node(tpl, lo, hi)
        if node_status not in ("optimal", "optimal_inaccurate"):
            continue   # infeasible node, prune
        if bound >= incumbent - _FRAC_TOL or bound > prune_cutoff + _FRAC_TOL:
            continue

        if len(free) == 0:
            frac_mask = np.zeros(0, dtype=bool)
        else:
            frac_mask = np.minimum(x_val[free], 1 - x_val[free]) > _FRAC_TOL

        if not frac_mask.any():
            # Integral solution -- candidate incumbent.
            cand_x = np.clip(np.round(x_val), 0, 1).astype(int)
            cand_value_max = float(VALUES @ cand_x)
            cand_min = -cand_value_max
            if cand_min < incumbent - _FRAC_TOL:
                incumbent, inc_x = cand_min, cand_x
                if early_stop and incumbent <= stop_cutoff + _FRAC_TOL:
                    # Reached the calibrated target -- stop without proving
                    # optimality (that proof, per the oracle experiment, costs
                    # as much as ordinary search; skipping it is the point).
                    return BnBResult(-incumbent, inc_x, nodes_explored,
                                      time.time() - t0, "early_stop")
            continue

        # Most-fractional branching.
        free_frac_idx = free[frac_mask]
        frac_dist = np.abs(x_val[free_frac_idx] - 0.5)
        branch_var = free_frac_idx[np.argmin(frac_dist)]

        lo_a, hi_a = lo.copy(), hi.copy()
        hi_a[branch_var] = 0.0   # child: fix to 0
        lo_b, hi_b = lo.copy(), hi.copy()
        lo_b[branch_var] = 1.0   # child: fix to 1

        heapq.heappush(heap, (bound, next(counter), lo_a, hi_a))
        heapq.heappush(heap, (bound, next(counter), lo_b, hi_b))

    return BnBResult(-incumbent, inc_x, nodes_explored, time.time() - t0, status)
