"""Robust 0-1 knapsack under ellipsoidal weight uncertainty.

Each item i has a fixed, known value v_i. Its weight is uncertain, lying in an
ellipsoid centered at mu with per-axis variance sigma_i^2 = Sigma_ii:

    U = { w : (w - mu)^T Sigma^{-1} (w - mu) <= rho^2 },   Sigma = diag(sigma^2)

Substituting xi = Sigma^{-1/2}(w - mu) (so ||xi||_2 <= rho and
w_i = mu_i + sigma_i * xi_i, sigma_i = sqrt(sigma_i^2)), robust feasibility --
the capacity constraint must hold for every w in U -- reduces via
Cauchy-Schwarz to a single exact second-order-cone constraint:

    max_{||xi||_2<=rho} sum_i sigma_i x_i xi_i = rho * ||sigma o x||_2
    (o = elementwise product; attained at xi_i proportional to sigma_i x_i)

so "for all w in U, sum_i w_i x_i <= B" becomes:

    sum_i mu_i x_i + rho * ||sigma o x||_2 <= B

giving the MISOCP (using x_i in {0,1} => x_i^2 = x_i so ||sigma o x||_2 is
exactly sqrt(sum sigma_i^2 x_i), no approximation):

    max_{x in {0,1}^n, t>=0}  sum_i v_i x_i
    s.t.  sum_i mu_i x_i + rho * t <= B
          ||(sigma_1 x_1, ..., sigma_n x_n)||_2 <= t     (t, sigma o x) in SOC

Item values v, item count n, robustness parameter rho, and budget B are FIXED
("the case", generated once below with a fixed seed) -- analogous to the
fixed network topology in the AC-OPF problem. Only the ellipsoid center mu and
variance sigma^2 vary across problem instances: theta = (mu, sigma^2), a
2n-vector, directly analogous to AC-OPF's (Pd, Qd). Note sigma^2 (variance) is
what is sampled/stored -- sigma = sqrt(sigma^2) is computed only where it
enters the SOC constraint below.

Standard problem interface:
    solve_exact(p, args=None) -> (value, result)

Unlike AC-OPF there is no separate relaxation/local-solver split: Gurobi
solves the MISOCP to certified global optimality directly, so "Cost" already
IS the true optimal value -- a genuinely combinatorial (non-smooth) function
of theta, in contrast to AC-OPF's near-linear relaxation value.
"""

import numpy as np
import cvxpy as cp

N_ITEMS = 50

# ---------------------------------------------------------------------------
# Fixed problem instance ("the case"): item values, nominal weight/uncertainty,
# robustness parameter, and knapsack budget.  Generated once with a fixed seed
# so every sample (train, test, and any future data) shares the same case.
#
# All items share the SAME nominal weight/uncertainty profile (mu0_i = 1,
# sigma0^2_i = 0.1 for every i) -- so the only thing that can make one item
# preferable to another is its value.  Values are chosen clustered around a
# common center v0 with a small asymmetric perturbation: similar enough that
# the problem is a genuine combinatorial selection (not greedy-solvable via a
# dominant value/weight ratio), but not perfectly symmetric (which would leave
# many exactly-tied optimal solutions).
# ---------------------------------------------------------------------------
_CASE_SEED = 2024
_rng = np.random.default_rng(_CASE_SEED)

MU_NOMINAL = np.full(N_ITEMS, 1.0)          # mu0_i = 1 for all items
SIGMA2_NOMINAL = np.full(N_ITEMS, 0.1)      # sigma0^2_i = 0.1 for all items

# Values are fixed (not sampled per instance) but drawn with the same 0.5x-1.5x
# relative-spread convention as the mu sampling range, so items are not all
# equally valuable (breaking the perfectly-symmetric, uninteresting case) while
# still keeping weight/uncertainty identical across items -- the combinatorial
# hardness comes from items having similar weight profiles but now genuinely
# differentiated values.
VALUES = _rng.uniform(0.5, 1.5, size=N_ITEMS)   # v_i, fixed

RHO = 2.0                                                    # robustness parameter
BUDGET_FRACTION = 0.5
BUDGET = float(BUDGET_FRACTION * MU_NOMINAL.sum())           # B, fixed


def _build_misocp():
    """Build the reusable CVXPY MISOCP. mu/sigma2 are Parameters so the same
    problem object can be re-solved across samples without rebuilding.

    sigma2_param is exposed with that name (it holds the VARIANCE, as sampled
    and stored), but internally we keep a second Parameter, _sigma_param, for
    the std-dev-like quantity sqrt(sigma2) that enters the SOC constraint
    linearly. Taking the sqrt at the numpy level (in solve_exact, before
    assigning Parameter .value) rather than as a cp.sqrt(...) expression keeps
    the constraint affine in its Parameters (DPP-compliant), so CVXPY reuses
    the canonicalization across solves instead of rebuilding it every call --
    without this the sqrt-in-CVXPY version was ~6x slower per solve.
    """
    n = N_ITEMS
    x = cp.Variable(n, boolean=True)
    t = cp.Variable(nonneg=True)
    mu_param = cp.Parameter(n, nonneg=True, name="mu")
    sigma2_param = cp.Parameter(n, nonneg=True, name="sigma2")   # variance (reported)
    sigma_param = cp.Parameter(n, nonneg=True, name="sigma")     # sqrt(variance), used below

    constraints = [
        mu_param @ x + RHO * t <= BUDGET,
        cp.SOC(t, cp.multiply(sigma_param, x)),
    ]
    objective = cp.Maximize(VALUES @ x)
    prob = cp.Problem(objective, constraints)
    return prob, x, t, mu_param, sigma2_param, sigma_param


def solve_exact(p, args=None):
    """Solve the robust knapsack MISOCP to certified global optimality.

    Parameters
    ----------
    p : array-like, shape (2*N_ITEMS,)
        Concatenated (mu, sigma^2): p[:N_ITEMS] = mu (ellipsoid centers),
        p[N_ITEMS:] = sigma^2 (ellipsoid per-axis variances).
    args : dict, optional
        'prob', 'x', 't', 'mu_param', 'sigma2_param', 'sigma_param' :
            pre-built CVXPY objects from _build_misocp(), to avoid rebuilding
            the problem every call.
        'solver_opts' : dict of extra kwargs passed to prob.solve() (e.g. a
            Gurobi 'TimeLimit' or 'MIPGap').

    Returns
    -------
    value  : float -- certified global-optimal objective (NaN if infeasible
             or the solver did not certify optimality)
    result : dict with 'x' (0/1 selection vector) and 'status'
    """
    args = args or {}
    n = N_ITEMS
    p = np.asarray(p, dtype=float)
    mu, sigma2 = p[:n], p[n:]

    prob = args.get("prob")
    x, t = args.get("x"), args.get("t")
    mu_param = args.get("mu_param")
    sigma2_param, sigma_param = args.get("sigma2_param"), args.get("sigma_param")
    if prob is None:
        prob, x, t, mu_param, sigma2_param, sigma_param = _build_misocp()

    mu_param.value = mu
    sigma2_param.value = sigma2
    sigma_param.value = np.sqrt(sigma2)

    solver_opts = args.get("solver_opts", {})
    prob.solve(solver=cp.GUROBI, verbose=False, **solver_opts)

    is_optimal = prob.status == cp.OPTIMAL
    value = float(prob.value) if is_optimal else np.nan
    x_val = np.round(x.value).astype(int) if x.value is not None else None

    return value, {"x": x_val, "status": prob.status}
