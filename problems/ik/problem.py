"""2-link planar inverse kinematics: QCQP local solver and SDP relaxation.

Problem
-------
Given target end-effector position (xd, yd) and link lengths l1, l2, find
joint angles θ1, θ2 minimizing squared end-effector distance to target.

Using the lifting substitution c_i = cos θ_i, s_i = sin θ_i and the
angle-addition identities for the forward kinematics:

    xe = l1·c1 + l2·(c1·c2 - s1·s2)
    ye = l1·s1 + l2·(s1·c2 + c1·s2)

QCQP (variables: c1, s1, c2, s2, xe, ye)
-----------------------------------------
    min  (xe - xd)² + (ye - yd)²
    s.t. c1² + s1² = 1
         c2² + s2² = 1
         xe = l1·c1 + l2·(c1·c2 - s1·s2)
         ye = l1·s1 + l2·(s1·c2 + c1·s2)

SDP relaxation (Lasserre / moment relaxation)
---------------------------------------------
Lift z = [1, c1, s1, c2, s2, xe, ye] ∈ R^7 to X = zz^T ∈ S^7_+.
Using 0-based indexing (z[0]=1, z[1]=c1, z[2]=s1, z[3]=c2, z[4]=s2,
z[5]=xe, z[6]=ye):

    min   X[5,5] + X[6,6] - 2·xd·X[0,5] - 2·yd·X[0,6]  (+ constants)
    s.t.  X ⪰ 0
          X[0,0] = 1
          X[1,1] + X[2,2] = 1               (c1² + s1² = 1)
          X[3,3] + X[4,4] = 1               (c2² + s2² = 1)
          X[0,5] = l1·X[0,1] + l2·(X[1,3] - X[2,4])   (xe linearised)
          X[0,6] = l1·X[0,2] + l2·(X[2,3] + X[1,4])   (ye linearised)

The objective is linear in X (xe²=X[5,5], ye²=X[6,6], xe=X[0,5],
ye=X[0,6] are all linear functions of the PSD matrix variable).

Ground truth
------------
The reachable workspace is the filled annulus r_min ≤ r ≤ r_max where
r = √(xd²+yd²), r_max = l1+l2, r_min = |l1-l2|.  The closest feasible
end-effector position to any target is the radial projection onto this
annulus, so the true optimal cost is:

    gt(xd, yd) = max(r - r_max, 0)² + max(r_min - r, 0)²

Standard interface
------------------
  ground_truth(p, args)      -> float                  closed-form optimum
  solve_local(p, args)       -> (value, result_dict)   QCQP via Pyomo/IPOPT
  solve_relaxation(p, args)  -> (value, result_dict)   order-1 SDP (Shor/7×7)
  solve_lasserre2(p, args)   -> (value, result_dict)   order-2 Lasserre SDP (28×28)
"""

import pathlib
import shutil
import sys
from collections import defaultdict
from itertools import combinations_with_replacement

import numpy as np
import cvxpy as cp
import pyomo.environ as pyo
from pyomo.environ import SolverFactory

# IPOPT binary: prefer the one co-located with the running Python interpreter
# so we don't accidentally pick up a broken system/base-conda binary.
_PYTHON_BIN_DIR = str(pathlib.Path(sys.executable).parent)
_IPOPT_BIN = shutil.which("ipopt", path=_PYTHON_BIN_DIR) or shutil.which("ipopt")

DEFAULT_L1 = 1.0
DEFAULT_L2 = 0.5


# ── Ground truth ──────────────────────────────────────────────────────────────

def ground_truth(p, args=None):
    """Closed-form optimal cost for the IK problem.

    The reachable workspace is the annulus r_min ≤ r ≤ r_max.  The optimal
    end-effector is the radial projection of the target onto this annulus.

    Parameters
    ----------
    p : array-like, shape (2,)   [xd, yd]
    args : dict, optional  'l1', 'l2'

    Returns
    -------
    float : max(r - r_max, 0)² + max(r_min - r, 0)²
    """
    args = args or {}
    l1 = args.get("l1", DEFAULT_L1)
    l2 = args.get("l2", DEFAULT_L2)
    xd, yd = float(p[0]), float(p[1])
    r = np.sqrt(xd**2 + yd**2)
    r_max = l1 + l2
    r_min = abs(l1 - l2)
    return float(max(r - r_max, 0.0)**2 + max(r_min - r, 0.0)**2)


# ── QCQP local solver (Pyomo + IPOPT) ────────────────────────────────────────

def solve_local(p, args=None):
    """Solve the IK QCQP with IPOPT from a fixed zero initialisation.

    Parameters
    ----------
    p : array-like, shape (2,)   [xd, yd]
    args : dict, optional
        'l1', 'l2' : link lengths (default DEFAULT_L1, DEFAULT_L2)

    Returns
    -------
    value  : float  — optimal (xe-xd)²+(ye-yd)², or np.nan on failure
    result : dict   — 'c1','s1','c2','s2','xe','ye' at the solution
    """
    args = args or {}
    l1 = args.get("l1", DEFAULT_L1)
    l2 = args.get("l2", DEFAULT_L2)
    xd, yd = float(p[0]), float(p[1])

    solver = SolverFactory("ipopt", executable=_IPOPT_BIN)
    solver.options["print_level"] = 0
    solver.options["max_iter"] = 500

    m = pyo.ConcreteModel()
    m.c1 = pyo.Var(initialize=1.0, bounds=(-1, 1))
    m.s1 = pyo.Var(initialize=0.0, bounds=(-1, 1))
    m.c2 = pyo.Var(initialize=1.0, bounds=(-1, 1))
    m.s2 = pyo.Var(initialize=0.0, bounds=(-1, 1))
    m.xe = pyo.Var(initialize=l1 + l2)
    m.ye = pyo.Var(initialize=0.0)

    m.obj   = pyo.Objective(expr=(m.xe - xd)**2 + (m.ye - yd)**2,
                            sense=pyo.minimize)
    m.unit1 = pyo.Constraint(expr=m.c1**2 + m.s1**2 == 1)
    m.unit2 = pyo.Constraint(expr=m.c2**2 + m.s2**2 == 1)
    m.xe_eq = pyo.Constraint(
        expr=m.xe == l1 * m.c1 + l2 * (m.c1 * m.c2 - m.s1 * m.s2))
    m.ye_eq = pyo.Constraint(
        expr=m.ye == l1 * m.s1 + l2 * (m.s1 * m.c2 + m.c1 * m.s2))

    try:
        result = solver.solve(m, tee=False)
        status = str(result.solver.termination_condition)
        if status in ("optimal", "locallyOptimal", "feasible"):
            return float(pyo.value(m.obj)), {
                "success": True,
                "c1": float(pyo.value(m.c1)),
                "s1": float(pyo.value(m.s1)),
                "c2": float(pyo.value(m.c2)),
                "s2": float(pyo.value(m.s2)),
                "xe": float(pyo.value(m.xe)),
                "ye": float(pyo.value(m.ye)),
            }
    except Exception:
        pass

    return np.nan, {"success": False}


# ── SDP relaxation ────────────────────────────────────────────────────────────

def _build_sdp_problem(l1, l2):
    """Build a re-usable cvxpy SDP parameterised by (xd, yd).

    z = [1, c1, s1, c2, s2, xe, ye]  (0-indexed: z[0]=1, ..., z[6]=ye)
    X = zz^T ∈ S^7_+

    Returns (prob, xd_param, yd_param, X)
    """
    X = cp.Variable((7, 7), symmetric=True)

    xd_p = cp.Parameter(name="xd")
    yd_p = cp.Parameter(name="yd")

    constraints = [
        X >> 0,
        X[0, 0] == 1,                                     # z[0] = 1
        X[1, 1] + X[2, 2] == 1,                           # c1² + s1² = 1
        X[3, 3] + X[4, 4] == 1,                           # c2² + s2² = 1
        # xe = l1·c1 + l2·(c1·c2 - s1·s2)
        # X[0,5]=xe, X[0,1]=c1, X[1,3]=c1·c2, X[2,4]=s1·s2
        X[0, 5] == l1 * X[0, 1] + l2 * (X[1, 3] - X[2, 4]),
        # ye = l1·s1 + l2·(s1·c2 + c1·s2)
        # X[0,6]=ye, X[0,2]=s1, X[2,3]=s1·c2, X[1,4]=c1·s2
        X[0, 6] == l1 * X[0, 2] + l2 * (X[2, 3] + X[1, 4]),
    ]

    # Objective: (xe-xd)²+(ye-yd)² = X[5,5]+X[6,6]-2·xd·X[0,5]-2·yd·X[0,6]+xd²+yd²
    # xd², yd² are constants w.r.t. X; we include them so the returned value
    # matches the true squared distance.
    obj = X[5, 5] + X[6, 6] - 2 * xd_p * X[0, 5] - 2 * yd_p * X[0, 6]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    return prob, xd_p, yd_p, X


def solve_relaxation(p, args=None):
    """Solve the IK SDP relaxation.

    Parameters
    ----------
    p : array-like, shape (2,)   [xd, yd]
    args : dict, optional
        'l1', 'l2'    : link lengths (default DEFAULT_L1, DEFAULT_L2)
        'prob_cache'  : dict for re-using built cvxpy problems
        'solver'      : cvxpy solver (default MOSEK)

    Returns
    -------
    value  : float  — SDP lower bound on the optimal cost (+ xd²+yd² constant)
    result : dict
        'exact'  : bool — True if X has rank 1 (relaxation is tight)
        'status' : str
    """
    args = args or {}
    l1 = args.get("l1", DEFAULT_L1)
    l2 = args.get("l2", DEFAULT_L2)
    xd, yd = float(p[0]), float(p[1])
    solver = args.get("solver", cp.MOSEK)

    cache = args.get("prob_cache")
    cache_key = (l1, l2)
    if cache is not None and cache_key in cache:
        prob, xd_p, yd_p, X = cache[cache_key]
    else:
        prob, xd_p, yd_p, X = _build_sdp_problem(l1, l2)
        if cache is not None:
            cache[cache_key] = (prob, xd_p, yd_p, X)

    xd_p.value = xd
    yd_p.value = yd

    try:
        prob.solve(solver=solver, verbose=False)
    except cp.error.SolverError:
        prob.solve(solver=cp.SCS, verbose=False)

    if prob.status in ("optimal", "optimal_inaccurate"):
        # Add back the constant xd²+yd² to recover the true squared distance
        value = float(prob.value) + xd**2 + yd**2
    else:
        value = np.nan

    exact = False
    if X.value is not None:
        eigvals = np.sort(np.linalg.eigvalsh(X.value))[::-1]
        exact = bool(eigvals[0] > 1e-8 and eigvals[1] < 1e-4 * eigvals[0])

    return value, {"exact": exact, "status": prob.status, "X": X.value}


# ── Order-2 Lasserre SDP relaxation ──────────────────────────────────────────

def _generate_monomials(n_vars, max_deg):
    """All multi-index tuples alpha in N^n_vars with |alpha| <= max_deg.

    Returned in graded-lexicographic order (degree 0 first, then degree 1, ...).
    Uses combinations_with_replacement so the ordering within each degree is
    consistent with lexicographic order on variable indices.
    """
    result = []
    for deg in range(max_deg + 1):
        for combo in combinations_with_replacement(range(n_vars), deg):
            alpha = [0] * n_vars
            for i in combo:
                alpha[i] += 1
            result.append(tuple(alpha))
    return result


def _build_lasserre2_problem(l1, l2):
    """Build the order-2 Lasserre moment SDP for the IK QCQP.

    Decision variables: x = (c1, s1, c2, s2, xe, ye), indices 0–5.

    Moment matrix M has shape (N, N), N = C(n+d, d) = C(8, 2) = 28.
    M[i, j] = y_{alpha_i + alpha_j} (moment of monomial x^alpha_i * x^alpha_j).

    Toeplitz constraints enforce M[i,j] = M[k,l] whenever
    basis[i]+basis[j] = basis[k]+basis[l].

    For each equality constraint g = 0 (all degree 2), the order-1 localizing
    condition L_g(y) = 0 adds 28 linear equality constraints on M
    (one per symmetric pair (alpha, beta) with |alpha|, |beta| <= 1).

    l1, l2 enter the constraint matrices as fixed numerical coefficients;
    the problem must be rebuilt when they change (use prob_cache keyed by
    ("lasserre2", l1, l2)).  xd, yd appear only in the objective and are
    handled via cp.Parameter so the problem can be re-solved cheaply.

    Returns (prob, xd_param, yd_param, M)
    """
    N_VARS = 6  # c1, s1, c2, s2, xe, ye
    D = 2       # relaxation order

    basis_2 = _generate_monomials(N_VARS, D)      # 28 monomials, |alpha| <= 2
    basis_1 = _generate_monomials(N_VARS, D - 1)  # 7  monomials, |alpha| <= 1
    N2 = len(basis_2)  # 28

    def add_idx(a, b):
        return tuple(a[k] + b[k] for k in range(N_VARS))

    # Group upper-triangle (i, j) pairs by moment index mu = alpha_i + alpha_j
    moment_groups = defaultdict(list)
    for i in range(N2):
        for j in range(i, N2):
            mu = add_idx(basis_2[i], basis_2[j])
            moment_groups[mu].append((i, j))

    # Canonical representative (i, j) for each moment index mu
    moment_rep = {mu: entries[0] for mu, entries in moment_groups.items()}

    # SDP variable: 28×28 symmetric moment matrix
    M = cp.Variable((N2, N2), symmetric=True)

    constraints = [M >> 0]

    # Toeplitz (moment consistency): all entries sharing the same moment index
    # must be equal.
    for mu, entries in moment_groups.items():
        i0, j0 = entries[0]
        for ik, jk in entries[1:]:
            constraints.append(M[i0, j0] == M[ik, jk])

    # Normalization: y_0 = 1
    zero = tuple([0] * N_VARS)
    i0, j0 = moment_rep[zero]
    constraints.append(M[i0, j0] == 1)

    def m(mu):
        """Return the M entry corresponding to moment y_mu."""
        i, j = moment_rep[mu]
        return M[i, j]

    # Localizing constraints for equality g = 0:
    # For all (alpha, beta) with |alpha|, |beta| <= 1:
    #   sum_gamma  g_gamma * y_{alpha+beta+gamma} = 0
    # Symmetry in (alpha, beta) halves the number of constraints.
    def localizing_zero(g_poly):
        for alpha in basis_1:
            for beta in basis_1:
                if beta >= alpha:
                    ab = add_idx(alpha, beta)
                    expr = sum(
                        coeff * m(add_idx(ab, gamma))
                        for gamma, coeff in g_poly.items()
                    )
                    constraints.append(expr == 0)

    # g1: c1^2 + s1^2 - 1 = 0
    localizing_zero({
        (2, 0, 0, 0, 0, 0):  1.0,
        (0, 2, 0, 0, 0, 0):  1.0,
        (0, 0, 0, 0, 0, 0): -1.0,
    })

    # g2: c2^2 + s2^2 - 1 = 0
    localizing_zero({
        (0, 0, 2, 0, 0, 0):  1.0,
        (0, 0, 0, 2, 0, 0):  1.0,
        (0, 0, 0, 0, 0, 0): -1.0,
    })

    # g3: xe - l1*c1 - l2*(c1*c2 - s1*s2) = 0
    localizing_zero({
        (0, 0, 0, 0, 1, 0):  1.0,
        (1, 0, 0, 0, 0, 0): -l1,
        (1, 0, 1, 0, 0, 0): -l2,
        (0, 1, 0, 1, 0, 0):  l2,
    })

    # g4: ye - l1*s1 - l2*(s1*c2 + c1*s2) = 0
    localizing_zero({
        (0, 0, 0, 0, 0, 1):  1.0,
        (0, 1, 0, 0, 0, 0): -l1,
        (0, 1, 1, 0, 0, 0): -l2,
        (1, 0, 0, 1, 0, 0): -l2,
    })

    # Objective: (xe - xd)^2 + (ye - yd)^2
    # = xe^2 + ye^2 - 2*xd*xe - 2*yd*ye  (+xd^2+yd^2 added back in solve)
    xd_p = cp.Parameter(name="xd_l2")
    yd_p = cp.Parameter(name="yd_l2")

    obj = (m((0, 0, 0, 0, 2, 0)) + m((0, 0, 0, 0, 0, 2))
           - 2 * xd_p * m((0, 0, 0, 0, 1, 0))
           - 2 * yd_p * m((0, 0, 0, 0, 0, 1)))

    prob = cp.Problem(cp.Minimize(obj), constraints)
    return prob, xd_p, yd_p, M


def solve_lasserre2(p, args=None):
    """Solve the order-2 Lasserre SDP relaxation for the IK problem.

    Tighter than solve_relaxation (order-1 / Shor, 7×7 matrix) at the cost of
    a larger SDP: 28×28 moment matrix with Toeplitz + localizing constraints.

    Parameters
    ----------
    p : array-like, shape (2,)   [xd, yd]
    args : dict, optional
        'l1', 'l2'    : link lengths (default DEFAULT_L1, DEFAULT_L2)
        'prob_cache'  : dict; keyed by ("lasserre2", l1, l2)
        'solver'      : cvxpy solver (default MOSEK)

    Returns
    -------
    value  : float  — SDP lower bound (+ xd^2+yd^2 constant added back)
    result : dict   — 'exact' (rank-1 check on M), 'status', 'M'
    """
    args = args or {}
    l1 = args.get("l1", DEFAULT_L1)
    l2 = args.get("l2", DEFAULT_L2)
    xd, yd = float(p[0]), float(p[1])
    solver = args.get("solver", cp.MOSEK)

    cache = args.get("prob_cache")
    cache_key = ("lasserre2", l1, l2)
    if cache is not None and cache_key in cache:
        prob, xd_p, yd_p, M = cache[cache_key]
    else:
        prob, xd_p, yd_p, M = _build_lasserre2_problem(l1, l2)
        if cache is not None:
            cache[cache_key] = (prob, xd_p, yd_p, M)

    xd_p.value = xd
    yd_p.value = yd

    try:
        prob.solve(solver=solver, verbose=False)
    except cp.error.SolverError:
        prob.solve(solver=cp.SCS, verbose=False)

    if prob.status in ("optimal", "optimal_inaccurate"):
        value = float(prob.value) + xd**2 + yd**2
    else:
        value = np.nan

    # Exactness via relaxation gap rather than rank-1 check.  Rank-1 is not
    # the right criterion here: reachable targets have two global optima
    # (elbow-up/down), giving a genuinely rank-2 moment matrix even when the
    # relaxation is tight.  Since this problem has a closed-form ground truth
    # we just check |value - GT| < threshold directly.
    exact = False
    if np.isfinite(value):
        gt = ground_truth([xd, yd], args={"l1": l1, "l2": l2})
        exact = bool(abs(value - gt) < 1e-3)

    return value, {"exact": exact, "status": prob.status, "M": M.value}
