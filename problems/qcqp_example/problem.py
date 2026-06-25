"""QCQP example problem:

    min_{x,y}  (x - a)^2 + (y - b)^2
    s.t.       x y >= 1

Parameterized by p = (a, b). The feasible set {x y >= 1} is non-convex, but
the SDP relaxation below is exact (guaranteed rank-1 by the S-lemma) whenever
the relaxation's solution matrix is rank-1.

Standard problem interface:
    solve_relaxation(p, args=None)  -> (value, result)
    solve_local(p, args=None)       -> (value, result)

Parameter sampling lives in generate_data.py.
"""

import numpy as np
import cvxpy as cp
import pyomo.environ as pyo


# ---------------------------------------------------------------------------
# SDP relaxation (cvxpy) -- value is always a valid lower bound; exact when
# the relaxation solution is rank-1
# ---------------------------------------------------------------------------

_M1 = np.array([
    [0, 0, 0],
    [0, 0, -0.5],
    [0, -0.5, 0],
])


def _M0(a, b):
    return np.array([
        [a ** 2 + b ** 2, -a, -b],
        [-a, 1, 0],
        [-b, 0, 1],
    ])


def _build_sdp_problem():
    X = cp.Variable((3, 3), symmetric=True)
    M0 = cp.Parameter((3, 3))
    objective = cp.Minimize(cp.trace(M0 @ X))
    constraints = [cp.trace(_M1 @ X) <= -1, X[0, 0] == 1, X >> 0]
    prob = cp.Problem(objective, constraints)
    return prob, M0, X


def solve_relaxation(p, args=None):
    """Solve the SDP relaxation for parameter p = (a, b).

    Returns (value, result), where `value` is the relaxation's optimal
    objective value (always a valid lower bound on the true optimum, and
    exact when `result["exact"]` is True), and `result` is a dict with:
        - "x", "y": recovered solution (heuristic if not exact)
        - "exact": whether the relaxation solution matrix is rank-1

    `args` may provide pre-built cvxpy objects under "prob"/"M0"/"X" (from
    `_build_sdp_problem`) to avoid rebuilding the problem on every call, and
    a rank tolerance under "tol" (default 1e-6).
    """
    args = args or {}
    a, b = p
    tol = args.get("tol", 1e-6)

    prob, M0, X = args.get("prob"), args.get("M0"), args.get("X")
    if prob is None:
        prob, M0, X = _build_sdp_problem()

    M0.value = _M0(a, b)
    value = prob.solve()

    eigvals, eigvecs = np.linalg.eigh(X.value)
    rank = np.sum(eigvals > tol * eigvals.max())
    exact = bool(rank == 1)

    idx = np.argmax(eigvals)
    v = eigvecs[:, idx]
    v = v / v[0]  # normalize so v[0] == X[0,0]**0.5 == 1
    x_rec, y_rec = v[1], v[2]

    return value, {"x": x_rec, "y": y_rec, "exact": exact}


# ---------------------------------------------------------------------------
# Local solve (Pyomo / IPOPT) -- finds *a* local optimum, not necessarily global
# ---------------------------------------------------------------------------

def solve_local(p, args=None):
    """Solve the problem locally via IPOPT from an initial point.

    Returns (value, result), where `result` has keys "x", "y".
    `args` may provide "x0", "y0" initial values (default 3.0, 1.75).
    """
    args = args or {}
    a, b = p
    x0 = args.get("x0", 3.0)
    y0 = args.get("y0", 1.75)

    model = pyo.ConcreteModel()

    model.x = pyo.Var(initialize=x0)
    model.y = pyo.Var(initialize=y0)
    model.a = pyo.Param(initialize=a, mutable=True)
    model.b = pyo.Param(initialize=b, mutable=True)

    def obj(m):
        return (m.x - m.a) ** 2 + (m.y - m.b) ** 2
    model.obj = pyo.Objective(rule=obj, sense=pyo.minimize)

    def constr(m):
        return m.x * m.y >= 1
    model.constr = pyo.Constraint(rule=constr)

    solver = pyo.SolverFactory("ipopt")
    solver.solve(model, tee=False)

    value = pyo.value(obj(model))
    return value, {"x": pyo.value(model.x), "y": pyo.value(model.y)}
