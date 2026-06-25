"""MIMO detection problem (from Section B.3 of the paper):

    v(y) = min_{x}  ||y - Hx||^2
    s.t.             x_i^2 = 1,  i = 1, ..., n

x in {-1, +1}^n is the transmitted binary signal, H (m x n complex) is the
fixed channel matrix, and y in R^{2m} is the received signal (real-valued:
real and imaginary parts stacked).

The channel matrix H is fixed and taken directly from Appendix B.3 of the
paper (m=4 receivers, n=2 transmitters). Only the received signal y varies
across problem instances.

The SDP relaxation lifts to an (n+1) x (n+1) PSD matrix with unit diagonal:

    ṽ(y) = min  <M, X>
            s.t.  X_ii = 1,  i = 1, ..., n+1
                  X >= 0

where M = [[Q, c], [c^T, r]] with Q = H^T H, c = -H^T y, r = y^T y
(after the real-valued reformulation). The SDP is exact (rank-1) when the
solution X has rank 1, which is checked and recorded as the `exact` flag.

Standard problem interface:
    solve_relaxation(p, args=None)  -> (value, result)
    solve_local(p, args=None)       -> (value, result)

Parameter sampling lives in generate_data.py.
"""

import numpy as np
import cvxpy as cp

# ---------------------------------------------------------------------------
# Fixed channel matrix -- taken verbatim from Appendix B.3 of the paper
# ---------------------------------------------------------------------------

M_RECEIVERS = 4
N_TRANSMITTERS = 2

# H_bar in C^{m x n} as given in the paper
A_COMPLEX = np.array([
    [ 0.70 + 0.41j,  0.40 + 0.11j],
    [-0.41 + 0.62j, -0.80 + 0.12j],
    [ 0.67 + 0.01j, -0.02 + 0.52j],
    [-0.42 + 0.28j, -0.28 + 0.76j],
])  # (4, 2) complex

# Real-valued representation: A_REAL @ x = [Re(Hx); Im(Hx)]
# so ||Hx - b||^2 = ||A_REAL @ x - y||^2
A_REAL = np.vstack([np.real(A_COMPLEX), np.imag(A_COMPLEX)])  # (2m, n)

# Gram matrix Q = H^T H (fixed since H is fixed)
_Q = A_REAL.T @ A_REAL  # (n, n)

# Zero Forcing pseudo-inverse W = (H^T H)^{-1} H^T
_ZF_W = np.linalg.pinv(A_REAL)  # (n, 2m)

# ---------------------------------------------------------------------------
# SDP relaxation (cvxpy)
# ---------------------------------------------------------------------------

def _M_matrix(y):
    """Build the (n+1) x (n+1) cost matrix M from received signal y in R^{2m}."""
    c = -(A_REAL.T @ y)          # -H^T y
    r = float(y @ y)             # y^T y
    return np.block([
        [_Q,              c.reshape(-1, 1)],
        [c.reshape(1, -1), np.array([[r]])],
    ])


def _build_sdp_problem():
    n = N_TRANSMITTERS
    Z = cp.Variable((n + 1, n + 1), symmetric=True)
    M_param = cp.Parameter((n + 1, n + 1), symmetric=True)
    objective = cp.Minimize(cp.trace(M_param @ Z))
    constraints = [Z >> 0] + [Z[i, i] == 1 for i in range(n + 1)]
    prob = cp.Problem(objective, constraints)
    return prob, M_param, Z


def solve_relaxation(p, args=None):
    """Solve the SDP relaxation for received signal p = y (real 2m-vector).

    Returns (value, result) where value is the SDP relaxation's optimal
    objective (a valid lower bound on ||Hx - y||^2 over x in {-1,+1}^n),
    and result is a dict with:
        - "x_rec": recovered {-1,+1}^n signal (heuristic if not exact)
        - "exact": True if the SDP solution matrix is rank-1

    Pass pre-built cvxpy objects via args["prob"], args["M_param"], args["Z"]
    to avoid rebuilding on every call.
    """
    args = args or {}
    tol = args.get("tol", 1e-6)

    prob = args.get("prob")
    M_param = args.get("M_param")
    Z = args.get("Z")
    if prob is None:
        prob, M_param, Z = _build_sdp_problem()

    M_param.value = _M_matrix(p)
    value = prob.solve()

    eigvals, eigvecs = np.linalg.eigh(Z.value)
    rank = int(np.sum(eigvals > tol * eigvals.max()))
    exact = rank == 1

    # Recover candidate x from the leading eigenvector.
    # The last component corresponds to the homogenizing variable (= 1),
    # so we normalize by it.
    idx = np.argmax(eigvals)
    v = eigvecs[:, idx]
    if abs(v[-1]) > tol:
        v = v / v[-1]
    x_rec = np.sign(v[:-1])
    x_rec[x_rec == 0] = 1

    return value, {"x_rec": x_rec, "exact": exact}


# ---------------------------------------------------------------------------
# Local solver: Zero Forcing (ZF) detection
# ---------------------------------------------------------------------------

def solve_local(p, args=None):
    """Detect x via Zero Forcing: x_hat = sign((H^T H)^{-1} H^T y).

    Applies the pseudo-inverse of A_REAL to the received signal and rounds
    each component to the nearest point in {-1, +1}.

    Returns (value, result) where value = ||H*x_hat - y||^2 and result has:
        - "x": detected {-1,+1}^n signal
        - "x_zf": continuous ZF estimate before rounding
    """
    y = np.asarray(p)
    x_zf = _ZF_W @ y
    x_hat = np.sign(x_zf)
    x_hat[x_hat == 0] = 1

    residual = A_REAL @ x_hat - y
    value = float(residual @ residual)
    return value, {"x": x_hat, "x_zf": x_zf}
