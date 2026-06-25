"""Generate labeled data for the MIMO detection problem."""

import pathlib

import numpy as np
import pandas as pd

from problems.mimo_detection.problem import (
    solve_relaxation, solve_local,
    _build_sdp_problem, A_COMPLEX, M_RECEIVERS, N_TRANSMITTERS,
)

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "mimo_detection"

_B_COLS = [f"b{i}" for i in range(2 * M_RECEIVERS)]

# Noise variance from the paper: eps ~ N(0, sigma^2 * I) with sigma^2 = 0.1
DEFAULT_SIGMA_SQ = 0.1


def sample_parameters(N, args=None):
    """Sample N received signal vectors y (real 2m-vectors).

    Each sample draws x uniformly from {-1,+1}^n, transmits through H, and
    adds complex Gaussian noise eps ~ N(0, sigma_sq * I_m):
        y_bar = H x + eps,   y = [Re(y_bar); Im(y_bar)] in R^{2m}.

    args keys:
        "sigma_sq": noise variance (default 0.1, matching the paper)
        "seed": RNG seed
    """
    args = args or {}
    sigma_sq = args.get("sigma_sq", DEFAULT_SIGMA_SQ)
    sigma = np.sqrt(sigma_sq)
    seed = args.get("seed", None)
    rng = np.random.default_rng(seed)

    m, n = M_RECEIVERS, N_TRANSMITTERS
    X = 2 * rng.integers(0, 2, size=(N, n)) - 1          # uniform {-1,+1}^n
    B_complex = (A_COMPLEX @ X.T).T                       # (N, m) noiseless
    noise = sigma * (rng.standard_normal((N, m)) + 1j * rng.standard_normal((N, m)))
    B_complex = B_complex + noise

    return np.hstack([np.real(B_complex), np.imag(B_complex)])  # (N, 2m)


def generate_dataset(n_samples, args=None):
    """Sample received signals and label each with the SDP relaxation value."""
    P = sample_parameters(n_samples, args=args)

    prob, M_param, Z = _build_sdp_problem()
    relax_args = dict(args or {}, prob=prob, M_param=M_param, Z=Z)

    costs, exact = [], []
    for p in P:
        value, result = solve_relaxation(p, args=relax_args)
        costs.append(value)
        exact.append(result["exact"])

    df = pd.DataFrame(P, columns=_B_COLS)
    df["Cost"] = costs
    df["Exact"] = exact
    return df


def generate_test_dataset(n_samples, args=None):
    """Like generate_dataset, but also runs the ZF local solver on each sample."""
    P = sample_parameters(n_samples, args=args)

    prob, M_param, Z = _build_sdp_problem()
    relax_args = dict(args or {}, prob=prob, M_param=M_param, Z=Z)

    costs, exact, local_costs = [], [], []
    for p in P:
        value, result = solve_relaxation(p, args=relax_args)
        costs.append(value)
        exact.append(result["exact"])

        local_value, _ = solve_local(p)
        local_costs.append(local_value)

    df = pd.DataFrame(P, columns=_B_COLS)
    df["Cost"] = costs
    df["Exact"] = exact
    df["LocalCost"] = local_costs
    return df
