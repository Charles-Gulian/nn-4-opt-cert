"""Generate labeled data for the robust 0-1 knapsack problem.

Sampling: each item's mu_i and sigma_i^2 are drawn independently and directly,
uniformly within a range around their FIXED nominal values (MU_NOMINAL,
SIGMA2_NOMINAL from problem.py) -- no shared system-wide scaling factor:

    mu_i       ~ Uniform(0.5 * mu0_i,        1.5 * mu0_i)
    sigma_i^2  ~ Uniform(0.8 * sigma0^2_i,    1.2 * sigma0^2_i)

Each sample is labeled with the robust knapsack's certified global-optimal
value (via Gurobi, see problem.solve_exact) -- there is no relaxation/local
split for this problem, so there is only a single "Cost" column (no "Exact").
"""

import pathlib

import numpy as np
import pandas as pd

from problems.robust_knapsack.problem import (
    N_ITEMS, MU_NOMINAL, SIGMA2_NOMINAL, solve_exact, _build_misocp,
)

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "robust_knapsack"

DEFAULT_MU_RANGE = 0.5       # half-width of the uniform range around mu0_i (0.5x-1.5x)
DEFAULT_SIGMA2_RANGE = 0.2   # half-width of the uniform range around sigma0^2_i (0.8x-1.2x)


def sample_parameters(N, args=None):
    """Sample N instances of theta = (mu, sigma^2), each shape (2*N_ITEMS,).

    Each mu_i, sigma_i^2 is drawn independently, directly uniform within a
    range around its own fixed nominal value -- no shared per-sample scaling
    factor.

    Returns
    -------
    P : np.ndarray, shape (N, 2*N_ITEMS)
        Rows are [mu_0, ..., mu_{n-1}, sigma^2_0, ..., sigma^2_{n-1}].
    """
    args = args or {}
    mu_range = args.get("mu_range", DEFAULT_MU_RANGE)
    sigma2_range = args.get("sigma2_range", DEFAULT_SIGMA2_RANGE)
    rng = np.random.default_rng(args.get("seed", None))

    n = N_ITEMS

    mu_lo = (1 - mu_range) * MU_NOMINAL
    mu_hi = (1 + mu_range) * MU_NOMINAL
    mu = rng.uniform(mu_lo, mu_hi, size=(N, n))

    sigma2_lo = (1 - sigma2_range) * SIGMA2_NOMINAL
    sigma2_hi = (1 + sigma2_range) * SIGMA2_NOMINAL
    sigma2 = rng.uniform(sigma2_lo, sigma2_hi, size=(N, n))

    return np.hstack([mu, sigma2])   # (N, 2*n)


def _col_names(suffix_mu="Mu", suffix_sigma2="Sigma2"):
    """Return column names: [Mu0, ..., Mu_{n-1}, Sigma2_0, ..., Sigma2_{n-1}]."""
    mu_cols = [f"{suffix_mu}{i}" for i in range(N_ITEMS)]
    sigma2_cols = [f"{suffix_sigma2}{i}" for i in range(N_ITEMS)]
    return mu_cols + sigma2_cols


def generate_dataset(n_samples, args=None):
    """Sample n_samples instances and label each with the exact MISOCP optimum."""
    P = sample_parameters(n_samples, args=args)
    feat_cols = _col_names()

    prob, x, t, mu_param, sigma2_param, sigma_param = _build_misocp()
    solve_args = dict(args or {}, prob=prob, x=x, t=t, mu_param=mu_param,
                       sigma2_param=sigma2_param, sigma_param=sigma_param)

    costs = []
    for p in P:
        value, _ = solve_exact(p, args=solve_args)
        costs.append(value)

    df = pd.DataFrame(P, columns=feat_cols)
    df["Cost"] = costs
    return df
