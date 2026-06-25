"""Generate labeled training / test data for the AC-OPF problem.

Sampling follows the PGLearn methodology (two-level uniform perturbation):

    p^d = α × η × p^d_ref
    q^d = α × η × q^d_ref

  α  — system-wide scaling factor, one per sample, Uniform(α_min, α_max).
       Default range 0.7–1.1 keeps AC-OPF feasible across operating conditions.
  η  — bus-level noise vector, one per load per sample, Uniform(1-η_range, 1+η_range).
       Default ±15% around each load's reference value.

Both α and η are independent draws; the same η is applied to P and Q at each bus.
"""

import pathlib

import numpy as np
import pandas as pd

from problems.acopf.network import NetworkData, load_network, DEFAULT_CASE
from problems.acopf.problem import solve_relaxation, solve_local

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "acopf"

DEFAULT_CASE_NAME  = DEFAULT_CASE
DEFAULT_ALPHA_MIN  = 0.6    # lower bound on the system-wide scaling factor α
DEFAULT_ALPHA_MAX  = 1.2    # upper bound on α
DEFAULT_ETA_RANGE  = 0.3   # half-width of the per-load uniform noise


def sample_parameters(N, args=None):
    """Sample N demand vectors using the ICNN paper's two-level perturbation scheme.

    Parameters
    ----------
    N    : int
    args : dict, optional
        'nd'         : pre-built NetworkData (avoids reloading the network)
        'case_name'  : pandapower case name (default: 'case9')
        'alpha_min'  : lower bound of the uniform α range (default 0.7)
        'alpha_max'  : upper bound of the uniform α range (default 1.1)
        'eta_range'  : half-width of the per-load uniform noise η (default 0.15)
        'seed'       : integer RNG seed

    Returns
    -------
    P : np.ndarray, shape (N, 2 * n_loads)
        Rows are [Pd_0, ..., Pd_{K-1}, Qd_0, ..., Qd_{K-1}] in MW / MVar.
    """
    args = args or {}
    nd = args.get("nd")
    if nd is None:
        _, nd = load_network(args.get("case_name", DEFAULT_CASE_NAME))

    alpha_min = args.get("alpha_min", DEFAULT_ALPHA_MIN)
    alpha_max = args.get("alpha_max", DEFAULT_ALPHA_MAX)
    eta_range = args.get("eta_range", DEFAULT_ETA_RANGE)
    rng       = np.random.default_rng(args.get("seed", None))

    # System-wide scaling factor α ~ Uniform(α_min, α_max), one per sample.
    alpha = rng.uniform(alpha_min, alpha_max, size=(N, 1))            # (N, 1)

    # Bus-level noise η ~ Uniform(1 - η_range, 1 + η_range), one per load per sample.
    eta = rng.uniform(1 - eta_range, 1 + eta_range, size=(N, nd.n_loads))  # (N, n_loads)

    # Same η draw applied to both P and Q at each bus.
    Pd = alpha * eta * nd.pd_nominal[np.newaxis, :]   # (N, n_loads) [MW]
    Qd = alpha * eta * nd.qd_nominal[np.newaxis, :]   # (N, n_loads) [MVar]

    return np.hstack([Pd, Qd])                         # (N, 2*n_loads)


def _col_names(nd: NetworkData, suffix_p="Pd", suffix_q="Qd"):
    """Return column names: [Pd0, Pd1, ..., Qd0, Qd1, ...]."""
    p_cols = [f"{suffix_p}{i}" for i in range(nd.n_loads)]
    q_cols = [f"{suffix_q}{i}" for i in range(nd.n_loads)]
    return p_cols + q_cols


def generate_dataset(n_samples, args=None):
    """Generate training data: demand parameters + relaxation lower bound.

    Returns a DataFrame with columns [Pd0, ..., Qd0, ..., Cost, Exact].
    """
    args = args or {}
    nd = args.get("nd")
    net = args.get("net")
    if nd is None:
        net, nd = load_network(args.get("case_name", DEFAULT_CASE_NAME))
        args = dict(args, nd=nd, net=net)

    relaxation = args.get("relaxation", "socp")
    P = sample_parameters(n_samples, args=args)

    # Pre-build and cache the cvxpy problem to avoid rebuilding per sample.
    prob_cache = {}
    relax_args = dict(args, prob_cache=prob_cache, relaxation=relaxation)

    costs, exact_flags = [], []
    for p in P:
        value, result = solve_relaxation(p, args=relax_args)
        costs.append(value)
        exact_flags.append(result["exact"])

    cols = _col_names(nd)
    df = pd.DataFrame(P, columns=cols)
    df["Cost"]  = costs
    df["Exact"] = exact_flags
    return df


def generate_test_dataset(n_samples, args=None):
    """Like generate_dataset, but also runs the local (IPOPT) solver per sample.

    Returns a DataFrame with columns [Pd0, ..., Qd0, ..., Cost, Exact, LocalCost].
    Used by certify_acopf.py.
    """
    args = args or {}
    nd = args.get("nd")
    net = args.get("net")
    if nd is None:
        net, nd = load_network(args.get("case_name", DEFAULT_CASE_NAME))
        args = dict(args, nd=nd, net=net)

    P = sample_parameters(n_samples, args=args)

    prob_cache = {}
    relax_args = dict(args, prob_cache=prob_cache)

    costs, exact_flags, local_costs = [], [], []
    for p in P:
        value, result = solve_relaxation(p, args=relax_args)
        costs.append(value)
        exact_flags.append(result["exact"])

        local_value, _ = solve_local(p, args=args)
        local_costs.append(local_value)

    cols = _col_names(nd)
    df = pd.DataFrame(P, columns=cols)
    df["Cost"]      = costs
    df["Exact"]     = exact_flags
    df["LocalCost"] = local_costs
    return df
