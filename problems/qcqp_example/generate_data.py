"""Generate labeled data for the QCQP example problem.

Samples parameters p = (a, b) and labels each with the SDP relaxation's
optimal value. This value is always a valid lower bound on the true optimum;
it is exact (equal to the true optimum) when the relaxation's solution is
rank-1, which is recorded in the `Exact` column. Samples are kept regardless
of exactness -- a lower bound is still a useful training signal.
"""

import pathlib

import numpy as np
import pandas as pd

from problems.qcqp_example.problem import solve_relaxation, solve_local, _build_sdp_problem

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "qcqp_example"

DEFAULT_BOUNDS = (-5, 5)


def sample_parameters(N, args=None):
    """Sample N parameter vectors p = (a, b) ~ Uniform(bounds)^2."""
    args = args or {}
    bounds = args.get("bounds", DEFAULT_BOUNDS)
    seed = args.get("seed", None)
    rng = np.random.default_rng(seed)
    return rng.uniform(bounds[0], bounds[1], size=(N, 2))


def generate_dataset(n_samples, args=None):
    P = sample_parameters(n_samples, args=args)

    prob, M0, X = _build_sdp_problem()
    relax_args = dict(args or {}, prob=prob, M0=M0, X=X)

    costs, exact = [], []
    for p in P:
        value, result = solve_relaxation(p, args=relax_args)
        costs.append(value)
        exact.append(result["exact"])

    return pd.DataFrame({"a": P[:, 0], "b": P[:, 1], "Cost": costs, "Exact": exact})


def generate_test_dataset(n_samples, args=None):
    """Like `generate_dataset`, but also solves each sample locally (IPOPT),
    for use in certifying the local solver's optimality against the
    relaxation and a trained NN's prediction.
    """
    P = sample_parameters(n_samples, args=args)

    prob, M0, X = _build_sdp_problem()
    relax_args = dict(args or {}, prob=prob, M0=M0, X=X)

    costs, exact, local_costs = [], [], []
    for p in P:
        value, result = solve_relaxation(p, args=relax_args)
        costs.append(value)
        exact.append(result["exact"])

        local_value, _ = solve_local(p, args=args)
        local_costs.append(local_value)

    return pd.DataFrame({
        "a": P[:, 0], "b": P[:, 1],
        "Cost": costs, "Exact": exact,
        "LocalCost": local_costs,
    })
