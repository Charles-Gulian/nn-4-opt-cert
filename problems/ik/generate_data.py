"""Generate labeled data for the IK QCQP/SDP experiment.

Parameter vector: p = [xd, yd]  (target end-effector position in 2D)

Sampling: draw (xd, yd) uniformly from a square that covers the full
reachable annulus [|l1-l2|, l1+l2] as well as slightly outside it.

Relaxation selection
--------------------
Pass ``args["relaxation"]`` to choose which convex relaxation labels the data:

    "lass1_SDP"  (default) — order-1 Lasserre / Shor SDP (7×7 moment matrix)
    "lass2_SDP"            — order-2 Lasserre SDP (28×28 moment matrix, tighter)

The relaxation key is also used as a suffix in data file names so that
datasets from different relaxations are kept separate.
"""

import pathlib
import numpy as np
import pandas as pd

from problems.ik.problem import (
    solve_relaxation, solve_lasserre2, solve_local,
    DEFAULT_L1, DEFAULT_L2,
)

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "ik"

_RELAXATION_SOLVERS = {
    "lass1_SDP": solve_relaxation,
    "lass2_SDP": solve_lasserre2,
}


def _get_relax_solver(args):
    key = args.get("relaxation", "lass1_SDP")
    if key not in _RELAXATION_SOLVERS:
        raise ValueError(f"Unknown relaxation '{key}'. Choose from {list(_RELAXATION_SOLVERS)}")
    return _RELAXATION_SOLVERS[key]


def sample_parameters(N, args=None):
    """Sample N target positions (xd, yd).

    Draws uniformly from the square [-r_max, r_max]² where r_max = l1 + l2,
    slightly enlarged to include near-boundary unreachable points.

    Returns
    -------
    P : np.ndarray, shape (N, 2)   columns [xd, yd]
    """
    args = args or {}
    l1 = args.get("l1", DEFAULT_L1)
    l2 = args.get("l2", DEFAULT_L2)
    rng = np.random.default_rng(args.get("seed", None))

    r_max = (l1 + l2) * 1.1
    xd = rng.uniform(-r_max, r_max, size=N)
    yd = rng.uniform(-r_max, r_max, size=N)
    return np.column_stack([xd, yd])


def generate_dataset(n_samples, args=None):
    """Generate training data: target positions + relaxation lower bound.

    Returns a DataFrame with columns [xd, yd, Cost, Exact].
    The relaxation used is controlled by args["relaxation"].
    """
    args = args or {}
    P = sample_parameters(n_samples, args=args)

    solve_relax = _get_relax_solver(args)
    prob_cache = {}
    relax_args = dict(args, prob_cache=prob_cache)

    costs, exact_flags = [], []
    for p in P:
        value, result = solve_relax(p, args=relax_args)
        costs.append(value)
        exact_flags.append(result["exact"])

    df = pd.DataFrame(P, columns=["xd", "yd"])
    df["Cost"]  = costs
    df["Exact"] = exact_flags
    return df


def generate_test_dataset(n_samples, args=None):
    """Like generate_dataset but also runs the local QCQP solver per sample.

    Returns a DataFrame with columns [xd, yd, Cost, Exact, LocalCost].
    """
    args = args or {}
    P = sample_parameters(n_samples, args=args)

    solve_relax = _get_relax_solver(args)
    prob_cache = {}
    relax_args = dict(args, prob_cache=prob_cache)

    costs, exact_flags, local_costs = [], [], []
    for p in P:
        value, result = solve_relax(p, args=relax_args)
        costs.append(value)
        exact_flags.append(result["exact"])

        local_val, _ = solve_local(p, args=args)
        local_costs.append(local_val)

    df = pd.DataFrame(P, columns=["xd", "yd"])
    df["Cost"]      = costs
    df["Exact"]     = exact_flags
    df["LocalCost"] = local_costs
    return df
