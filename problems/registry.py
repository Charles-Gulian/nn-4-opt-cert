"""Problem registry for the unified per-fold certification workflow.

Every experiment in the paper shares the same abstract interface:

    sample_parameters(N, args)      -> (N, d) array of instance parameters
    solve_relax(p, relax_args)      -> (value, result)  value = relaxation lower bound v(theta)
    solve_local(p, args)            -> (value, result)  value = feasible upper bound f(theta)

but each problem sets up its convex relaxation slightly differently (a cvxpy
problem object + auxiliary handles that are cheap to reuse across many solves).
`build_relax_args` encapsulates that per-problem setup so the generic driver
scripts (generate_dataset.py / train_generic.py / evaluate_certify.py) never
have to special-case a problem.

Naming/layout is uniform across the small problems:
    data/<key>/train_{N}.csv , data/<key>/test_{N}.csv
    models/<key>/dnn_<key>_n{N}_fold{f}.pt
    results/<key>/...

AC-OPF is the exception: its data + fold checkpoints already exist under the
SAVIO layout (data/acopf-hpc/, models/acopf/), so its registry entry only needs
the paths + checkpoint pattern used by the certification step -- generation and
training are NOT repeated for AC-OPF.
"""

import pathlib
from dataclasses import dataclass, field
from typing import Callable

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Columns that are labels, not features (shared by every problem's CSVs).
# Cost = relaxation lower bound v; LocalCost = local upper bound f; Exact = tightness flag.
LABEL_COLS = ("Cost", "Exact", "LocalCost")


@dataclass
class ProblemSpec:
    """Everything the generic driver scripts need to run one config end-to-end.

    For AC-OPF, the solver callables are unused (no regen/retrain) and left None;
    only the path/pattern fields are consulted by evaluate_certify.py.
    """
    key: str
    feature_cols: list
    # Solver interface (None for acopf-style reuse-only configs).
    sample_parameters: Callable = None
    solve_relax: Callable = None
    solve_local: Callable = None
    build_relax_args: Callable = None
    default_args: dict = field(default_factory=dict)
    # Where data / models / results live for this config.
    train_csv: pathlib.Path = None
    test_csv: pathlib.Path = None
    models_dir: pathlib.Path = None
    ckpt_pattern: str = None          # str with a "{fold}" placeholder
    results_dir: pathlib.Path = None
    can_generate: bool = True         # False for acopf (reuse existing artifacts)


# ── small problems: lazy builders so importing the registry doesn't force a ───
#    cvxpy/pyomo import unless a solver is actually needed (e.g. pure eval runs).

def _qcqp_spec(n_train=20_000, n_test=5_000):
    from problems.qcqp_example import problem as P

    def build_relax_args(args):
        prob, M0, X = P._build_sdp_problem()
        return dict(args or {}, prob=prob, M0=M0, X=X)

    from problems.qcqp_example.generate_data import sample_parameters
    return ProblemSpec(
        key="qcqp",
        feature_cols=["a", "b"],
        sample_parameters=sample_parameters,
        solve_relax=P.solve_relaxation,
        solve_local=P.solve_local,
        build_relax_args=build_relax_args,
        train_csv=PROJECT_ROOT / "data" / "qcqp" / f"train_{n_train}.csv",
        test_csv=PROJECT_ROOT / "data" / "qcqp" / f"test_{n_test}.csv",
        models_dir=PROJECT_ROOT / "models" / "qcqp",
        ckpt_pattern=f"dnn_qcqp_n{n_train}_fold{{fold}}.pt",
        results_dir=PROJECT_ROOT / "results" / "qcqp",
    )


def _mimo_spec(n_train=20_000, n_test=5_000):
    from problems.mimo_detection import problem as P
    from problems.mimo_detection.generate_data import sample_parameters, _B_COLS

    def build_relax_args(args):
        prob, M_param, Z = P._build_sdp_problem()
        return dict(args or {}, prob=prob, M_param=M_param, Z=Z)

    return ProblemSpec(
        key="mimo",
        feature_cols=list(_B_COLS),
        sample_parameters=sample_parameters,
        solve_relax=P.solve_relaxation,
        solve_local=P.solve_local,      # accepts (p, args=None) despite generate calling solve_local(p)
        build_relax_args=build_relax_args,
        train_csv=PROJECT_ROOT / "data" / "mimo" / f"train_{n_train}.csv",
        test_csv=PROJECT_ROOT / "data" / "mimo" / f"test_{n_test}.csv",
        models_dir=PROJECT_ROOT / "models" / "mimo",
        ckpt_pattern=f"dnn_mimo_n{n_train}_fold{{fold}}.pt",
        results_dir=PROJECT_ROOT / "results" / "mimo",
    )


def _mimo_large_spec(n_train=20_000, n_test=5_000):
    """Large-scale (m=64, n=32) MIMO detection, kind="self" for certification
    (see problems/mimo_detection_large/problem.py docstring): exact ML ground
    truth is intractable at n=32, so the SDP relaxation value is the ground
    truth used downstream, not an independent exact solve."""
    from problems.mimo_detection_large import problem as P
    from problems.mimo_detection_large.generate_data import sample_parameters, _B_COLS

    def build_relax_args(args):
        prob, M_param, Z = P._build_sdp_problem()
        return dict(args or {}, prob=prob, M_param=M_param, Z=Z)

    return ProblemSpec(
        key="mimo_large",
        feature_cols=list(_B_COLS),
        sample_parameters=sample_parameters,
        solve_relax=P.solve_relaxation,
        solve_local=P.solve_local,
        build_relax_args=build_relax_args,
        train_csv=PROJECT_ROOT / "data" / "mimo_detection_large" / f"train_{n_train}.csv",
        test_csv=PROJECT_ROOT / "data" / "mimo_detection_large" / f"test_{n_test}.csv",
        models_dir=PROJECT_ROOT / "models" / "mimo_large",
        ckpt_pattern=f"dnn_mimo_large_n{n_train}_fold{{fold}}.pt",
        results_dir=PROJECT_ROOT / "results" / "mimo_large",
    )


def _ik_spec(order, n_train=20_000, n_test=5_000):
    """order in {1, 2} selects the Lasserre relaxation level."""
    from problems.ik import problem as P
    from problems.ik.generate_data import sample_parameters

    key = f"ik_lass{order}"
    relax_key = "lass1_SDP" if order == 1 else "lass2_SDP"
    solve_relax = P.solve_relaxation if order == 1 else P.solve_lasserre2

    def build_relax_args(args):
        # IK caches the built SDP keyed on (l1, l2); a fresh per-worker cache is
        # correct and lets the generator parallelize across processes.
        return dict(args or {}, prob_cache={})

    return ProblemSpec(
        key=key,
        feature_cols=["xd", "yd"],
        sample_parameters=sample_parameters,
        solve_relax=solve_relax,
        solve_local=P.solve_local,
        build_relax_args=build_relax_args,
        default_args={"l1": P.DEFAULT_L1, "l2": P.DEFAULT_L2, "relaxation": relax_key},
        train_csv=PROJECT_ROOT / "data" / key / f"train_{n_train}.csv",
        test_csv=PROJECT_ROOT / "data" / key / f"test_{n_test}.csv",
        models_dir=PROJECT_ROOT / "models" / key,
        ckpt_pattern=f"dnn_{key}_n{n_train}_fold{{fold}}.pt",
        results_dir=PROJECT_ROOT / "results" / key,
    )


def _acopf_data_dir():
    """AC-OPF data lives under a different directory name depending on where
    we are: SAVIO writes it to data/acopf/ (see generate_acopf_data_parallel.py
    / submit_acopf_train.sh), while the local laptop pull-back convention is
    data/acopf-hpc/ (see scripts/train_acopf.py's DEFAULT_DATA_DIR). Prefer
    whichever actually contains the data so this registry works unmodified in
    both places instead of hardcoding one location.
    """
    on_savio = PROJECT_ROOT / "data" / "acopf"
    on_laptop = PROJECT_ROOT / "data" / "acopf-hpc"
    return on_savio if on_savio.exists() else on_laptop


def _acopf_spec(case, relax, n_train=20_000, n_test=5_000):
    """Reuse-only entry: AC-OPF data + fold checkpoints already exist on SAVIO.

    Feature columns are inferred from the CSV header at eval time (they vary by
    case), so feature_cols is left empty here.
    """
    key = f"acopf_{relax}_{case}"
    data_dir = _acopf_data_dir()
    return ProblemSpec(
        key=key,
        feature_cols=[],   # inferred from CSV header (all cols minus LABEL_COLS)
        can_generate=False,
        train_csv=data_dir / f"train_{n_train}_{relax}_{case}.csv",
        test_csv=data_dir / f"test_{n_test}_{relax}_{case}.csv",
        models_dir=PROJECT_ROOT / "models" / "acopf",
        ckpt_pattern=f"dnn_{relax}_{case}_n{n_train}_fold{{fold}}.pt",
        results_dir=PROJECT_ROOT / "results" / "acopf-cert" / key,
    )


# Small-problem builders keyed by config name; called lazily via get_spec().
_BUILDERS = {
    "qcqp":       _qcqp_spec,
    "mimo":       _mimo_spec,
    "mimo_large": _mimo_large_spec,
    "ik_lass1":   lambda **kw: _ik_spec(1, **kw),
    "ik_lass2":   lambda **kw: _ik_spec(2, **kw),
}

SMALL_PROBLEM_KEYS = list(_BUILDERS)

ACOPF_CASES = ["case9", "case14", "case39", "case89pegase",
               "case118", "case300", "case1354pegase", "case2869pegase"]
ACOPF_RELAX = ["socp", "chordal_sdp"]


def get_spec(key, n_train=20_000, n_test=5_000):
    """Return the ProblemSpec for a config key.

    Small problems: "qcqp", "mimo", "ik_lass1", "ik_lass2".
    AC-OPF: "acopf_<relax>_<case>", e.g. "acopf_socp_case9".
    """
    if key in _BUILDERS:
        return _BUILDERS[key](n_train=n_train, n_test=n_test)
    if key.startswith("acopf_"):
        _, relax, case = _split_acopf(key)
        return _acopf_spec(case, relax, n_train=n_train, n_test=n_test)
    raise KeyError(f"Unknown problem key '{key}'. "
                   f"Known small keys: {SMALL_PROBLEM_KEYS}; or acopf_<relax>_<case>.")


def _split_acopf(key):
    """Split 'acopf_<relax>_<case>' where <relax> may itself contain underscores
    (e.g. 'chordal_sdp'). Match the relaxation against the known set."""
    rest = key[len("acopf_"):]
    for relax in ACOPF_RELAX:
        if rest.startswith(relax + "_"):
            return "acopf", relax, rest[len(relax) + 1:]
    raise KeyError(f"Cannot parse AC-OPF key '{key}'.")


def acopf_keys():
    """All AC-OPF config keys, skipping the intractable case2869pegase/chordal_sdp."""
    keys = []
    for case in ACOPF_CASES:
        for relax in ACOPF_RELAX:
            if case == "case2869pegase" and relax == "chordal_sdp":
                continue
            keys.append(f"acopf_{relax}_{case}")
    return keys
