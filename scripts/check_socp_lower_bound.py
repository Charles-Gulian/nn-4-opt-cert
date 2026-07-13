"""Regression check: the SOCP (Jabr) relaxation must be a valid lower bound.

The Jabr power-balance matrices must read the from- and to-bus branch
admittances separately (Y[fr,to] and Y[to,fr]).  For phase-shifting
transformers the Ybus is asymmetric, and reusing the from-bus admittance for
the to-bus injection makes the relaxation solve a different network than the
local solver -- yielding a relaxation value that can *exceed* the true optimum
(v_r > f), which silently corrupts every downstream certification.

This script guards two properties on a phase-shifter case (case89pegase) and a
symmetric control (case9):
  1. v_r(theta) <= f(theta) + tol on real sampled instances (valid lower bound).
  2. The to-bus admittance actually differs from the from-bus admittance on the
     phase-shifter case (so the check is exercising the asymmetric path).

Run:  python scripts/check_socp_lower_bound.py
Exit code is non-zero if the invariant is violated, so it can gate CI / a
pre-regeneration sanity step.
"""
import sys
import pathlib
import warnings

warnings.filterwarnings("ignore")
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from problems.acopf.network import load_network
from problems.acopf.problem import solve_relaxation, solve_local
from problems.acopf.generate_data import (
    sample_parameters, DEFAULT_ALPHA_MIN, DEFAULT_ALPHA_MAX, DEFAULT_ETA_RANGE,
)

# Relative tolerance on the lower-bound check.  The relaxation and the local
# solver each converge to ~1e-4..1e-6; we only flag violations well above that
# noise floor, so a genuine formulation error (the phase-shifter bug produced
# ~1% violations) is caught while solver tolerance is not.
REL_TOL = 1e-3


def check_case(case, n_samples=25, expect_asymmetry=False):
    net, nd = load_network(case)
    args = {"nd": nd, "net": net, "case_name": case,
            "relaxation": "socp", "prob_cache": {}}

    fr, to = nd.branch_from, nd.branch_to
    asym = int((nd.Y[fr, to] != nd.Y[to, fr]).sum())
    if expect_asymmetry and asym == 0:
        print(f"  {case}: expected phase-shifter branches but found none")
        return False

    # Use the SAME parameter sampler as data generation: it scales the whole
    # system by a random alpha and adds independent per-load noise.  A uniform
    # scale on every load leaves the phase-shifter branches non-binding and
    # hides the bug, so we must reproduce the generation distribution here.
    sample_args = {
        "nd": nd, "case_name": case,
        "alpha_min": DEFAULT_ALPHA_MIN, "alpha_max": DEFAULT_ALPHA_MAX,
        "eta_range": DEFAULT_ETA_RANGE, "seed": 0,
    }
    X = sample_parameters(n_samples, args=sample_args)

    worst = 0.0
    n_ok = 0
    for p in X:
        v_r, _ = solve_relaxation(p, args=args)
        f, _ = solve_local(p, args=args)
        if not (np.isfinite(v_r) and np.isfinite(f)):
            continue
        n_ok += 1
        rel = (v_r - f) / abs(f)          # >0 means invalid lower bound
        worst = max(worst, rel)
    ok = worst <= REL_TOL
    print(f"  {case:16s} asym_branches={asym:3d}  n_ok={n_ok:3d}  "
          f"worst (v_r-f)/f = {worst:+.3e}  -> {'PASS' if ok else 'FAIL'}")
    return ok and n_ok > 0


def main():
    print("SOCP lower-bound regression check:")
    results = [
        check_case("case89pegase", expect_asymmetry=True),
        check_case("case9"),
    ]
    if all(results):
        print("OK: SOCP relaxation is a valid lower bound on all checked cases.")
        sys.exit(0)
    print("FAIL: SOCP relaxation violated the lower-bound property.")
    sys.exit(1)


if __name__ == "__main__":
    main()
