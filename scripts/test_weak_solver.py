"""Sanity test: can solve_local(weak=True) induce feasible-but-suboptimal AC-OPF
solutions? Compares the tight solve vs. the weakened solve on ~100 test thetas
for a few cases, reporting the induced optimality gap and confirming feasibility.

Usage:  python scripts/test_weak_solver.py [--n 100]
"""
import argparse
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from problems.acopf.network import load_network
from problems.acopf.problem import solve_local

DATA = PROJECT_ROOT / "data" / "acopf-hpc"
# Voltage-bound overrides used at data generation (submit_acopf_pipeline_batch.sh).
VBOUNDS = {"case300": (0.90, 1.10), "case1354pegase": (0.90, 1.10),
           "case2869pegase": (0.90, 1.10)}


def run_case(case, n):
    vmin, vmax = VBOUNDS.get(case, (None, None))
    net, nd = load_network(case, v_min=vmin, v_max=vmax)
    X = np.load(DATA / f"X_test_5000_{case}_seed344.npy")[:n]
    args = {"nd": nd, "net": net, "case_name": case}

    rel_gaps, viols, n_tight, n_weak, n_worse = [], [], 0, 0, 0
    for i, p in enumerate(X):
        c_tight, _ = solve_local(p, args=args)
        c_weak, r_weak = solve_local(p, args=args, weak=True, weak_seed=i)
        if np.isfinite(c_tight):
            n_tight += 1
        if np.isfinite(c_weak):
            n_weak += 1
            viols.append(r_weak.get("max_constr_viol", np.nan))
        if np.isfinite(c_tight) and np.isfinite(c_weak):
            rel = (c_weak - c_tight) / abs(c_tight)
            rel_gaps.append(rel)
            if rel > 1e-4:
                n_worse += 1
    rel_gaps = np.array(rel_gaps)
    viols = np.array(viols)
    print(f"\n=== {case} (n={len(X)}) ===")
    print(f"  tight feasible: {n_tight}/{len(X)}   weak feasible: {n_weak}/{len(X)}")
    print(f"  weak strictly worse (>0.01% gap): {n_worse}/{len(rel_gaps)}  "
          f"({100*n_worse/max(len(rel_gaps),1):.0f}%)")
    if len(rel_gaps):
        print(f"  induced rel. optimality gap (weak-tight)/tight: "
              f"mean={rel_gaps.mean():.3e}  p50={np.median(rel_gaps):.3e}  "
              f"p95={np.percentile(rel_gaps,95):.3e}  max={rel_gaps.max():.3e}")
    if len(viols):
        print(f"  weak max constraint violation (pu): "
              f"p50={np.nanmedian(viols):.2e}  max={np.nanmax(viols):.2e}  "
              f"(feasible if ~1e-8)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--cases", nargs="+",
                    default=["case89pegase", "case300", "case1354pegase"])
    args = ap.parse_args()
    for case in args.cases:
        run_case(case, args.n)


if __name__ == "__main__":
    main()
