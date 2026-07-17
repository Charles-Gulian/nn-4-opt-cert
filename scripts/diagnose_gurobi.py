"""Diagnose whether Gurobi is usable in the current environment/node.

MIMO detection (no Gurobi dependency) succeeded at n=640000 on SAVIO while the
robust-knapsack run (Gurobi MISOCP) silently produced all-NaN labels. This
script isolates *why*: it checks the gurobipy import, license, relevant env
vars, and does a real solve through both gurobipy directly and through the
project's own solve_exact() (via cvxpy), so we can see exactly which layer
fails before resubmitting the 640k knapsack job.

Usage (run on a SAVIO compute node via srun/sbatch, in the same env/module
setup as the actual data-gen job -- NOT the login node):
    python scripts/diagnose_gurobi.py
"""

import os
import pathlib
import sys
import traceback

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}", flush=True)


def check_env_vars():
    _section("1. Relevant environment variables")
    for var in ["GUROBI_HOME", "GRB_LICENSE_FILE", "LD_LIBRARY_PATH",
                "PATH", "CONDA_DEFAULT_ENV", "SLURM_JOB_NODELIST", "SLURMD_NODENAME"]:
        val = os.environ.get(var)
        print(f"  {var} = {val!r}", flush=True)


def check_gurobipy_import():
    _section("2. gurobipy import")
    try:
        import gurobipy as gp
        print(f"  OK: gurobipy imported from {gp.__file__}", flush=True)
        print(f"  gurobipy version: {gp.gurobi.version()}", flush=True)
        return gp
    except Exception:
        print("  FAILED to import gurobipy:", flush=True)
        traceback.print_exc()
        return None


def check_gurobi_env(gp):
    _section("3. Gurobi environment / license acquisition")
    if gp is None:
        print("  Skipped (import failed).", flush=True)
        return False
    try:
        env = gp.Env()
        print("  OK: acquired a Gurobi Env (license check passed).", flush=True)
        env.dispose()
        return True
    except Exception:
        print("  FAILED to acquire a Gurobi Env / license:", flush=True)
        traceback.print_exc()
        return False


def check_tiny_gurobipy_solve(gp):
    _section("4. Tiny direct gurobipy solve (bypassing cvxpy)")
    if gp is None:
        print("  Skipped (import failed).", flush=True)
        return False
    try:
        m = gp.Model("diag")
        m.setParam("OutputFlag", 0)
        x = m.addVar(vtype=gp.GRB.BINARY, name="x")
        y = m.addVar(vtype=gp.GRB.BINARY, name="y")
        m.setObjective(x + y, gp.GRB.MAXIMIZE)
        m.addConstr(x + y <= 1)
        m.optimize()
        print(f"  OK: status={m.Status}, objVal={m.ObjVal if m.SolCount else None}", flush=True)
        return m.Status == gp.GRB.OPTIMAL
    except Exception:
        print("  FAILED tiny gurobipy solve:", flush=True)
        traceback.print_exc()
        return False


def check_cvxpy_gurobi():
    _section("5. cvxpy solver registration (GUROBI visible to cvxpy?)")
    try:
        import cvxpy as cp
        installed = cp.installed_solvers()
        print(f"  cvxpy installed_solvers(): {installed}", flush=True)
        if "GUROBI" not in installed:
            print("  WARNING: GUROBI not in cvxpy's installed_solvers() list.", flush=True)
        return "GUROBI" in installed
    except Exception:
        print("  FAILED to import cvxpy:", flush=True)
        traceback.print_exc()
        return False


def check_solve_exact():
    _section("6. Real project call: problems/robust_knapsack solve_exact()")
    try:
        import numpy as np
        from problems.robust_knapsack.problem import _build_misocp, solve_exact
        from problems.robust_knapsack.generate_data import sample_parameters

        X = sample_parameters(2, args={"seed": 343})
        prob, x, t, mu_param, sigma2_param, sigma_param = _build_misocp()
        args = {"prob": prob, "x": x, "t": t, "mu_param": mu_param,
                "sigma2_param": sigma2_param, "sigma_param": sigma_param}

        for i in range(2):
            val, info = solve_exact(X[i], args=args)
            print(f"  instance {i}: value={val}, status={info['status']}", flush=True)
        return True
    except Exception:
        print("  FAILED calling solve_exact():", flush=True)
        traceback.print_exc()
        return False


def main():
    check_env_vars()
    gp = check_gurobipy_import()
    env_ok = check_gurobi_env(gp)
    solve_ok = check_tiny_gurobipy_solve(gp)
    cvxpy_ok = check_cvxpy_gurobi()
    project_ok = check_solve_exact()

    _section("SUMMARY")
    print(f"  gurobipy import:        {'OK' if gp is not None else 'FAIL'}", flush=True)
    print(f"  Gurobi env/license:     {'OK' if env_ok else 'FAIL'}", flush=True)
    print(f"  Tiny gurobipy solve:    {'OK' if solve_ok else 'FAIL'}", flush=True)
    print(f"  cvxpy sees GUROBI:      {'OK' if cvxpy_ok else 'FAIL'}", flush=True)
    print(f"  project solve_exact():  {'OK' if project_ok else 'FAIL'}", flush=True)

    if all([gp is not None, env_ok, solve_ok, cvxpy_ok, project_ok]):
        print("\nAll checks passed -- Gurobi is usable on this node. The n=640000 "
              "job's failure was likely node-specific; resubmitting (ideally "
              "pinned to this node type/partition) should work.", flush=True)
    else:
        print("\nAt least one check failed -- do not resubmit the 640k knapsack job "
              "until the first FAIL above is fixed (see traceback).", flush=True)


if __name__ == "__main__":
    main()
