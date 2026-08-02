"""Correlated-mu knapsack: NN certification experiment, exact OR SOCP-relaxation
target, arbitrary data size. Saves arrays + sweeps delta.

Model (n=25): values c~U(20,30) fixed; input mu ~ N(c, Sigma),
Sigma = tau^2[(1-rho)I + rho 11^T]; robust constraint
mu^T x + RHO_ROB*sqrt(x^T Sigma x) <= B, B=0.5*sum(c). See prototype_knapsack_corr.py.

--target exact : train the NN on the exact MISOCP optimum v* (Gurobi).
--target relax : train on the CONTINUOUS SOCP relaxation value v_r (x in [0,1],
                 CLARABEL) -- smoother/easier to learn (smaller q_r) but v_r >= v*,
                 so the relaxation gap v_r-v* adds to the certificate slack.

Certificate (MAX-sense): UB = vhat + q99 >= (target) >= v*  [v_r >= v* makes the
relax UB valid too], certify f delta-optimal iff UB - f <= delta => v* - f <= delta.
Ground truth on the TEST set is always the exact v* (+ greedy f); for --target relax
we also solve v_r on test to report the relaxation gap.

Usage:
    .../python scripts/prototype_knapsack_corr_full.py --target exact --n-train 64000 --n-cal 16000
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import cvxpy as cp
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nn.models import DNN
from nn.training import train_model, to_loader
from nn.metrics import conformal_offset

N = 25
BUDGET_FRAC = 0.5
RHO_ROB = 1.0
MU_FLOOR = 0.1
OUT = pathlib.Path("/private/tmp/claude-501/-Users-charlesgulian-Desktop-Projects-optimal-rolling-blackout/22e95144-1b5d-4142-af66-ceb1b4ad9159/scratchpad")

_C = np.random.default_rng(2024).uniform(20.0, 30.0, size=N)
BUDGET = BUDGET_FRAC * _C.sum()


def sigma_matrix(tau, rho):
    return tau ** 2 * ((1 - rho) * np.eye(N) + rho * np.ones((N, N)))


def build_problem(Sigma, relax):
    """MISOCP (boolean, Gurobi) or continuous SOCP relaxation (CLARABEL)."""
    L = np.linalg.cholesky(Sigma)
    x = cp.Variable(N, boolean=not relax)
    t = cp.Variable(nonneg=True)
    mu_p = cp.Parameter(N, nonneg=True)
    cons = [mu_p @ x + RHO_ROB * t <= BUDGET, cp.SOC(t, L.T @ x)]
    if relax:
        cons += [x >= 0, x <= 1]
    return cp.Problem(cp.Maximize(_C @ x), cons), x, mu_p


def greedy(Sigma, mu):
    order = np.argsort(-_C / np.maximum(mu, 1e-12))
    x = np.zeros(N)
    for i in order:
        x[i] = 1
        if mu @ x + RHO_ROB * np.sqrt(x @ Sigma @ x) > BUDGET:
            x[i] = 0
    return float(_C @ x)


def solve_many(prob, x, mu_p, mu, solver):
    v = np.empty(len(mu))
    for i in range(len(mu)):
        mu_p.value = mu[i]
        prob.solve(solver=solver, verbose=False)
        v[i] = float(prob.value)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["exact", "relax"], required=True)
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--n-train", type=int, default=16000)
    ap.add_argument("--n-cal", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=1000)   # standard paper recipe
    ap.add_argument("--reuse", action="store_true",
                    help="load saved raw arrays and skip (expensive) generation")
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[256] * 6)
    ap.add_argument("--suffix", default="", help="tag suffix so arch sweeps don't collide")
    args = ap.parse_args()
    relax = args.target == "relax"
    tag = f"corr_{args.target}_{args.n_train//1000}k{args.suffix}"
    print(f"=== {tag}  tau={args.tau} rho={args.rho} epochs={args.epochs} "
          f"n_train={args.n_train} n_cal={args.n_cal} ===", flush=True)

    Sigma = sigma_matrix(args.tau, args.rho)
    n_all = args.n_train + args.n_cal
    base_tag = f"corr_{args.target}_{args.n_train//1000}k"   # data shared across arch sweep
    data_path = OUT / f"{base_tag}_data.npz"

    if args.reuse and data_path.exists():
        d = np.load(data_path)
        mu_trcal, y_trcal = d["mu_trcal"], d["y_trcal"]
        mu_te, vstar_te, f_te, y_te = d["mu_te"], d["vstar_te"], d["f_te"], d["y_te"]
        print(f"[reuse] loaded raw arrays from {data_path.name}", flush=True)
    else:
        prob_t, x_t, mu_t = build_problem(Sigma, relax)         # target problem
        solver_t = cp.CLARABEL if relax else cp.GUROBI
        prob_e, x_e, mu_e = build_problem(Sigma, relax=False)   # exact truth (Gurobi)

        def sample(n, seed):
            r = np.random.default_rng(seed)
            return np.maximum(r.multivariate_normal(_C, Sigma, size=n), MU_FLOOR)

        mu_trcal = sample(n_all, 100)
        t0 = time.time()
        y_trcal = solve_many(prob_t, x_t, mu_t, mu_trcal, solver_t)   # target labels
        print(f"[gen] {n_all} train/cal {args.target} in {time.time()-t0:.0f}s", flush=True)

        mu_te = sample(args.n_test, 200)
        t0 = time.time()
        vstar_te = solve_many(prob_e, x_e, mu_e, mu_te, cp.GUROBI)    # ground-truth optimum
        f_te = np.array([greedy(Sigma, mu_te[i]) for i in range(args.n_test)])
        y_te = (solve_many(prob_t, x_t, mu_t, mu_te, solver_t) if relax else vstar_te)
        print(f"[gen] {args.n_test} test in {time.time()-t0:.0f}s", flush=True)
        np.savez(data_path, mu_trcal=mu_trcal, y_trcal=y_trcal, mu_te=mu_te,
                 vstar_te=vstar_te, f_te=f_te, y_te=y_te)
        print(f"[save] raw arrays -> {data_path.name}", flush=True)

    tr, cal = slice(0, args.n_train), slice(args.n_train, n_all)
    xm, xs = mu_trcal[tr].mean(0), mu_trcal[tr].std(0); xs[xs == 0] = 1.0
    ym, ysd = float(y_trcal[tr].mean()), float(y_trcal[tr].std())
    nrm = lambda M: ((M - xm) / xs).astype(np.float32)

    model = DNN(input_dim=N, hidden_dims=args.hidden_dims)
    tl = to_loader(nrm(mu_trcal[tr]), ((y_trcal[tr] - ym) / ysd).astype(np.float32), batch_size=256)
    vl = to_loader(nrm(mu_trcal[cal]), ((y_trcal[cal] - ym) / ysd).astype(np.float32),
                   batch_size=256, shuffle=False)
    t0 = time.time()
    model, _, _ = train_model(model, tl, vl, n_epochs=args.epochs,
                              learning_rate=1e-3, weight_decay=1e-4, verbose=False)
    print(f"[train] {args.epochs} epochs in {time.time()-t0:.0f}s", flush=True)

    def predict(M):
        model.eval()
        with torch.no_grad():
            return model(torch.tensor(nrm(M))).numpy().ravel() * ysd + ym

    vhat_cal = predict(mu_trcal[cal])
    q99 = conformal_offset(y_trcal[cal] - vhat_cal, 0.99)
    vhat_te = predict(mu_te)

    sv = vstar_te.std()
    mae = np.mean(np.abs(vhat_te - y_te))
    resid_cal = y_trcal[cal] - vhat_cal          # (target - prediction) on calibration
    np.savez(OUT / f"{tag}.npz", mu_te=mu_te, vstar_te=vstar_te, f_te=f_te,
             y_te=y_te, vhat_te=vhat_te, q99=q99, sv=sv, resid_cal=resid_cal)

    print(f"\n===== {tag} RESULTS =====")
    print(f"std(v*)={sv:.3f}  CV(v*)={sv/vstar_te.mean():.4f}")
    print(f"NN MAE(test on {args.target})={mae:.4f} ({mae/sv:.4f} std)")
    print(f"q99={q99:.4f}  q99/std={q99/sv:.4f}")
    if relax:
        gap = np.mean(y_te - vstar_te)   # v_r - v*  (>=0)
        print(f"relaxation gap v_r-v* (test): mean={gap:.4f} ({gap/sv:.4f} std)  "
              f"effective slack (q99+gap)/std={(q99+gap)/sv:.4f}")

    print(f"\n{'delta':>7} {'truly-opt':>10} {'certified':>10} {'TPR':>7} "
          f"{'jFPR':>7} {'ceiling':>8}")
    for frac in (0.01, 0.03, 0.05, 0.08, 0.10, 0.12):
        delta = frac * sv
        UB = vhat_te + q99
        cert = (UB - f_te) <= delta
        truth = (vstar_te - f_te) <= delta
        tp = int(np.sum(cert & truth)); fp = int(np.sum(cert & ~truth))
        fn = int(np.sum(~cert & truth))
        ceil = np.mean((UB - vstar_te) <= delta)
        print(f"{frac*100:6.0f}% {100*np.mean(truth):9.1f}% {100*(tp+fp)/args.n_test:9.1f}% "
              f"{100*tp/max(tp+fn,1):6.1f}% {100*fp/args.n_test:6.2f}% {100*ceil:7.1f}%",
              flush=True)


if __name__ == "__main__":
    main()
