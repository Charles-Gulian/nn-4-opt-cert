"""Full NN experiment for the correlated-mu knapsack redesign.

Model (see prototype_knapsack_corr.py for the gate): n=25, values c~U(20,30)
fixed, only input is mu ~ N(c, Sigma) with Sigma = tau^2[(1-rho)I + rho 11^T].
Robust constraint mu^T x + RHO_ROB*sqrt(x^T Sigma x) <= B, B = 0.5*sum(c).

The gate showed tau=5, rho=0.8 gives CV(v*)=0.17 (4x the old independent-per-item
baseline ~0.04) with greedy only ~8% exact (a large, honest negative class). This
script tests the decisive question the gate cannot: does the NN's RELATIVE error
q99/std(v*) drop below the old ~0.11, because the correlated common factor is a
smooth, learnable driver of v*? If so, certification separates cleanly.

Pipeline: generate n_train+n_cal+n_test instances (exact v* via Gurobi), train
DNN(6x256) on v*(mu), conformal q99 on a held-out calibration split, then certify
delta-optimality of the greedy value f on the test set.

Usage:
    /opt/anaconda3/envs/nn4opt/bin/python scripts/prototype_knapsack_corr_nn.py
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

_C = np.random.default_rng(2024).uniform(20.0, 30.0, size=N)
BUDGET = BUDGET_FRAC * _C.sum()


def sigma_matrix(tau, rho):
    return tau ** 2 * ((1 - rho) * np.eye(N) + rho * np.ones((N, N)))


def build_misocp(Sigma):
    L = np.linalg.cholesky(Sigma)
    x = cp.Variable(N, boolean=True)
    t = cp.Variable(nonneg=True)
    mu_p = cp.Parameter(N, nonneg=True)
    prob = cp.Problem(cp.Maximize(_C @ x),
                      [mu_p @ x + RHO_ROB * t <= BUDGET, cp.SOC(t, L.T @ x)])
    return prob, x, mu_p


def greedy(Sigma, mu):
    order = np.argsort(-_C / np.maximum(mu, 1e-12))
    x = np.zeros(N)
    for i in order:
        x[i] = 1
        if mu @ x + RHO_ROB * np.sqrt(x @ Sigma @ x) > BUDGET:
            x[i] = 0
    return float(_C @ x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--n-train", type=int, default=16000)
    ap.add_argument("--n-cal", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    Sigma = sigma_matrix(args.tau, args.rho)
    prob, x, mu_p = build_misocp(Sigma)
    rng = np.random.default_rng(args.seed)

    def gen(n, seed):
        r = np.random.default_rng(seed)
        mu = np.maximum(r.multivariate_normal(_C, Sigma, size=n), MU_FLOOR)
        v = np.empty(n)
        for i in range(n):
            mu_p.value = mu[i]
            prob.solve(solver=cp.GUROBI, verbose=False)
            v[i] = float(prob.value)
        return mu, v

    n_all = args.n_train + args.n_cal
    t0 = time.time()
    mu_tr, v_tr = gen(n_all, 100)
    print(f"[gen] {n_all} train/cal v* in {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    mu_te, v_te = gen(args.n_test, 200)
    f_te = np.array([greedy(Sigma, mu_te[i]) for i in range(args.n_test)])
    print(f"[gen] {args.n_test} test v*+greedy in {time.time()-t0:.0f}s", flush=True)

    tr, cal = slice(0, args.n_train), slice(args.n_train, n_all)
    xm, xs = mu_tr[tr].mean(0), mu_tr[tr].std(0); xs[xs == 0] = 1.0
    ym, ysd = float(v_tr[tr].mean()), float(v_tr[tr].std())
    nrm = lambda M: ((M - xm) / xs).astype(np.float32)

    model = DNN(input_dim=N, hidden_dims=[256] * 6)
    tl = to_loader(nrm(mu_tr[tr]), ((v_tr[tr] - ym) / ysd).astype(np.float32), batch_size=256)
    vl = to_loader(nrm(mu_tr[cal]), ((v_tr[cal] - ym) / ysd).astype(np.float32),
                   batch_size=256, shuffle=False)
    t0 = time.time()
    model, _, _ = train_model(model, tl, vl, n_epochs=args.epochs,
                              learning_rate=1e-3, weight_decay=1e-4, verbose=False)
    print(f"[train] {args.epochs} epochs in {time.time()-t0:.0f}s", flush=True)

    def predict(M):
        model.eval()
        with torch.no_grad():
            return model(torch.tensor(nrm(M))).numpy().ravel() * ysd + ym

    vhat_cal = predict(mu_tr[cal])
    q99 = conformal_offset(v_tr[cal] - vhat_cal, 0.99)   # Pr(v* <= vhat + q99) >= 0.99

    sv = v_te.std()
    vhat_te = predict(mu_te)
    mae = np.mean(np.abs(vhat_te - v_te))

    print("\n===== RESULTS (n=%d items, tau=%.1f, rho=%.1f) =====" % (N, args.tau, args.rho))
    print(f"std(v*)={sv:.3f}  CV(v*)={sv/v_te.mean():.4f}")
    print(f"NN MAE(test)={mae:.4f} ({mae/sv:.4f} std)")
    print(f"q99={q99:.4f}   q99/std(v*)={q99/sv:.4f}   (old independent case ~0.11)")

    for frac in (0.01, 0.03, 0.05):
        delta = frac * sv
        UB = vhat_te + q99
        cert = (UB - f_te) <= delta
        truth = (v_te - f_te) <= delta
        tp = int(np.sum(cert & truth)); fp = int(np.sum(cert & ~truth))
        fn = int(np.sum(~cert & truth))
        ceil = np.mean((UB - v_te) <= delta)      # perfect-solver ceiling (f=v*)
        print(f"\ndelta={frac*100:.0f}%*std={delta:.3f}")
        print(f"  truly delta-opt={100*np.mean(truth):.1f}%  certified={100*(tp+fp)/args.n_test:.1f}%"
              f"  TPR={100*tp/max(tp+fn,1):.1f}%  joint FPR={100*fp/args.n_test:.2f}%")
        print(f"  perfect-solver ceiling={100*ceil:.1f}%")


if __name__ == "__main__":
    main()
