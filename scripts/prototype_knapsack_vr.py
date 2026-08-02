"""Prototype: learn the CONTINUOUS SOCP-relaxation value v_r for a larger
robust knapsack (n items), and see whether the NN error q_r stays small enough
that certifying against the exact optimum v* works well.

Motivation (measured): the SOCP relaxation gap v_r - v* is ~O(one item) in
absolute terms, so it shrinks ~1/n relative to std(v*). Training on the smooth,
continuous v_r (vs the discontinuous exact v*) should give a small NN error q_r.
The open question is whether q_r stays small as the input dimension 2n grows.

Certificate (MAX-sense): v_hat_r + q_r >= v_r >= v*, so certify a feasible f as
delta-optimal iff (v_hat_r + q_r) - f <= delta  =>  v* - f <= delta. Ground truth
uses the exact v* (Gurobi); the NN only ever sees v_r (CLARABEL, no Gurobi).

Usage:
    /opt/anaconda3/envs/nn4opt/bin/python scripts/prototype_knapsack_vr.py --n 100
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

RHO = 2.0


def build_problems(n, c, B):
    mu_p = cp.Parameter(n, nonneg=True)
    sig_p = cp.Parameter(n, nonneg=True)
    xr = cp.Variable(n); tr = cp.Variable(nonneg=True)
    relax = cp.Problem(cp.Maximize(c @ xr),
                       [mu_p @ xr + RHO * tr <= B, cp.SOC(tr, cp.multiply(sig_p, xr)),
                        xr >= 0, xr <= 1])
    xi = cp.Variable(n, boolean=True); ti = cp.Variable(nonneg=True)
    exact = cp.Problem(cp.Maximize(c @ xi),
                       [mu_p @ xi + RHO * ti <= B, cp.SOC(ti, cp.multiply(sig_p, xi))])
    return mu_p, sig_p, relax, exact


def greedy_value(mu, s2, c, B, n):
    sig = np.sqrt(s2)
    order = np.argsort(-c / np.maximum(mu, 1e-12))
    x = np.zeros(n)
    for i in order:
        x[i] = 1.0
        if B - mu @ x - RHO * np.sqrt(np.sum((sig * x) ** 2)) < 0:
            x[i] = 0.0
    return float(c @ x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=16000)
    ap.add_argument("--n-cal", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=500)
    args = ap.parse_args()
    n = args.n

    rng = np.random.default_rng(0)
    c = rng.uniform(0.5, 1.5, size=n)          # item values (fixed across instances)
    B = 0.5 * n                                # budget scales with n
    mu_p, sig_p, relax, exact = build_problems(n, c, B)

    def solve_vr(mu, s2):
        mu_p.value = mu; sig_p.value = np.sqrt(s2)
        relax.solve(solver=cp.CLARABEL); return float(relax.value)

    def solve_vstar(mu, s2):
        mu_p.value = mu; sig_p.value = np.sqrt(s2)
        exact.solve(solver=cp.GUROBI); return float(exact.value)

    def sample(N, seed):
        r = np.random.default_rng(seed)
        return r.uniform(0.5, 1.5, size=(N, n)), r.uniform(0.08, 0.12, size=(N, n))

    # ---- training + calibration data: v_r labels only (CLARABEL, no Gurobi) ----
    Ntot = args.n_train + args.n_cal
    mu_tr, s2_tr = sample(Ntot, 1)
    t0 = time.time()
    vr_tr = np.array([solve_vr(mu_tr[i], s2_tr[i]) for i in range(Ntot)])
    print(f"[gen] {Ntot} v_r solves (CLARABEL) in {time.time()-t0:.0f}s", flush=True)
    X_tr = np.hstack([mu_tr, s2_tr])

    # ---- test data: v_r (target check) + exact v* (ground truth) + greedy f ----
    mu_te, s2_te = sample(args.n_test, 2)
    t0 = time.time()
    vr_te = np.array([solve_vr(mu_te[i], s2_te[i]) for i in range(args.n_test)])
    vs_te = np.array([solve_vstar(mu_te[i], s2_te[i]) for i in range(args.n_test)])
    f_te = np.array([greedy_value(mu_te[i], s2_te[i], c, B, n) for i in range(args.n_test)])
    print(f"[gen] {args.n_test} test (v_r+v*+greedy) in {time.time()-t0:.0f}s", flush=True)
    X_te = np.hstack([mu_te, s2_te])

    # ---- standardize + train DNN on v_r (single model; calib split held out) ----
    tr, cal = slice(0, args.n_train), slice(args.n_train, Ntot)
    xm, xs = X_tr[tr].mean(0), X_tr[tr].std(0); xs[xs == 0] = 1.0
    ym, ysd = float(vr_tr[tr].mean()), float(vr_tr[tr].std())
    nrm = lambda X: ((X - xm) / xs).astype(np.float32)

    model = DNN(input_dim=2 * n, hidden_dims=[256] * 6)
    tl = to_loader(nrm(X_tr[tr]), ((vr_tr[tr] - ym) / ysd).astype(np.float32), batch_size=256)
    vl = to_loader(nrm(X_tr[cal]), ((vr_tr[cal] - ym) / ysd).astype(np.float32),
                   batch_size=256, shuffle=False)
    t0 = time.time()
    model, _, _ = train_model(model, tl, vl, n_epochs=args.epochs,
                              learning_rate=1e-3, weight_decay=1e-4, verbose=False)
    print(f"[train] {args.epochs} epochs in {time.time()-t0:.0f}s", flush=True)

    def predict(X):
        model.eval()
        with torch.no_grad():
            p = model(torch.tensor(nrm(X))).numpy().ravel()
        return p * ysd + ym

    # ---- conformal q_r from calibration residuals (v_r - v_hat_r) ----
    vhat_cal = predict(X_tr[cal]); resid_cal = vr_tr[cal] - vhat_cal
    q_r = conformal_offset(resid_cal, 0.99)     # Pr(v_r <= v_hat_r + q_r) >= 0.99

    sv = vs_te.std()                            # spread of the TRUE optimum
    vhat_te = predict(X_te)
    mae_r = np.mean(np.abs(vhat_te - vr_te))
    relax_gap = np.mean(vr_te - vs_te)

    print("\n===== RESULTS (n=%d, 2n=%d inputs) =====" % (n, 2 * n))
    print(f"std(v*)={sv:.4f}   NN MAE on v_r (test)={mae_r:.4f} ({mae_r/sv:.3f} std)")
    print(f"q_r={q_r:.4f}   q_r/std(v*)={q_r/sv:.3f}   (was 0.108 for exact-v* at n=50)")
    print(f"relaxation gap v_r-v* (test): mean={relax_gap:.4f} ({relax_gap/sv:.3f} std)")

    delta = 0.03 * sv
    UB = vhat_te + q_r
    cert = (UB - f_te) <= delta
    truth = (vs_te - f_te) <= delta
    tp = int(np.sum(cert & truth)); fp = int(np.sum(cert & ~truth))
    tn = int(np.sum(~cert & ~truth)); fn = int(np.sum(~cert & truth))
    ceiling = np.mean((UB - vs_te) <= delta)   # perfect-solver ceiling (f = v*)
    print(f"\ndelta=3%*std(v*)={delta:.4f}")
    print(f"  certified={100*(tp+fp)/args.n_test:.1f}%  TPR={100*tp/max(tp+fn,1):.1f}%  "
          f"joint FPR={100*fp/args.n_test:.2f}%  truly-delta-opt={100*np.mean(truth):.1f}%")
    print(f"  perfect-solver ceiling (f=v*): {100*ceiling:.1f}%   (was 4.2% for exact-v* at n=50)")


if __name__ == "__main__":
    main()
