"""4-fold OOF training + conformal calibration for the correlated-mu robust
knapsack, matching the protocol used by every other experiment in the paper.

The expensive MISOCP/SOCP labels are already cached by
scripts/prototype_knapsack_corr_full.py (corr_{target}_{n}k_data.npz), so this
script does NO optimization solves -- it only trains and calibrates.

Protocol (mirrors train_generic.py + the q_alpha construction):
  * KFold(4) over the pooled train+cal pool; each fold trains on 3/4 and
    predicts the held-out 1/4 -> OUT-OF-FOLD residuals, which is what the
    conformal offset must be built from (an in-sample prediction is
    over-optimistic and would bias q).
  * each fold model also predicts the shared test set, so certification can be
    averaged over folds exactly as _knapsack_row does.

MAX-sense conventions (this is the only maximization problem in the suite):
  residual  = target - prediction ;  q = the (1-alpha) upper offset, so
  vhat + q is a (1-alpha) UPPER confidence bound on the target, and
  certify iff  vhat + q - f <= delta.
For --target relax the NN learns v_r >= v*, so the bound covers v* as well.

Outputs (per config, under results/knapsack_corr_{target}_{n}k/):
  fold_test_predictions.csv   fold, idx, g (NN pred), v (learned target),
                              vstar (true optimum), f (greedy value)
  oof_residuals.npy           pooled out-of-fold (target - prediction)
  summary.json                q99, std(v*), MAE, relaxation gap

Usage:
    .../python scripts/knapsack_corr_kfold.py --target exact --n-train 16000 --n-cal 4000
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nn.models import DNN
from nn.training import train_model, to_loader
from nn.metrics import conformal_offset

DATA = PROJECT_ROOT / "data" / "knapsack_corr"
RESULTS = PROJECT_ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["exact", "relax"], required=True)
    ap.add_argument("--n-pool", type=int, default=20000,
                    help="training pool size; uses the first n rows of the shared pool")
    ap.add_argument("--pool-file", default="pool_80000.npz")
    ap.add_argument("--test-file", default="test_5000.npz")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[256] * 6)
    ap.add_argument("--level", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="",
                    help="appended to the results dir, so architecture sweeps "
                         "do not overwrite the headline run")
    args = ap.parse_args()

    n_pool = args.n_pool
    ycol = "vstar" if args.target == "exact" else "vr"
    p = np.load(DATA / args.pool_file)
    t = np.load(DATA / args.test_file)
    # nested subset: the n=20k config trains on the first 20k rows of the same
    # pool the n=80k config uses, so data scaling is a clean nested comparison.
    mu, y = p["mu"][:n_pool], p[ycol][:n_pool]
    ok = np.isfinite(y)
    mu, y = mu[ok], y[ok]
    mu_te, y_te = t["mu"], t[ycol]
    vstar_te, f_te = t["vstar"], t["f"]
    out_dir = RESULTS / f"knapsack_corr_{args.target}_{n_pool//1000}k{args.suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {out_dir.name}: target={args.target} pool={len(mu)} "
          f"test={len(mu_te)} folds={args.folds} epochs={args.epochs} ===", flush=True)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.full(len(mu), np.nan)
    rows = []
    for k, (tr_idx, va_idx) in enumerate(kf.split(mu)):
        t0 = time.time()
        xm, xs = mu[tr_idx].mean(0), mu[tr_idx].std(0); xs[xs == 0] = 1.0
        ym, ysd = float(y[tr_idx].mean()), float(y[tr_idx].std()) or 1.0
        nrm = lambda M: ((M - xm) / xs).astype(np.float32)

        model = DNN(input_dim=mu.shape[1], hidden_dims=args.hidden_dims)
        tl = to_loader(nrm(mu[tr_idx]), ((y[tr_idx] - ym) / ysd).astype(np.float32),
                       batch_size=256)
        vl = to_loader(nrm(mu[va_idx]), ((y[va_idx] - ym) / ysd).astype(np.float32),
                       batch_size=256, shuffle=False)
        model, _, _ = train_model(model, tl, vl, n_epochs=args.epochs,
                                  learning_rate=1e-3, weight_decay=1e-4, verbose=False)

        def predict(M):
            model.eval()
            with torch.no_grad():
                return model(torch.tensor(nrm(M))).numpy().ravel() * ysd + ym

        oof[va_idx] = y[va_idx] - predict(mu[va_idx])       # OOF residual (target - pred)
        g_te = predict(mu_te)
        rows.append(pd.DataFrame({"fold": k, "idx": np.arange(len(mu_te)),
                                  "g": g_te, "v": y_te, "vstar": vstar_te, "f": f_te}))
        print(f"  fold {k}: OOF resid mean={np.nanmean(oof[va_idx]):+.4f} "
              f"q{int(args.level*100)}={conformal_offset(oof[va_idx], args.level):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    preds = pd.concat(rows, ignore_index=True)
    preds.to_csv(out_dir / "fold_test_predictions.csv", index=False)
    np.save(out_dir / "oof_residuals.npy", oof)

    q = float(conformal_offset(oof, args.level))
    sv = float(np.std(vstar_te))
    mae = float(np.mean(np.abs(preds["g"].values - np.tile(y_te, args.folds))))
    gap = float(np.mean(y_te - vstar_te))     # 0 for exact; >0 for relax
    summary = {"config": out_dir.name, "target": args.target, "n_pool": int(n_pool),
               "folds": args.folds, "epochs": args.epochs, "level": args.level,
               "q": q, "q_over_std": q / sv, "std_vstar": sv, "mae": mae,
               "mean_relax_gap": gap, "eff_slack_over_std": (q + max(gap, 0.0)) / sv}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  q{int(args.level*100)}={q:.4f}  q/std={q/sv:.4f}  MAE={mae:.4f}  "
          f"relax_gap={gap:.4f}  eff_slack/std={summary['eff_slack_over_std']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
