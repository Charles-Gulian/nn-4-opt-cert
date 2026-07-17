"""Main experiment: 3-way comparison of B&B strategies on the robust knapsack.

  1. baseline  -- standard B&B (no NN, no oracle). Reference / correctness
                  check: must reproduce the exact Gurobi-certified Cost.
  2. oracle    -- the TRUE optimal value drives BOTH pruning and early
                  stopping. The ceiling: the best any predictor (even a
                  perfect one) could achieve with this mechanism.
  3. nn        -- the NN's (raw) prediction drives both mechanisms, each with
                  its own independently-calibrated margin:
                    prune_cutoff = nn_pred - prune_margin(tau_prune)
                    stop_cutoff  = nn_pred - stop_margin(tau_stop)
                  (both margins computed by the same one-sided conformal
                  quantile of over-prediction residuals, at their own tau;
                  default tau=0.05 / 95% confidence for both, per the
                  confirmed calibration convention.)

Usage:
    python scripts/calibrate_knapsack_conformal.py --n-train 20000   # once
    python scripts/run_bnb_experiment.py --n-train 20000 --n-instances 200
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nn.training import load_checkpoint, predict_denorm
from nn.metrics import mean_ci
from problems.robust_knapsack.branch_and_bound import solve_bnb
from problems.robust_knapsack.relaxation import build_relaxation_template

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"    / "robust_knapsack"
MODELS_DIR       = PROJECT_ROOT / "models"  / "robust_knapsack"
RESULTS_DIR      = PROJECT_ROOT / "results" / "robust_knapsack"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS = ("Cost",)


def _load_ensemble(n_train, folds):
    models, scalers = [], []
    for fold in range(folds):
        path = MODELS_DIR / f"dnn_knapsack_n{n_train}_fold{fold}.pt"
        m, s, _ = load_checkpoint(path)
        models.append(m); scalers.append(s)
    return models, scalers


def _ensemble_predict(models, scalers, X):
    preds = [predict_denorm(m, X, s) for m, s in zip(models, scalers)]
    return np.mean(preds, axis=0)


def _margin_from_residuals(e, tau):
    """One-sided split-conformal margin: the (1-tau) quantile of over-
    prediction residuals e_i = pred_i - cost_i, using the exact finite-sample
    order statistic (not np.quantile's interpolated default)."""
    n = len(e)
    e_sorted = np.sort(e)
    k = min(int(np.ceil((n + 1) * (1 - tau))), n)
    return float(e_sorted[k - 1])


def main():
    p = argparse.ArgumentParser(description="3-way B&B comparison: baseline / oracle / NN.")
    p.add_argument("--n-train", type=int, default=20_000)
    p.add_argument("--n-test", type=int, default=5_000)
    p.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--n-instances", type=int, default=200)
    p.add_argument("--tau-prune", type=float, default=0.01,
                   help="Risk level for the pruning margin (confidence = 1 - tau).")
    p.add_argument("--tau-stop", type=float, default=0.01,
                   help="Risk level for the early-stop margin (confidence = 1 - tau).")
    p.add_argument("--max-nodes", type=int, default=200_000)
    p.add_argument("--max-seconds", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-suffix", default="")
    p.add_argument("--raw-cutoffs", action="store_true",
                   help="Use v_hat directly (margin=0) for both cutoffs: prune if "
                        "LB >= v_hat, stop if UB <= v_hat. Error rates are then "
                        "bounded by the residual tails Pr(v_hat - v >= delta) "
                        "(stop) and Pr(v_hat - v <= -delta) (prune), rather than "
                        "controlled a priori by a conformal margin.")
    args = p.parse_args()

    residual_path = RESULTS_DIR / f"knapsack_oof_residuals_n{args.n_train}.npy"
    if not residual_path.exists():
        print(f"Missing {residual_path} -- run scripts/calibrate_knapsack_conformal.py first.")
        return
    e = np.load(residual_path)
    # The prune and stop cutoffs guard OPPOSITE error tails:
    #   - Pruning risk = the NN OVER-predicting Cost (prunes the optimal branch),
    #     so prune_margin is the upper (1 - tau_prune) quantile of e = pred - Cost.
    #   - Early-stop risk = the NN UNDER-predicting Cost (stops on a suboptimal
    #     incumbent). Given stop_cutoff = stop_margin - pred and the stop test
    #     incumbent <= stop_cutoff (i.e. cand_value >= pred - stop_margin), a
    #     SMALLER (more negative) stop_margin raises the acceptance threshold and
    #     makes stopping conservative. The right choice is the LOWER tau_stop
    #     quantile of e (its under-prediction tail), so that P(unsafe stop) <=
    #     tau_stop. Using the same (over-prediction) tail for both -- as an
    #     earlier version did -- is what made the NN arm ~23% unsafe.
    if args.raw_cutoffs:
        prune_margin = stop_margin = 0.0     # use v_hat directly (no margin)
    else:
        prune_margin = _margin_from_residuals(e, args.tau_prune)      # upper tail
        stop_margin = _margin_from_residuals(e, 1.0 - args.tau_stop)  # lower tail
    print(f"prune_margin(tau={args.tau_prune}) = {prune_margin:.4f}   "
          f"stop_margin(tau={args.tau_stop}) = {stop_margin:.4f}")

    models, scalers = _load_ensemble(args.n_train, args.folds)

    test_csv = args.data_dir / f"test_{args.n_test}.csv"
    df = pd.read_csv(test_csv)
    df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce")
    df = df[np.isfinite(df["Cost"])].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(df), size=min(args.n_instances, len(df)), replace=False)
    df = df.iloc[idx].reset_index(drop=True)

    mu_cols = [c for c in df.columns if c.startswith("Mu")]
    s2_cols = [c for c in df.columns if c.startswith("Sigma2")]
    feat_cols = [c for c in df.columns if c not in LABEL_COLS]
    X = df[feat_cols].values.astype(np.float64)
    ens_pred = _ensemble_predict(models, scalers, X)

    rows = []
    template = build_relaxation_template()

    n_correctness_fail = 0
    for i in range(len(df)):
        mu = df.loc[i, mu_cols].values.astype(float)
        sigma2 = df.loc[i, s2_cols].values.astype(float)
        true_cost = float(df.loc[i, "Cost"])

        # ---- 1. baseline ----
        t0 = time.time()
        base = solve_bnb(mu, sigma2, prune_cutoff=np.inf, stop_cutoff=np.inf,
                          max_nodes=args.max_nodes, max_seconds=args.max_seconds,
                          template=template, early_stop=False)
        base_time = time.time() - t0
        # Gurobi's DEFAULT MIPGap (1e-4 relative) was active during the
        # original data generation, so the stored "Cost" can occasionally be
        # a hair below the true optimum. Our custom B&B runs to PROVABLE
        # exact optimality, so it can legitimately find a marginally BETTER
        # value than the stored label -- that is not a bug. Only
        # base.value < true_cost indicates a genuine implementation problem.
        correct = base.value >= true_cost - 1e-3
        n_correctness_fail += not correct
        ref_optimum = max(true_cost, base.value)
        rows.append({
            "instance": i, "config": "baseline",
            "nodes": base.nodes_explored, "time_s": base_time,
            "value": base.value, "true_cost": ref_optimum,
            "unsafe": False, "opt_gap": 0.0,
            "bnb_status": base.status, "correctness_ok": correct,
        })

        # ---- 2. oracle: exact value drives BOTH pruning and stopping ----
        t0 = time.time()
        oracle = solve_bnb(mu, sigma2, prune_cutoff=-ref_optimum, stop_cutoff=-ref_optimum,
                            max_nodes=args.max_nodes, max_seconds=args.max_seconds,
                            template=template, early_stop=True)
        oracle_time = time.time() - t0
        oracle_unsafe = oracle.value < ref_optimum - 1e-4
        rows.append({
            "instance": i, "config": "oracle",
            "nodes": oracle.nodes_explored, "time_s": oracle_time,
            "value": oracle.value, "true_cost": ref_optimum,
            "unsafe": oracle_unsafe,
            "opt_gap": (ref_optimum - oracle.value) if oracle_unsafe else 0.0,
            "bnb_status": oracle.status, "correctness_ok": np.nan,
        })

        # ---- 3. nn: independently-calibrated prune/stop cutoffs ----
        prune_cutoff = prune_margin - ens_pred[i]   # MIN-sense: margin - raw pred
        stop_cutoff = stop_margin - ens_pred[i]
        t0 = time.time()
        nn = solve_bnb(mu, sigma2, prune_cutoff=prune_cutoff, stop_cutoff=stop_cutoff,
                       max_nodes=args.max_nodes, max_seconds=args.max_seconds,
                       template=template, early_stop=True)
        nn_time = time.time() - t0
        nn_unsafe = nn.value < ref_optimum - 1e-4
        rows.append({
            "instance": i, "config": "nn",
            "nodes": nn.nodes_explored, "time_s": nn_time,
            "value": nn.value, "true_cost": ref_optimum,
            "unsafe": nn_unsafe,
            "opt_gap": (ref_optimum - nn.value) if nn_unsafe else 0.0,
            "bnb_status": nn.status, "correctness_ok": np.nan,
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(df)} instances done", flush=True)

    per_instance = pd.DataFrame(rows)
    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    per_instance.to_csv(RESULTS_DIR / f"bnb_results{suffix}.csv", index=False)

    n_beat_gurobi = int((per_instance[per_instance.config == "baseline"]["value"]
                         > per_instance[per_instance.config == "baseline"]["true_cost"] + 1e-4).sum())
    if n_correctness_fail:
        print(f"\nWARNING: baseline B&B found a WORSE value than the stored Gurobi "
              f"Cost on {n_correctness_fail}/{len(df)} instances -- check implementation!")
    else:
        print(f"\nCorrectness check passed: baseline never found a value below the "
              f"stored Gurobi Cost on any of {len(df)} instances.")
    if n_beat_gurobi:
        print(f"(baseline found a marginally BETTER value than stored Cost on "
              f"{n_beat_gurobi}/{len(df)} instances -- expected, Gurobi's default "
              f"MIPGap during data generation can leave ~0.01% on the table.)")

    # ---- aggregate summary ----
    base_nodes_ci = mean_ci(per_instance[per_instance.config == "baseline"]["nodes"])
    base_time_ci = mean_ci(per_instance[per_instance.config == "baseline"]["time_s"])

    summary_rows = []
    for config in ["baseline", "oracle", "nn"]:
        sub = per_instance[per_instance.config == config]
        nodes_ci = mean_ci(sub["nodes"])
        time_ci = mean_ci(sub["time_s"])
        gap_when_unsafe = sub.loc[sub["unsafe"], "opt_gap"]
        summary_rows.append({
            "config": config, "n_instances": len(sub),
            "nodes_mean": nodes_ci["mean"], "nodes_ci_lo": nodes_ci["ci_lower"], "nodes_ci_hi": nodes_ci["ci_upper"],
            "time_mean": time_ci["mean"], "time_ci_lo": time_ci["ci_lower"], "time_ci_hi": time_ci["ci_upper"],
            "node_speedup": base_nodes_ci["mean"] / nodes_ci["mean"] if nodes_ci["mean"] else np.nan,
            "time_speedup": base_time_ci["mean"] / time_ci["mean"] if time_ci["mean"] else np.nan,
            "unsafe_rate": float(sub["unsafe"].mean()),
            "mean_gap_when_unsafe": float(gap_when_unsafe.mean()) if len(gap_when_unsafe) else 0.0,
            "max_gap_when_unsafe": float(gap_when_unsafe.max()) if len(gap_when_unsafe) else 0.0,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / f"bnb_summary{suffix}.csv", index=False)

    print(f"\n{'config':>10} {'unsafe%':>8} {'nodes(base->cfg)':>20} {'node_speedup':>13} "
          f"{'time_speedup':>13} {'mean_gap':>9}")
    for r in summary_rows:
        print(f"{r['config']:>10} {100*r['unsafe_rate']:>7.1f}% "
              f"{base_nodes_ci['mean']:>9.0f} -> {r['nodes_mean']:>7.0f}  "
              f"{r['node_speedup']:>12.2f}x  {r['time_speedup']:>12.2f}x  {r['mean_gap_when_unsafe']:>8.4f}")

    # Note: "oracle" is the ceiling for GUARANTEED-EXACT speedup (0% unsafe by
    # construction -- it only stops on an exact match to the true optimum).
    # The NN arm can legitimately exceed it in raw speed, since its cutoffs
    # (shaded by a margin) can be satisfied by a "good enough" solution found
    # even earlier than the true optimum -- trading away the exactness
    # guarantee for extra speed. So NN speedup > oracle speedup is not a bug;
    # it reflects that the two arms answer different questions ("fastest
    # provably-exact" vs "fastest at a controlled risk level").
    print(f"\nOracle speedup (ceiling for GUARANTEED-EXACT results): "
          f"{summary_rows[1]['node_speedup']:.2f}x  (unsafe rate {100*summary_rows[1]['unsafe_rate']:.1f}%)")
    print(f"NN speedup (at tau_prune={args.tau_prune}, tau_stop={args.tau_stop}): "
          f"{summary_rows[2]['node_speedup']:.2f}x  (unsafe rate {100*summary_rows[2]['unsafe_rate']:.1f}%)")

    print(f"\nWrote results to {RESULTS_DIR}/bnb_results{suffix}.csv and bnb_summary{suffix}.csv")


if __name__ == "__main__":
    main()
