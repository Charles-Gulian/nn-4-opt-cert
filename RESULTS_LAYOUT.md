# Where the paper results live

A one-page map of every results artifact and the code that produces it. Two Jupyter
notebooks in `notebooks/` consume these: `paper_figures.ipynb` (all figures) and
`results_analysis.ipynb` (inspect predictions, recompute every table number).

## The reporting unit: per-fold test predictions

Every certification result is derived from **raw per-instance test predictions**, one CSV
per config:

- Small problems: `results/{qcqp,mimo,ik_lass1,ik_lass2}/fold_test_predictions.csv`
- AC-OPF: `results/acopf-cert/acopf_{socp,chordal_sdp}_{case}/fold_test_predictions.csv`

Columns: `fold, idx, g, v, f` where
`g = v̂(θ)` (NN prediction), `v = v_r(θ)` (relaxation value / training target),
`f = f(x̂;θ)` (local/heuristic solver cost). **`idx` is the position among the
*feasible* test rows** (rows with non-NaN `Cost` and `LocalCost` are dropped first), so
`idx` lines up with the feasible rows of the raw test CSV in order — not the raw row number.
Four folds are pooled (each row appears once per fold).

## Ground-truth optimum `v(θ)` (for the ROC positive class)

Not stored in the prediction CSV; recomputed from the raw θ:
- **QCQP** — SDP relaxation is exact (S-lemma): `v = v_r` (the `v` column).
- **MIMO** — brute force over `x ∈ {−1,1}^n` (n=2 ⇒ 4 candidates): `problems.mimo_detection.problem.A_REAL`.
- **IK** — analytic closed form: `problems.ik.problem.ground_truth(θ)`.
- **AC-OPF** — chordal-SDP value at the same θ (best available); `case2869pegase` has no chordal ⇒ SOCP-as-truth.

The join logic lives in `scripts/compute_roc_auc.py` (`_ik_truth`, `_mimo_truth`,
`_acopf_chordal_truth`, `_feasible_theta`).

## Summary tables (the numbers that go in the paper)

- `results/roc_auc_summary.csv` — QCQP, MIMO, IK, AC-OPF. Columns: `config, experiment,
  relaxation, case, mean_vr, std_vr, mean_relax_gap (v−v_r), mean_opt_gap (f−v),
  pct_feasible, mae (|g−v|), ae_p5, ae_p95, delta0, auc, n_pos, n_neg`.
  Produced by `scripts/compute_roc_auc.py`.
- `results/roc_auc_knapsack.csv` — robust knapsack, both dataset sizes. Columns: `config,
  n_train, n_test, mean_cost, mae, mean_greedy_gap, auc, n_pos, n_neg, delta0`.
  Produced by `scripts/compute_roc_auc_knapsack.py`.
- `results/roc_curves/<config>.npz` — per-config ROC arrays: `fpr, tpr, tau, auc, exp`.

## Per-config auxiliary CSVs (fixed-(δ,α) view, pre-ROC reframing)

In each `results/<config>/` dir:
- `certification.csv` — TP/FP/TN/FN + TPR/FPR/TNR/FNR per (fold, level) at δ=0.1% (relative form).
- `conformal_offsets.csv` — split-conformal offset `q_α` per (fold, level).
- `fold_test_metrics.csv` — per-fold MAE-style error summary + signed-error percentiles.
- `solve_times.json` — mean relaxation and local solve seconds (**small problems only**;
  AC-OPF has none — its data predates this record).
- `val_residuals.csv` — per-fold validation residuals `g−v` used for calibration.
Produced by `scripts/evaluate_certify.py`; aggregated by `scripts/build_summary_table.py`
into `results/summary_{per_fold,by_config}.csv`.

## Branch-and-bound / robust knapsack

`results/robust_knapsack/`:
- `bnb_results.csv` — per-instance × {baseline, oracle, nn}: `nodes, time_s, value,
  true_cost, unsafe, opt_gap, bnb_status, correctness_ok`.
- `bnb_summary.csv` — aggregate node/time speedup, unsafe rate, mean gap per arm.
- `knapsack_oof_residuals_n{20000,80000}.npy` — OOF residuals `pred−Cost` for calibration.
- `eval_summary.csv`, `fold_metrics.csv` — NN error summaries.
Produced by `scripts/run_bnb_experiment.py` (+ `calibrate_knapsack_conformal.py`).
NOTE: the B&B speedup result is still under analysis (see the team notes) — the
certification AUC in `roc_auc_knapsack.csv` is the settled part.

## Raw data and models

- Raw test sets (θ + `Cost` + `LocalCost`): `data/{qcqp,mimo,ik_lass1,ik_lass2}/test_5000.csv`;
  AC-OPF `data/acopf-hpc/test_5000_{relax}_{case}.csv` (pegase socp overwritten with the
  post-phase-shifter-fix data); knapsack `data/robust_knapsack/test_{5000,20000}.csv`.
- Fold checkpoints: `models/{key}/`, `models/acopf/`, `models/robust_knapsack/`
  (self-contained: weights + input/target scalers; load via `nn.training.load_checkpoint`,
  predict via `nn.training.predict_denorm`).

## Figures

`figures/`: `training_history_qcqp_fold0.{csv,png}`, `error_distribution_boxplot.png`,
`framework_diagram.pdf`. The notebooks regenerate the paper-ready PDFs (normalized-error
box-and-whisker, ROC curves, training history).
