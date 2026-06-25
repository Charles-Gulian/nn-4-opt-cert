# nn-4-opt-cert

Testing whether deep neural networks can predict the optimal value of convex
relaxations of hard, non-convex optimization problems (QCQP, AC-OPF, MIMO
detection, polynomial optimization, ...). The longer-term goal is to use these
predictions inside a branch-and-bound solver to prune the search tree.

## Workflow

Each test problem follows the same three-stage workflow:

1. **Generate inputs** — sample random problem parameters.
2. **Label data** — solve a convex relaxation of the problem for each sampled
   parameter set to obtain a "ground truth" optimal value.
3. **Train / evaluate a DNN** — fit a feedforward network to predict the
   relaxation's optimal value from the problem parameters, then compare
   predictions against the relaxation solver on held-out data.

## Project layout

- `problems/<name>/` — one subpackage per test problem. Each exposes a
  `problem.py` with a standard interface (see below) and a `generate_data.py`
  for building labeled datasets.
  - `problems/qcqp_example/` — `min (x-a)^2 + (y-b)^2 s.t. xy >= 1`,
    parameterized by `p = (a, b)`. Labeled via an SDP relaxation (cvxpy);
    the relaxation value is always a valid lower bound, and exact by the
    S-lemma whenever the relaxation solution is rank-1.
- `nn/` — problem-agnostic model (`models.py`, `DNN` with configurable
  `hidden_dims`) and training utilities (`training.py`: batching, training
  loop, k-fold cross-validation; `metrics.py`: error summary stats).
- `scripts/` — generic, problem-parameterized pipeline scripts (data
  generation, training, evaluation).
- `data/`, `models/`, `results/` — generated artifacts (gitignored).
- `legacy/` — old AC-OPF code (`core/`) and notebooks, kept for reference.
  The AC-OPF pipeline depends on a MATLAB SDP solver (`OPF_Solver`, not
  included) and will be revisited separately.

## Standard problem interface

Each `problems/<name>/problem.py` exposes three functions, so pipeline
scripts can be reused across problems by swapping the module:

- `sample_parameters(N, args=None)` — returns an `(N, d)` array of sampled
  parameter vectors `p`.
- `solve_relaxation(p, args=None)` — returns `(value, result)`, where `value`
  is the convex relaxation's optimal value (always a valid lower bound on the
  true optimum, and exact when `result["exact"]` is `True`), and `result` is
  a dict that may also carry the recovered solution.
- `solve_local(p, args=None)` — returns `(value, result)` from a local search
  / heuristic solve of the original non-convex problem (e.g. IPOPT via
  Pyomo), for reference/comparison.

## Running the QCQP example

```
python scripts/generate_qcqp_data.py --n-samples 5000 --seed 0
python scripts/train_qcqp.py --n-samples 5000 --n-epochs 2000
python scripts/evaluate_results.py results/qcqp_example/cv_predictions_QCQP_example_2d_5000samples.csv
```

This generates labeled data (via `solve_relaxation`), trains a `DNN` with
2-fold cross-validation (saving per-fold model checkpoints and out-of-fold
predictions), and prints mean/95% CI/max absolute prediction error per fold
and overall.

### Certifying local-solver optimality

A separate held-out test stage checks whether the trained DNN can be used to
certify that a local solver (IPOPT) has found a globally optimal solution,
by comparing the local solver's value against both the SDP relaxation value
(ground truth lower bound) and the DNN's predicted relaxation value:

```
python scripts/generate_qcqp_test_data.py --n-samples 1000 --seed 1
python scripts/certify_qcqp.py --n-samples 1000 --model-path models/qcqp_example/dnn_QCQP_example_2d_5000samples_fold0.pth
```

`generate_qcqp_test_data.py` solves each test point both via the SDP
relaxation and via IPOPT (`solve_local`), and saves `Cost` (relaxation),
`Exact`, and `LocalCost` (local solver) columns. `certify_qcqp.py` adds the
DNN's prediction (`Pred`) and reports a confusion matrix: a local solution is
*actually* optimal if `|LocalCost - Cost| <= tol` (relaxation is tight), and
the DNN *certifies* optimality if `|LocalCost - Pred| <= tol`. False
positives (certifying a suboptimal local solution as optimal) are the
dangerous error mode for pruning; false negatives are merely conservative.

**Note:** `solve_local` calls IPOPT via Pyomo. In the `nn4opt` conda env on
macOS, IPOPT's bundled `libdmumps_seq.dylib` has a broken `@rpath` to
`liblapack.3.dylib` in the base conda env rather than `nn4opt`'s own lib
directory. Work around this by setting:

```
export DYLD_FALLBACK_LIBRARY_PATH=/opt/anaconda3/envs/nn4opt/lib
```

before running any script that calls `solve_local` (currently
`generate_qcqp_test_data.py` and the notebook's "Bonus" / certification
sections).

## Roadmap

- Add additional test problems: non-convex QCQP variants, MIMO detection, a
  polynomial optimization problem, and (separately) AC-OPF.
- Evaluate on a held-out test set (cross-validation above is for model
  selection / sanity checking).
- Use trained DNN predictions for tree pruning in a custom branch-and-bound
  implementation.
