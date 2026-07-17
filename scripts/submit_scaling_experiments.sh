#!/bin/bash
# Data-scaling experiments on SAVIO: push MIMO and robust knapsack to large
# training sets to see how much certification improves for these hard value
# functions. Submits three independent jobs:
#
#   1. MIMO      n_train = 80,000    (generic pipeline: gen -> train -> certify)
#   2. MIMO      n_train = 640,000   (32x the base 20k)
#   3. Knapsack  n_train = 640,000   (32x the base 20k; Gurobi gen is serial ~7h)
#
# All sizes reuse the SAME test set (MIMO test_5000, knapsack test_20000) so only
# the training size varies -- a clean data-scaling comparison. MIMO certification
# outputs are copied to results/mimo_n<N>/ so they don't clobber the n=20000 run.
# Knapsack needs no results dir: the table scripts load its checkpoints + OOF
# residuals by n_train directly.
#
# Run from the project root on a login node:  bash scripts/submit_scaling_experiments.sh
# Pass one or more keys as args to submit only a subset, e.g. to resubmit just
# the knapsack job after fixing an environment issue:
#   bash scripts/submit_scaling_experiments.sh knapsack
# (matches the JOBS "key" field below; "mimo" would submit both mimo rows)
#
# COMPUTE NOTES: we keep the EXACT standard training recipe (train_*.py defaults:
# 1000 epochs, batch 256, deep 6x256) at every size -- no recipe changes to
# explain. Training is CPU-only; at 640k this is ~8-9h for the 4 folds (the net
# is small, ~3 ms/step), which fits inside the 24h wall. MIMO SDP generation
# parallelizes over N_WORKERS (<1h). Knapsack Gurobi generation was assumed
# ~39ms/solve but measured ~150ms/solve on SAVIO's academic WLS license --
# serial would take ~26h at 640k, missing the 24h wall. It now supports
# --n-workers to parallelize across processes (each with its own Gurobi Env).
#
# KNAPSACK PARALLELISM: the bottleneck was CPU oversubscription, NOT the Gurobi
# license. Each knapsack instance is a MISOCP solved by branch-and-bound, which
# defaults to Threads=0 (all cores). Running K workers each with all-core B&B on
# a shared core pool caused ~90x per-solve slowdown (0.15s -> 14s). The fix is
# to cap Gurobi Threads per worker so KNAPSACK_WORKERS * KNAPSACK_THREADS <=
# CPUS (no oversubscription). Measured: 8 workers all succeed with no session
# errors (the earlier "~5 ceiling" was phantom sessions from rapid probing, not
# a real cap), ~0.13-0.19s/solve, effective ~0.016s/row => 640k ~= 3h. The
# generator still retries GurobiError 10030 with backoff and aborts loudly
# rather than writing NaN.

CONDA_ENV="nn4opt"
PARTITION="savio4_htc"
ACCOUNT="fc_power"
N_WORKERS=32
KNAPSACK_WORKERS=8    # 8 concurrent Gurobi sessions (verified OK, no 10030 errors)
KNAPSACK_THREADS=4    # 8 * 4 = 32 = CPUS -> no core oversubscription
CPUS=32
FOLDS=4
mkdir -p logs

# (key  n_train  mem  time)   -- training uses the standard recipe (script defaults)
JOBS=(
  "mimo      80000   64G  12:00:00"
  "mimo      640000  96G  24:00:00"
  "knapsack  640000  96G  24:00:00"
)

# Optional filter: only submit rows whose key (e.g. "mimo", "knapsack") or
# "key:n_train" (e.g. "mimo:640000") matches one of the given args.
FILTERS=("$@")

for spec in "${JOBS[@]}"; do
  read -r KEY N MEM TIME <<< "$spec"
  if [ ${#FILTERS[@]} -gt 0 ]; then
    MATCH=0
    for f in "${FILTERS[@]}"; do
      if [ "$f" = "${KEY}" ] || [ "$f" = "${KEY}:${N}" ]; then
        MATCH=1
      fi
    done
    if [ "$MATCH" -eq 0 ]; then
      continue
    fi
  fi
  JOB="scale_${KEY}_n${N}"
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB}
#SBATCH --output=logs/${JOB}_%j.out
#SBATCH --error=logs/${JOB}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}

echo "Starting ${JOB} on \$(hostname) at \$(date)"
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}
export PATH="\$(conda info --base)/envs/${CONDA_ENV}/bin:\$PATH"

if [ "${KEY}" = "knapsack" ]; then
    # ---- Robust knapsack: dedicated pipeline (Gurobi gen, parallel+checkpointed) ----
    echo "[1/3] generating data (Gurobi, ${KNAPSACK_WORKERS} workers x ${KNAPSACK_THREADS} threads, resumable)"
    python scripts/generate_knapsack_data.py --n-train ${N} --n-test 20000 \\
        --n-workers ${KNAPSACK_WORKERS} --threads ${KNAPSACK_THREADS}

    export OMP_NUM_THREADS=${CPUS} OPENBLAS_NUM_THREADS=${CPUS} MKL_NUM_THREADS=${CPUS}
    echo "[2/3] training 4-fold deep nets (standard recipe)"
    python scripts/train_knapsack.py --n-train ${N} --folds ${FOLDS}

    echo "[3/3] calibrating OOF residuals"
    python scripts/calibrate_knapsack_conformal.py --n-train ${N} --folds ${FOLDS}
else
    # ---- MIMO (and any generic small problem): gen -> train -> certify ----
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1
    echo "[1/3] generating data (${N_WORKERS} workers)"
    python scripts/generate_dataset.py --key ${KEY} \\
        --n-train ${N} --n-test 5000 --n-workers ${N_WORKERS} \\
        --seed-train 0 --seed-test 1

    export OMP_NUM_THREADS=${CPUS} OPENBLAS_NUM_THREADS=${CPUS} MKL_NUM_THREADS=${CPUS}
    echo "[2/3] training 4-fold deep nets (standard recipe)"
    python scripts/train_generic.py --key ${KEY} \\
        --n-train ${N} --n-test 5000 --folds ${FOLDS}

    echo "[3/3] per-fold calibration + certification"
    python scripts/evaluate_certify.py --key ${KEY} \\
        --n-train ${N} --n-test 5000 --folds ${FOLDS}

    # Preserve this size's outputs (evaluate_certify writes to results/${KEY}).
    DEST="results/${KEY}_n${N}"; mkdir -p "\$DEST"
    cp results/${KEY}/{fold_test_predictions,val_residuals,conformal_offsets,fold_test_metrics,certification}.csv "\$DEST"/ 2>/dev/null
    echo "copied ${KEY} n=${N} results -> \$DEST"
fi

echo "Finished ${JOB} at \$(date)"
EOF
  echo "Submitted: ${JOB}"
done
