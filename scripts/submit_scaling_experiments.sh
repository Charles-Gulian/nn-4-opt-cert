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
#
# COMPUTE NOTES: we keep the EXACT standard training recipe (train_*.py defaults:
# 1000 epochs, batch 256, deep 6x256) at every size -- no recipe changes to
# explain. Training is CPU-only; at 640k this is ~8-9h for the 4 folds (the net
# is small, ~3 ms/step), which fits inside the 24h wall. MIMO SDP generation
# parallelizes over N_WORKERS (<1h); knapsack Gurobi generation is serial
# (~39 ms/solve ⇒ ~7h for 640k) but checkpointed/resumable.

CONDA_ENV="nn4opt"
PARTITION="savio4_htc"
ACCOUNT="fc_power"
N_WORKERS=32
CPUS=32
FOLDS=4
mkdir -p logs

# (key  n_train  mem  time)   -- training uses the standard recipe (script defaults)
JOBS=(
  "mimo      80000   64G  12:00:00"
  "mimo      640000  96G  24:00:00"
  "knapsack  640000  96G  24:00:00"
)

for spec in "${JOBS[@]}"; do
  read -r KEY N MEM TIME <<< "$spec"
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
    # ---- Robust knapsack: dedicated pipeline (Gurobi gen, serial+checkpointed) ----
    echo "[1/3] generating data (Gurobi, serial, resumable)"
    python scripts/generate_knapsack_data.py --n-train ${N} --n-test 20000

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
