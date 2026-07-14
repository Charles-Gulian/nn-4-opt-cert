#!/bin/bash
# Retrain + per-fold certify the three PEGASE SOCP configs after the Jabr
# phase-shifter data fix.  Only these three configs changed (their SOCP labels
# were regenerated); every other AC-OPF config's data/checkpoints are untouched
# and do NOT need retraining.
#
# Each job, for one case: trains the 4 fold checkpoints on the corrected data,
# then runs the per-fold conformal calibration + certification
# (scripts/evaluate_certify.py) that produces the paper result CSVs.
#
# Run from the project root on a login node:
#     bash scripts/submit_acopf_socp_pegase_train.sh

CONDA_ENV="nn4opt"
N_TRAIN=20000
N_TEST=5000
FOLDS=4
# SAVIO generation wrote CSVs to data/acopf (the scripts' laptop default is
# data/acopf-hpc); train_acopf.py needs the explicit override.  evaluate_certify
# reads via the registry, which auto-prefers data/acopf when it exists.
DATA_DIR="data/acopf"
CPUS=16                 # torch BLAS threads (single-process training)
PARTITION="savio4_htc"
ACCOUNT="fc_power"
TIME="12:00:00"
MEM="32G"

CASES=(case89pegase case1354pegase case2869pegase)

mkdir -p logs

for CASE in "${CASES[@]}"; do
  JOB_NAME="train_certify_${CASE}_socp"
  KEY="acopf_socp_${CASE}"
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/${JOB_NAME}_%j.out
#SBATCH --error=logs/${JOB_NAME}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}

echo "Starting ${JOB_NAME} on \$(hostname) at \$(date)"

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

# Single-process training: let torch use all allocated cores for BLAS.
export OMP_NUM_THREADS=${CPUS} OPENBLAS_NUM_THREADS=${CPUS} MKL_NUM_THREADS=${CPUS}

# 1) Train the 4 fold checkpoints on the corrected SOCP data.
python scripts/train_acopf.py --cases ${CASE} --relax socp \\
    --data-dir ${DATA_DIR} --n-train ${N_TRAIN} --folds ${FOLDS}

# 2) Per-fold conformal calibration + certification -> result CSVs.
python scripts/evaluate_certify.py --key ${KEY} \\
    --n-train ${N_TRAIN} --n-test ${N_TEST} --folds ${FOLDS}

echo "Finished ${JOB_NAME} at \$(date)"
EOF

  echo "Submitted: ${JOB_NAME} (key: ${KEY})"
done
