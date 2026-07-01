#!/bin/bash
# Train (and evaluate) the large AC-OPF cases on SAVIO.
# case1354pegase and case2869pegase have very high-dimensional inputs; 5-fold
# two-phase training is slow on a laptop, so run them on a compute node.
# Run from the project root: bash scripts/submit_acopf_train.sh

CONDA_ENV="nn4opt"
N_TRAIN=20000
N_TEST=5000
FOLDS=4                 # 20000/4 = 5000 val per fold = test-set size
# On SAVIO the generation wrote CSVs to data/acopf (the local pull-back lives in
# data/acopf-hpc, which is the scripts' default — override it here for SAVIO).
DATA_DIR="data/acopf"
CPUS=16                 # torch uses these as BLAS threads (single process)
PARTITION="savio4_htc"
ACCOUNT="fc_power"
TIME="12:00:00"
MEM="32G"               # MLP training is light on memory

# All cases — the whole final training+eval run on the cluster, so nothing
# depends on a laptop staying awake.  Small cases finish in minutes; case2869 is
# SOCP-only (guarded below).
CASES=(case9 case14 case39 case89pegase case118 case300 case1354pegase case2869pegase)

mkdir -p logs

for CASE in "${CASES[@]}"; do
  # case2869pegase is SOCP-only; case1354 has both relaxations.
  if [ "$CASE" = "case2869pegase" ]; then RELAX="socp"; else RELAX="socp chordal_sdp"; fi

  JOB_NAME="train_${CASE}"
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
# (Do NOT pin threads to 1 here — that was only for the multi-worker data-gen.)
export OMP_NUM_THREADS=${CPUS} OPENBLAS_NUM_THREADS=${CPUS} MKL_NUM_THREADS=${CPUS}

python scripts/train_acopf.py --cases ${CASE} --relax ${RELAX} \\
    --data-dir ${DATA_DIR} --n-train ${N_TRAIN} --folds ${FOLDS}

python scripts/evaluate_acopf.py --cases ${CASE} --relax ${RELAX} \\
    --data-dir ${DATA_DIR} --n-train ${N_TRAIN} --n-test ${N_TEST} --folds ${FOLDS}

echo "Finished ${JOB_NAME} at \$(date)"
EOF

  echo "Submitted: ${JOB_NAME} (relax: ${RELAX})"
done
