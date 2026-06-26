#!/bin/bash
# Submit a single data generation job for one (case, relaxation) config.
# Run from the project root: bash scripts/submit_acopf_data_gen.sh
#
# Edit the variables below before submitting.

# ── job configuration ─────────────────────────────────────────────────────────
CASE="case14"          # case9 | case14 | case39
RELAX="socp"           # socp  | sdp

N_TRAIN=10000
N_TEST=5000
SEED=11
N_WORKERS=56
CHECKPOINT_EVERY=500

PARTITION="savio4_htc"
ACCOUNT="fc_power"
TIME="24:00:00"
CONDA_ENV="nn4opt"
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT="scripts/generate_acopf_data_parallel.py"
JOB_NAME="acopf_${CASE}_${RELAX}"

mkdir -p logs

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/${JOB_NAME}_%j.out
#SBATCH --error=logs/${JOB_NAME}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${N_WORKERS}
#SBATCH --time=${TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}

echo "Starting ${JOB_NAME} on \$(hostname) at \$(date)"
echo "CPUs allocated: \${SLURM_CPUS_PER_TASK}"

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

python ${SCRIPT} \\
    --case ${CASE} \\
    --relaxation ${RELAX} \\
    --n-train ${N_TRAIN} \\
    --n-test ${N_TEST} \\
    --seed ${SEED} \\
    --n-workers ${N_WORKERS} \\
    --checkpoint-every ${CHECKPOINT_EVERY}

echo "Finished ${JOB_NAME} at \$(date)"
EOF

echo "Submitted: ${JOB_NAME}"
