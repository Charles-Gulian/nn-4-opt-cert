#!/bin/bash
# Submit a single data generation job for one (case, relaxation) config.
# Run from the project root: bash scripts/submit_acopf_data_gen.sh
#
# Edit the variables below before submitting.

# ── job configuration ─────────────────────────────────────────────────────────
CASE="case14"          # case9 | case14 | case39 | case89pegase | case118 | case300 | case1354pegase
RELAX="socp"           # socp  | chordal_sdp | sdp

N_TRAIN=10000
N_TEST=5000
SEED=11
N_WORKERS=56
CHECKPOINT_EVERY=500

# Voltage bounds override (leave empty to use pandapower case defaults)
# For case300 use: V_MIN=0.90  V_MAX=1.10
V_MIN=""
V_MAX=""

PARTITION="savio4_htc"
ACCOUNT="fc_power"
TIME="24:00:00"
CONDA_ENV="nn4opt"
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT="scripts/generate_acopf_data_parallel.py"
JOB_NAME="acopf_${CASE}_${RELAX}"

# Build optional voltage-bound flags
V_FLAGS=""
[ -n "$V_MIN" ] && V_FLAGS="$V_FLAGS --v-min ${V_MIN}"
[ -n "$V_MAX" ] && V_FLAGS="$V_FLAGS --v-max ${V_MAX}"

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
    --checkpoint-every ${CHECKPOINT_EVERY}${V_FLAGS:+ \\
    ${V_FLAGS}}

echo "Finished ${JOB_NAME} at \$(date)"
EOF

echo "Submitted: ${JOB_NAME}"
