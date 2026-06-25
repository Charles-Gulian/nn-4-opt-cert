#!/bin/bash
# Submit one SLURM job per (case, relaxation) configuration.
# Run from the project root: bash scripts/submit_acopf_data.sh

SCRIPT="scripts/generate_acopf_data_parallel.py"
CONDA_ENV="nn4opt"
N_WORKERS=56          # CPUs per node on savio3
N_TRAIN=10000
N_TEST=5000
SEED=11
PARTITION="savio4_htc"    # adjust to your target partition
TIME="24:00:00"

for CASE in case9 case14 case39; do
  for RELAX in socp sdp; do

    JOB_NAME="acopf_${CASE}_${RELAX}"

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

echo "Starting ${JOB_NAME} on \$(hostname) at \$(date)"
echo "CPUs allocated: \${SLURM_CPUS_PER_TASK}"

# Activate conda — source the init script so 'conda activate' works in bash
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

python $SCRIPT \\
    --case ${CASE} \\
    --relaxation ${RELAX} \\
    --n-train ${N_TRAIN} \\
    --n-test ${N_TEST} \\
    --seed ${SEED} \\
    --n-workers ${N_WORKERS} \\
    --checkpoint-evert 500

echo "Finished ${JOB_NAME} at \$(date)"
EOF

    echo "Submitted: ${JOB_NAME}"
  done
done
