#!/bin/bash
# Submit one SLURM job per (case, relaxation) configuration.
# Run from the project root: bash scripts/submit_acopf_data.sh

SCRIPT="scripts/generate_acopf_data_parallel.py"
PYTHON="/path/to/conda/envs/nn4opt/bin/python"   # <-- update to your Savio conda env path
N_WORKERS=32          # CPUs per node on savio3
N_TRAIN=10000
N_TEST=5000
SEED=42
PARTITION="savio3"    # adjust to your target partition
TIME="08:00:00"

for CASE in case9 case14 case39; do
  for RELAX in socp sdp; do

    # SDP on case39 needs more time; give it extra headroom
    if [[ "$CASE" == "case39" && "$RELAX" == "sdp" ]]; then
      TIME_THIS="12:00:00"
    else
      TIME_THIS="$TIME"
    fi

    JOB_NAME="acopf_${CASE}_${RELAX}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/${JOB_NAME}_%j.out
#SBATCH --error=logs/${JOB_NAME}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${N_WORKERS}
#SBATCH --time=${TIME_THIS}
#SBATCH --partition=${PARTITION}

echo "Starting ${JOB_NAME} on \$(hostname) at \$(date)"
echo "CPUs allocated: \${SLURM_CPUS_PER_TASK}"

$PYTHON $SCRIPT \\
    --case ${CASE} \\
    --relaxation ${RELAX} \\
    --n-train ${N_TRAIN} \\
    --n-test ${N_TEST} \\
    --seed ${SEED} \\
    --n-workers ${N_WORKERS}

echo "Finished ${JOB_NAME} at \$(date)"
EOF

    echo "Submitted: ${JOB_NAME}"
  done
done
