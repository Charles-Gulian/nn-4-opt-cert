#!/bin/bash
# Submit one SLURM job per (case, relaxation) configuration.
# Run from the project root: bash scripts/submit_acopf_pipeline_batch.sh

SCRIPT="scripts/generate_acopf_data_parallel.py"
CONDA_ENV="nn4opt"
N_WORKERS=56          # CPUs per node
N_TRAIN=10000
N_TEST=5000
SEED=11
CHECKPOINT_EVERY=500
PARTITION="savio4_htc"
ACCOUNT="fc_power"
TIME="24:00:00"
# High-memory allocation: the largest cases peak at ~1.8 GB/worker (case2869
# SOCP), so 56 workers can need ~100 GB.  256 GB leaves ~2.5x margin.  Confirm
# the partition's per-node RAM is >= this before submitting.
MEM="256G"

# Cases to run, and which relaxations each gets.  case2869pegase is SOCP-only:
# the chordal SDP is intractable at that size (see project notes), and SOCP is
# already within ~0.01% of the local optimum there.
CASES=(case9 case14 case39 case89pegase case118 case300 case1354pegase case2869pegase)
RELAXATIONS=(socp chordal_sdp)

# Per-case voltage bound overrides (empty/unset = use pandapower defaults).
# Large cases with tight or relaxation-infeasible default bounds are loosened
# slightly to keep the infeasible-sample rate low.
declare -A V_MIN_MAP=( ["case300"]="0.90" ["case1354pegase"]="0.90" ["case2869pegase"]="0.90" )
declare -A V_MAX_MAP=( ["case300"]="1.10" ["case1354pegase"]="1.10" ["case2869pegase"]="1.10" )

mkdir -p logs

for CASE in "${CASES[@]}"; do
  for RELAX in "${RELAXATIONS[@]}"; do

    # case2869pegase: SOCP only, skip the SDP.
    if [ "$CASE" = "case2869pegase" ] && [ "$RELAX" = "chordal_sdp" ]; then
      continue
    fi

    JOB_NAME="acopf_${CASE}_${RELAX}"

    # Build optional voltage-bound flags for this case
    V_FLAGS=""
    [ -n "${V_MIN_MAP[$CASE]}" ] && V_FLAGS="$V_FLAGS --v-min ${V_MIN_MAP[$CASE]}"
    [ -n "${V_MAX_MAP[$CASE]}" ] && V_FLAGS="$V_FLAGS --v-max ${V_MAX_MAP[$CASE]}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/${JOB_NAME}_%j.out
#SBATCH --error=logs/${JOB_NAME}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${N_WORKERS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}

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
    --checkpoint-every ${CHECKPOINT_EVERY}${V_FLAGS:+ \\
    ${V_FLAGS}}

echo "Finished ${JOB_NAME} at \$(date)"
EOF

    echo "Submitted: ${JOB_NAME}"
  done
done
