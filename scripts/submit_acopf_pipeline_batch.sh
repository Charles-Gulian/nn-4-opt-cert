#!/bin/bash
# Submit one SLURM job per (case, relaxation) configuration.
# Run from the project root: bash scripts/submit_acopf_pipeline_batch.sh

SCRIPT="scripts/generate_acopf_data_parallel.py"
CONDA_ENV="nn4opt"
N_WORKERS=56          # CPUs per node (savio4_htc nodes have 56)
N_TRAIN=20000
N_TEST=5000
SEED=343
CHECKPOINT_EVERY=500
PARTITION="savio4_htc"
ACCOUNT="fc_power"
TIME="24:00:00"
# Memory: the 4-worker steady-state footprint (~0.5 GB/worker) badly under-
# predicts the real peak, because all 56 workers initialise at once and the
# chordal_sdp path re-canonicalises on EVERY solve (ignore_dpp), so up to 56
# multi-GB canonicalization transients overlap.  A 64 GB cap OOM-killed
# case1354 chordal.  savio4_htc nodes have 257 GB (some 515 GB); request 200 GB
# (fits the 257 GB nodes, leaves ample room for concurrent canonicalization).
MEM="200G"

# Dry-run mode: `DRY_RUN=1 bash scripts/submit_acopf_pipeline_batch.sh` submits
# every config with just 5 train / 5 test samples and a short wall clock, to
# validate the full pipeline on SAVIO before the real run.
if [ "${DRY_RUN:-0}" = "1" ]; then
  N_TRAIN=5
  N_TEST=5
  CHECKPOINT_EVERY=2
  TIME="01:00:00"
  echo "*** DRY RUN: ${N_TRAIN} train / ${N_TEST} test, time=${TIME} ***"
fi

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

# One thread per worker process — we run one worker per core, so every threaded
# layer must be capped to 1 to avoid ~56x56 oversubscription.  BLAS/OpenMP covers
# numpy/scipy canonicalization; RAYON covers CLARABEL's Rust solver thread pool.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1

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
