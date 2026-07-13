#!/bin/bash
# Regenerate SOCP data for the three PEGASE cases after the Jabr phase-shifter
# fix (problems/acopf/problem.py: direction-specific Y[to,fr] in the to-bus
# injection).  These are the only cases whose SOCP labels were wrong: their
# Ybus is asymmetric because of phase-shifting transformers, which the old
# symmetric construction mis-modeled, producing invalid lower bounds
# (v_r > f).  All other SOCP cases are unaffected (the fix is a bitwise no-op
# where Ybus is symmetric) and do NOT need regeneration; the chordal SDP build
# already handled the asymmetry and is likewise untouched.
#
# Submits one SLURM job per case.  Run from the project root on a login node:
#     bash scripts/submit_acopf_socp_pegase_regen.sh
#
# Seed 343 (train) / 344 (test) and the per-case voltage bounds below are
# copied from scripts/submit_acopf_pipeline_batch.sh so the regenerated data
# samples the SAME parameter points as the original run — only the labels change.
#
# Optional pre-flight (recommended) on a compute node before trusting a run:
#     python scripts/check_socp_lower_bound.py

SCRIPT="scripts/generate_acopf_data_parallel.py"
CONDA_ENV="nn4opt"
N_WORKERS=56
N_TRAIN=20000
N_TEST=5000
SEED=343
CHECKPOINT_EVERY=500
PARTITION="savio4_htc"
ACCOUNT="fc_power"
TIME="24:00:00"
MEM="200G"

# Only the phase-shifter (PEGASE) SOCP cases.
CASES=(case89pegase case1354pegase case2869pegase)

# Per-case voltage-bound overrides — MUST match the original generation
# (submit_acopf_pipeline_batch.sh) or the parameter set / feasibility changes.
# case89pegase used pandapower defaults; the two larger cases were loosened.
declare -A V_MIN_MAP=( ["case1354pegase"]="0.90" ["case2869pegase"]="0.90" )
declare -A V_MAX_MAP=( ["case1354pegase"]="1.10" ["case2869pegase"]="1.10" )

mkdir -p logs

for CASE in "${CASES[@]}"; do
  JOB_NAME="acopf_${CASE}_socp_regen"

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

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

# One thread per worker process (one worker per core) to avoid oversubscription;
# RAYON caps CLARABEL's Rust thread pool, the BLAS vars cap canonicalization.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1

# --regen deletes the stale (buggy-label) CSVs and re-labels from scratch.
python ${SCRIPT} \\
    --case ${CASE} \\
    --relaxation socp \\
    --n-train ${N_TRAIN} \\
    --n-test ${N_TEST} \\
    --seed ${SEED} \\
    --n-workers ${N_WORKERS} \\
    --checkpoint-every ${CHECKPOINT_EVERY} \\
    --regen${V_FLAGS:+ \\
    ${V_FLAGS}}

echo "Finished ${JOB_NAME} at \$(date)"
EOF

  echo "Submitted: ${JOB_NAME}"
done
