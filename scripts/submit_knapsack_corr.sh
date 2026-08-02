#!/bin/bash
# Submit the remaining correlated-mu knapsack training runs to SAVIO as a job
# array (one task per config), run in parallel instead of serially on a laptop.
#
# knapsack_corr_exact_20k (the headline 20k/exact result) already completed
# locally and is NOT re-run here. This array covers:
#   1. relax_20k, exact_80k, relax_80k        -- the remaining headline rows
#   2. capacity ablation (appendix), at n=20k/exact:
#      6x512, 10x256, 8x512  (6x256 is the completed headline run)
#
# No optimization solves happen on SAVIO -- data/knapsack_corr/{pool_80000.npz,
# test_5000.npz} must already exist (rsync'd up) and are read-only inputs;
# scripts/knapsack_corr_kfold.py only trains + calibrates.
#
# Usage (from the project root, on SAVIO):
#     bash scripts/submit_knapsack_corr.sh
#     bash scripts/submit_knapsack_corr.sh --dry-run

set -u
CONDA_ENV="nn4opt"
PARTITION="savio4_htc"
ACCOUNT="fc_power"
TIME="04:00:00"
MEM="16G"
CPUS=6
LOG_DIR="logs/knapsack_corr"

# each entry: target:n-pool:extra-args (extra-args may be empty, use "-")
TASKS=(
  "relax:20000:-"
  "exact:80000:-"
  "relax:80000:-"
  "exact:20000:--hidden-dims 512 512 512 512 512 512 --suffix _w512"
  "exact:20000:--hidden-dims 256 256 256 256 256 256 256 256 256 256 --suffix _d10"
  "exact:20000:--hidden-dims 512 512 512 512 512 512 512 512 --suffix _d8w512"
)
N_TASKS=${#TASKS[@]}

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "Would submit ${N_TASKS} tasks:"
  for (( i=0; i<N_TASKS; i++ )); do
    IFS=: read tgt npool extra <<< "${TASKS[$i]}"
    printf "  [%d] target=%-6s n-pool=%-6s extra=%s\n" "$i" "$tgt" "$npool" "$extra"
  done
  exit 0
fi

mkdir -p "$LOG_DIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=knap-corr
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --array=0-$(( N_TASKS - 1 ))
#SBATCH --output=${LOG_DIR}/knap_%A_%a.out
#SBATCH --error=${LOG_DIR}/knap_%A_%a.err

source ~/.bashrc
conda activate ${CONDA_ENV}
cd \$SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=${CPUS}
export MKL_NUM_THREADS=${CPUS}

TASKS=($(printf '"%s" ' "${TASKS[@]}"))
IFS=: read TGT NPOOL EXTRA <<< "\${TASKS[\$SLURM_ARRAY_TASK_ID]}"
[[ "\$EXTRA" == "-" ]] && EXTRA=""

echo "task \$SLURM_ARRAY_TASK_ID: target=\$TGT n-pool=\$NPOOL extra=\$EXTRA"
python scripts/knapsack_corr_kfold.py --target "\$TGT" --n-pool "\$NPOOL" \$EXTRA
EOF
