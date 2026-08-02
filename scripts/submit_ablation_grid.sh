#!/bin/bash
# Submit the full AC-OPF architecture / training-budget ablation grid to SAVIO
# as a job ARRAY (one task per configuration), for an overnight run.
#
# Grid: one-factor-at-a-time around the reference config used in the paper
# (6 hidden layers x 256 units, 1000 epochs, n=20k, 4-fold, SOCP):
#     depth  1..6            @ width 256, 1000 epochs
#     epochs 100,200,500,1500,2000 @ 6x256   (1000 comes from the depth sweep)
#     width  64,128,512      @ depth 6, 1000 epochs  (256 from the depth sweep)
# for case9, case118, case1354pegase  ->  3 x 14 = 42 tasks.
#
# Every task uses the SINGLE-PHASE cosine schedule of the main pipeline
# (train_generic.py default), so T_max equals the epoch budget being ablated --
# each budget is a genuinely separate run, never a checkpoint of a longer one.
#
# IMPORTANT: tasks run concurrently, so each writes its OWN part file under
# results/acopf/ablation_parts/ (ablation_acopf.py's upsert is not concurrency
# safe). Merge afterwards with:
#     python scripts/merge_ablation_parts.py
#
# Usage (from the project root, on SAVIO):
#     bash scripts/submit_ablation_grid.sh
#     bash scripts/submit_ablation_grid.sh --dry-run     # print the task table

set -u
CONDA_ENV="nn4opt"
DATA_DIR="data/acopf"          # SAVIO-side data location
PARTITION="savio4_htc"
ACCOUNT="fc_power"
TIME="20:00:00"
MEM="16G"
CPUS=8
FOLDS=4
N_TRAIN=20000
OUT_DIR="results/acopf/ablation_parts"
LOG_DIR="logs/ablation"

CASES=(case9 case118 case1354pegase)
# each entry: depth:width:epochs
CONFIGS=(
  1:256:1000 2:256:1000 3:256:1000 4:256:1000 5:256:1000 6:256:1000
  6:256:100  6:256:200  6:256:500  6:256:1500 6:256:2000
  6:64:1000  6:128:1000 6:512:1000
)

N_CASES=${#CASES[@]}
N_CONF=${#CONFIGS[@]}
N_TASKS=$(( N_CASES * N_CONF ))

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "Would submit ${N_TASKS} tasks (${N_CASES} cases x ${N_CONF} configs):"
  for (( i=0; i<N_TASKS; i++ )); do
    c=${CASES[$(( i / N_CONF ))]}; IFS=: read d w e <<< "${CONFIGS[$(( i % N_CONF ))]}"
    printf "  [%2d] %-16s depth=%s width=%-4s epochs=%s\n" "$i" "$c" "$d" "$w" "$e"
  done
  exit 0
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=abl-grid
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --array=0-$(( N_TASKS - 1 ))%14
#SBATCH --output=${LOG_DIR}/abl_%A_%a.out
#SBATCH --error=${LOG_DIR}/abl_%A_%a.err

source ~/.bashrc
conda activate ${CONDA_ENV}
cd \$SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=${CPUS}
export MKL_NUM_THREADS=${CPUS}

CASES=(${CASES[@]})
CONFIGS=(${CONFIGS[@]})
N_CONF=${N_CONF}

IDX=\$SLURM_ARRAY_TASK_ID
CASE=\${CASES[\$(( IDX / N_CONF ))]}
IFS=: read DEPTH WIDTH EPOCHS <<< "\${CONFIGS[\$(( IDX % N_CONF ))]}"

echo "task \$IDX: case=\$CASE depth=\$DEPTH width=\$WIDTH epochs=\$EPOCHS"

python scripts/ablation_acopf.py \\
    --cases "\$CASE" --relax socp \\
    --depths "\$DEPTH" --widths "\$WIDTH" \\
    --folds ${FOLDS} --n-train ${N_TRAIN} \\
    --data-dir ${DATA_DIR} \\
    --single-phase --pretrain-epochs "\$EPOCHS" --finetune-epochs 0 \\
    --out ${OUT_DIR}/part_\${IDX}.csv
EOF
