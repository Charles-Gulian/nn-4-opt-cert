#!/bin/bash
# Follow-up ablation: the width sweep (submit_ablation_grid.sh) showed width=64
# beating the paper's default width=256 on the certification-relevant tail
# metric (q95_overpred_pct) at EVERY case tested, not just on mean MAPE -- e.g.
# case118 tail error nearly doubles going 64->256, and 6x256 is not even the
# best DEPTH by this metric on case118 (depth=5 beats depth=6). Before deciding
# whether to change the paper's default architecture (which would mean re-running
# every other experiment), we re-run the depth and epoch axes AT width=64, to
# see whether the same "smaller is better" pattern holds there too, or whether
# width=64 was a fluke specific to depth=6/epochs=1000.
#
# Grid (single-phase, n=20k, 4-fold, SOCP), width FIXED at 64:
#     depth  1,2,3,4,5,6            @ 1000 epochs
#     epochs 500,1000,1500,2000     @ depth 6
# for case9, case118, case1354pegase.
#
# depth=6/epochs=1000/width=64 was already computed by the original width sweep
# (results/acopf/ablation_summary.csv) and is EXCLUDED here to avoid redundant
# compute; scripts/merge_ablation_parts.py will combine this run with the
# existing point by the same (case, depth, width, epochs) key.
#
# Usage (from the project root, on SAVIO):
#     bash scripts/submit_ablation_width64.sh
#     bash scripts/submit_ablation_width64.sh --dry-run

set -u
CONDA_ENV="nn4opt"
DATA_DIR="data/acopf"
PARTITION="savio4_htc"
ACCOUNT="fc_power"
TIME="20:00:00"
MEM="16G"
CPUS=8
FOLDS=4
N_TRAIN=20000
OUT_DIR="results/acopf/ablation_parts"
LOG_DIR="logs/ablation_width64"

CASES=(case9 case118 case1354pegase)
# each entry: depth:width:epochs  (width fixed at 64; d6/e1000 excluded, already have it)
CONFIGS=(
  1:64:1000 2:64:1000 3:64:1000 4:64:1000 5:64:1000
  6:64:500  6:64:1500 6:64:2000
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
#SBATCH --job-name=abl-w64
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
    --out ${OUT_DIR}/part_w64_\${IDX}.csv
EOF
