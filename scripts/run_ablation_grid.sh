#!/bin/bash
# One-factor-at-a-time AC-OPF ablation grid around the reference config
# (6 hidden layers x 256 units, 1000 epochs, n=20k, 4-fold, SOCP).
#
#   depth  : 1..6          @ width 256, 1000 epochs
#   epochs : 100..2000     @ 6x256           (1000 supplied by the depth sweep)
#   width  : 64,128,512    @ depth 6, 1000 ep (256 supplied by the depth sweep)
#
# Every run uses the SINGLE-PHASE cosine schedule of the main pipeline
# (train_generic.py default), so T_max == the epoch budget being ablated --
# each budget is a genuinely separate run, not a checkpoint of a longer one.
#
# Usage: bash scripts/run_ablation_grid.sh <case> <threads>
set -u
CASE="$1"
THREADS="${2:-2}"
PY=/opt/anaconda3/envs/nn4opt/bin/python
OUT=results/acopf/ablation_summary.csv
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"

run () {   # run <depths> <widths> <epochs>
  echo "### $CASE depths=$1 widths=$2 epochs=$3" >&2
  $PY scripts/ablation_acopf.py --cases "$CASE" --relax socp \
      --depths $1 --widths $2 --folds 4 --n-train 20000 \
      --single-phase --pretrain-epochs $3 --finetune-epochs 0 \
      --out "$OUT"
}

# depth sweep (also supplies the 6x256@1000 reference row)
run "1 2 3 4 5 6" "256" 1000
# training-budget sweep at the reference architecture
for E in 100 200 500 1500 2000; do run "6" "256" $E; done
# width sweep at the reference depth/budget
run "6" "64 128 512" 1000

echo "ABLATION_GRID_DONE_$CASE"
