#!/bin/bash
# Correlated-mu robust knapsack: all training runs, in priority order.
# Requires data/knapsack_corr/{pool_80000.npz,test_5000.npz}
# (scripts/generate_knapsack_corr_data.py). No optimization solves happen here.
#
#   1. headline 4-fold runs: {exact, relax} x {20k, 80k}  -> results tables
#   2. capacity ablation at 20k/exact: 6x256 (= headline), 6x512, 10x256, 8x512
#      -> appendix. Together with the 20k-vs-80k data scaling this is the
#      evidence that the certification tail is irreducible: neither more data
#      nor more capacity moves q_99.
#
# 20k configs come first so the main tables can be built before the slower
# 80k runs finish.
#
# Usage: bash scripts/run_knapsack_corr_all.sh
set -u
PY=/opt/anaconda3/envs/nn4opt/bin/python
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

kf () { echo "### $*" >&2; $PY scripts/knapsack_corr_kfold.py "$@"; }

# 1. headline runs
kf --target exact --n-pool 20000
kf --target relax --n-pool 20000
kf --target exact --n-pool 80000
kf --target relax --n-pool 80000

# 2. capacity ablation (appendix): does more capacity shrink the tail?
kf --target exact --n-pool 20000 --hidden-dims 512 512 512 512 512 512      --suffix _w512
kf --target exact --n-pool 20000 --hidden-dims 256 256 256 256 256 256 256 256 256 256 --suffix _d10
kf --target exact --n-pool 20000 --hidden-dims 512 512 512 512 512 512 512 512       --suffix _d8w512

echo "KNAPSACK_CORR_ALL_DONE"
