#!/bin/bash
# Local pre-flight smoke test for the AC-OPF data-generation pipeline.
#
# Runs every (case, relaxation) config we intend to launch on SAVIO, but with a
# tiny sample count (5 train / 5 test) and a small worker pool, SERIALLY.  The
# point is not to produce usable data — it is to confirm the code runs end to
# end for each case with no exceptions, the worker path + per-case CLARABEL
# options work, output CSVs have the right shape, and at least some solves are
# feasible (non-NaN Cost).
#
# Usage:
#   bash scripts/smoke_test_local.sh
#   PYTHON=/opt/anaconda3/envs/nn4opt/bin/python bash scripts/smoke_test_local.sh
#
# n=5 produces filenames like train_5_socp_case9.csv, which will NOT collide
# with the real n=20000 data.  Pass --clean to delete the smoke CSVs afterward.

set -u
PYTHON="${PYTHON:-python}"
SCRIPT="scripts/generate_acopf_data_parallel.py"
N_TRAIN=5
N_TEST=5
N_WORKERS=2
SEED=11
DATA_DIR="data/acopf"

# config rows: "case relaxation vmin vmax"   (vmin/vmax empty = pandapower default)
CONFIGS=(
  "case9          socp         -    -"
  "case9          chordal_sdp  -    -"
  "case14         socp         -    -"
  "case14         chordal_sdp  -    -"
  "case39         socp         -    -"
  "case39         chordal_sdp  -    -"
  "case89pegase   socp         -    -"
  "case89pegase   chordal_sdp  -    -"
  "case118        socp         -    -"
  "case118        chordal_sdp  -    -"
  "case300        socp         0.90 1.10"
  "case300        chordal_sdp  0.90 1.10"
  "case1354pegase socp         0.90 1.10"
  "case1354pegase chordal_sdp  0.90 1.10"
  "case2869pegase socp         0.90 1.10"
)

PASS=(); FAIL=()

for row in "${CONFIGS[@]}"; do
  read -r CASE RELAX VMIN VMAX <<< "$row"
  tag="${CASE}/${RELAX}"
  echo "================================================================"
  echo ">>> SMOKE: ${tag}  (vmin=${VMIN} vmax=${VMAX})"
  echo "================================================================"

  V_FLAGS=()
  [ "$VMIN" != "-" ] && V_FLAGS+=(--v-min "$VMIN")
  [ "$VMAX" != "-" ] && V_FLAGS+=(--v-max "$VMAX")

  "$PYTHON" "$SCRIPT" \
      --case "$CASE" \
      --relaxation "$RELAX" \
      --n-train "$N_TRAIN" \
      --n-test "$N_TEST" \
      --seed "$SEED" \
      --n-workers "$N_WORKERS" \
      --checkpoint-every 2 \
      --regen \
      "${V_FLAGS[@]}"
  rc=$?

  train_csv="${DATA_DIR}/train_${N_TRAIN}_${RELAX}_${CASE}.csv"
  test_csv="${DATA_DIR}/test_${N_TEST}_${RELAX}_${CASE}.csv"

  # Validate: exit code 0, both CSVs exist with header + N rows, and at least
  # one finite Cost in the test set (proves the solve path actually ran).
  problem=""
  [ $rc -ne 0 ] && problem="exit=$rc"
  if [ -z "$problem" ]; then
    for f in "$train_csv:$N_TRAIN" "$test_csv:$N_TEST"; do
      path="${f%:*}"; want="${f##*:}"
      if [ ! -f "$path" ]; then problem="missing $(basename "$path")"; break; fi
      rows=$(($(wc -l < "$path") - 1))
      [ "$rows" -ne "$want" ] && { problem="$(basename "$path") has $rows rows (want $want)"; break; }
    done
  fi
  if [ -z "$problem" ]; then
    finite=$("$PYTHON" - "$test_csv" <<'PY'
import sys, pandas as pd, numpy as np
df = pd.read_csv(sys.argv[1])
print(int(np.isfinite(df["Cost"]).sum()))
PY
)
    [ "${finite:-0}" -eq 0 ] && problem="all Cost values are NaN/infeasible"
  fi

  if [ -z "$problem" ]; then
    echo "    PASS: ${tag}"
    PASS+=("$tag")
  else
    echo "    FAIL: ${tag}  (${problem})"
    FAIL+=("${tag} (${problem})")
  fi
done

echo "================================================================"
echo "SMOKE SUMMARY:  ${#PASS[@]} passed, ${#FAIL[@]} failed"
for t in "${FAIL[@]}"; do echo "  FAIL: $t"; done

if [ "${1:-}" = "--clean" ]; then
  echo "Cleaning smoke-test artifacts (n=${N_TRAIN}/${N_TEST}) ..."
  rm -f "${DATA_DIR}"/train_${N_TRAIN}_*.csv "${DATA_DIR}"/test_${N_TEST}_*.csv
  rm -f "${DATA_DIR}"/X_train_${N_TRAIN}_*.npy "${DATA_DIR}"/X_test_${N_TEST}_*.npy
fi

[ "${#FAIL[@]}" -eq 0 ]
