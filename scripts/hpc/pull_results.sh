#!/bin/bash
# Pull the small, portable per-fold result CSVs back from SAVIO. By default this
# excludes the two large intermediate artifacts (fold_test_predictions.csv,
# val_residuals.csv) so a routine sync stays fast -- pass --full to also pull
# those (e.g. once, to do local box-and-whisker plotting of the test-error
# distribution).
#
# Usage:
#   bash scripts/hpc/pull_results.sh user@dtn.brc.berkeley.edu:/path/to/nn-4-opt-cert
#   bash scripts/hpc/pull_results.sh user@dtn.brc.berkeley.edu:/path/to/nn-4-opt-cert --full

set -euo pipefail

REMOTE="${1:?usage: pull_results.sh <user@host:/remote/project/root> [--full]}"
FULL="${2:-}"

EXCLUDES=()
if [ "$FULL" != "--full" ]; then
  EXCLUDES=(--exclude "fold_test_predictions.csv" --exclude "val_residuals.csv")
  echo "Pulling small summary CSVs only (pass --full to include large intermediates)"
else
  echo "Pulling ALL result artifacts, including large per-instance prediction files"
fi

rsync -avz "${EXCLUDES[@]}" \
    "${REMOTE}/results/" \
    "./results/"

echo "Done."
