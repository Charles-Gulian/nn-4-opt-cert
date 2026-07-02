#!/bin/bash
# Per-fold certification ONLY for AC-OPF, on SAVIO -- no data regeneration, no
# retraining. Reuses the fold checkpoints and CSVs that already live on the
# cluster (models/acopf/, data/acopf-hpc/), so this is a single lightweight
# CPU job that reads existing artifacts and writes small result CSVs.
#
# Run from the project root: bash scripts/hpc/submit_acopf_certify.sh
#
# IMPORTANT: before submitting, confirm the final n=20000 4-fold checkpoints
# for case300 / case1354pegase / case2869pegase actually exist at
#   models/acopf/dnn_{relax}_{case}_n20000_fold{0-3}.pt
# on THIS cluster filesystem (the local laptop only has case9-118 checkpoints;
# the bigger cases were only ever finalized on SAVIO). If any are missing,
# retrain them first with scripts/train_acopf.py before running this.

N_TRAIN=20000
N_TEST=5000
FOLDS=4

CPUS=4          # CPU-light: just loads checkpoints and predicts, no solving
PARTITION="savio4_htc"
ACCOUNT="fc_power"
TIME="02:00:00"
MEM="16G"
CONDA_ENV="nn4opt"

mkdir -p logs
JOB_NAME="acopf_certify_all"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/${JOB_NAME}_%j.out
#SBATCH --error=logs/${JOB_NAME}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}

echo "Starting ${JOB_NAME} on \$(hostname) at \$(date)"

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

# Certify one case+relaxation at a time (rather than --all-acopf in one call)
# so a single missing/corrupt checkpoint only skips that config, not the batch.
python - <<'PYEOF'
import subprocess, sys
sys.path.insert(0, ".")
from problems.registry import acopf_keys

failures = []
for key in acopf_keys():
    print(f"\n### {key} ###", flush=True)
    rc = subprocess.run([
        "python", "scripts/evaluate_certify.py", "--key", key,
        "--n-train", "${N_TRAIN}", "--n-test", "${N_TEST}", "--folds", "${FOLDS}",
    ]).returncode
    if rc != 0:
        failures.append(key)

print(f"\nDone. {len(failures)} failures: {failures}")
PYEOF

echo "Finished ${JOB_NAME} at \$(date)"
EOF

echo "Submitted: ${JOB_NAME}"
