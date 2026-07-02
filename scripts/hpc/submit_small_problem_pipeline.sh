#!/bin/bash
# Full gen -> train -> certify pipeline for ONE small-problem config, on SAVIO.
# Run from the project root: bash scripts/hpc/submit_small_problem_pipeline.sh
#
# Edit KEY below before submitting. Covers: qcqp | mimo | ik_lass1 | ik_lass2.
# ik_lass2 is the slow one (~1-5s/SDP solve x 25k instances) -- this is why the
# whole pipeline runs on the cluster with N_WORKERS parallelizing generation,
# rather than locally. Everything else finishes in minutes even serially.

# ── job configuration ─────────────────────────────────────────────────────────
KEY="ik_lass2"        # qcqp | mimo | ik_lass1 | ik_lass2

N_TRAIN=20000
N_TEST=5000
FOLDS=4
SEED_TRAIN=0
SEED_TEST=1

N_WORKERS=32          # data-gen parallelism; irrelevant (serial) for fast keys
CPUS=32               # also used as torch/BLAS thread budget during training

PARTITION="savio4_htc"
ACCOUNT="fc_power"
TIME="24:00:00"       # ik_lass2 generation is the long pole; trim for fast keys
MEM="64G"
CONDA_ENV="nn4opt"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs
JOB_NAME="pipeline_${KEY}"

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
# ipopt (local solver) and mosek/clarabel (relaxation solver) both live in the
# conda env's own bin -- make sure it's found ahead of any system solver.
export PATH="\$(conda info --base)/envs/${CONDA_ENV}/bin:\$PATH"

# One thread per worker process during data-gen; torch training is single-
# process so it gets the full CPU budget instead.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1

echo "[1/3] generating data"
python scripts/generate_dataset.py --key ${KEY} \\
    --n-train ${N_TRAIN} --n-test ${N_TEST} --n-workers ${N_WORKERS} \\
    --seed-train ${SEED_TRAIN} --seed-test ${SEED_TEST}

# Training is one process; give it back all the threads data-gen split up.
export OMP_NUM_THREADS=${CPUS} OPENBLAS_NUM_THREADS=${CPUS} MKL_NUM_THREADS=${CPUS}

echo "[2/3] training 4-fold deep nets"
python scripts/train_generic.py --key ${KEY} \\
    --n-train ${N_TRAIN} --n-test ${N_TEST} --folds ${FOLDS}

echo "[3/3] per-fold calibration + certification"
python scripts/evaluate_certify.py --key ${KEY} \\
    --n-train ${N_TRAIN} --n-test ${N_TEST} --folds ${FOLDS}

echo "Finished ${JOB_NAME} at \$(date)"
EOF

echo "Submitted: ${JOB_NAME}"
