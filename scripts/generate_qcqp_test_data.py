"""Generate a held-out test set for the QCQP example problem.

For each sampled parameter p = (a, b), records:
    - Cost: the SDP relaxation's optimal value (lower bound, exact iff `Exact`)
    - LocalCost: the local solver's (IPOPT) optimal value

Used by `scripts/certify_qcqp.py` to test a trained NN's ability to certify
optimality of the local solver's solutions.

Usage:
    python scripts/generate_qcqp_test_data.py --n-samples 1000 --seed 1
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from problems.qcqp_example.generate_data import generate_test_dataset, DATA_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_test_dataset(args.n_samples, args={"seed": args.seed})

    out_path = DATA_DIR / f"test_QCQP_example_2d_{args.n_samples}samples.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} test samples ({df['Exact'].sum()} exact) to {out_path}")
