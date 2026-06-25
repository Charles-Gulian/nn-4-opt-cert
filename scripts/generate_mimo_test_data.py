"""Generate a held-out test set for the MIMO detection problem.

Records Cost (SDP relaxation), Exact, and LocalCost (ZF detector) for each
sampled received signal b. Used by certify_mimo.py.

Usage:
    python scripts/generate_mimo_test_data.py --n-samples 1000 --seed 1
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from problems.mimo_detection.generate_data import generate_test_dataset, DATA_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sigma-sq", type=float, default=0.1, help="noise variance sigma^2")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_test_dataset(args.n_samples, args={"seed": args.seed, "sigma_sq": args.sigma_sq})

    out_path = DATA_DIR / f"test_MIMO_{args.n_samples}samples.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} test samples ({df['Exact'].sum()} exact) to {out_path}")
