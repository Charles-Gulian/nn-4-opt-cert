"""Generate SDP-labeled training data for the MIMO detection problem.

Usage:
    python scripts/generate_mimo_data.py --n-samples 5000 --seed 0
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from problems.mimo_detection.generate_data import generate_dataset, DATA_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma-sq", type=float, default=0.1, help="noise variance sigma^2")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(args.n_samples, args={"seed": args.seed, "sigma_sq": args.sigma_sq})

    out_path = DATA_DIR / f"training_MIMO_{args.n_samples}samples.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} samples ({df['Exact'].sum()} exact) to {out_path}")
