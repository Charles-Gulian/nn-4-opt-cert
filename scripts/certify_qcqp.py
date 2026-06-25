"""Test a trained NN's ability to certify optimality of the local solver.

For each test point, compares:
    - Cost:      SDP relaxation value (ground truth lower bound)
    - LocalCost: local solver (IPOPT) value
    - Pred:      NN prediction

A local solution is *actually* optimal if |LocalCost - Cost| <= tol (the
relaxation is tight). The NN *certifies* optimality if |LocalCost - Pred| <=
tol. Reports the resulting confusion matrix (TP/TN/FP/FN) and false
positive/negative rates.

Usage:
    python scripts/certify_qcqp.py --n-samples 1000 --model-path models/qcqp_example/dnn_QCQP_example_2d_5000samples_fold0.pth
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from problems.qcqp_example.generate_data import DATA_DIR
from nn.models import DNN
from nn.training import load_model, predict
from nn.metrics import optimality_confusion_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--model-path", type=pathlib.Path, required=True)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--tol", type=float, default=1e-2)
    args = parser.parse_args()

    data_path = DATA_DIR / f"test_QCQP_example_2d_{args.n_samples}samples.csv"
    df = pd.read_csv(data_path)

    model = DNN(input_dim=2, hidden_dims=tuple(args.hidden_dims))
    load_model(model, args.model_path)

    df["Pred"] = predict(model, df[["a", "b"]].values)

    cm = optimality_confusion_matrix(df["Cost"], df["LocalCost"], df["Pred"], tol=args.tol)

    print(f"Optimality certification (tol={args.tol}), n={cm['n']}")
    print(f"{'':>20} {'Predicted Optimal':>18} {'Predicted Suboptimal':>22}")
    print(f"{'Actually Optimal':>20} {cm['tp']:>18} {cm['fn']:>22}")
    print(f"{'Actually Suboptimal':>20} {cm['fp']:>18} {cm['tn']:>22}")
    print()
    print(f"False positive rate: {cm['fpr']:.4f}")
    print(f"False negative rate: {cm['fnr']:.4f}")
