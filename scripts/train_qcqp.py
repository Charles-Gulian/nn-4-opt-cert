"""Stage 2: train a DNN on the QCQP example data using k-fold cross-validation.

Saves one model checkpoint per fold to `models/qcqp_example/`, and a single
CSV of out-of-fold predictions to `results/qcqp_example/` (consumed by
`scripts/evaluate_qcqp.py`).

Usage:
    python scripts/train_qcqp.py --n-samples 5000 --n-epochs 2000
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from problems.qcqp_example.generate_data import DATA_DIR
from nn.models import DNN
from nn.training import cross_validate, save_model

MODEL_DIR = pathlib.Path(__file__).resolve().parents[1] / "models" / "qcqp_example"
RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results" / "qcqp_example"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-splits", type=int, default=2)
    parser.add_argument("--n-epochs", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    data_path = DATA_DIR / f"training_QCQP_example_2d_{args.n_samples}samples.csv"
    df = pd.read_csv(data_path)
    X, y = df[["a", "b"]].values, df["Cost"].values

    fold_results = cross_validate(
        X, y,
        model_fn=lambda: DNN(input_dim=2, hidden_dims=tuple(args.hidden_dims)),
        n_splits=args.n_splits,
        batch_size=50,
        random_state=args.seed,
        train_kwargs=dict(
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            verbose=not args.quiet,
        ),
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pred_dfs = []
    for fold in fold_results:
        save_path = MODEL_DIR / f"dnn_QCQP_example_2d_{args.n_samples}samples_fold{fold['fold']}.pth"
        save_model(fold["model"], save_path)
        print(f"Fold {fold['fold']}: saved model to {save_path}")

        pred_dfs.append(pd.DataFrame({
            "fold": fold["fold"],
            "Cost": fold["y_true"],
            "Pred": fold["y_pred"],
        }))

    preds_path = RESULTS_DIR / f"cv_predictions_QCQP_example_2d_{args.n_samples}samples.csv"
    pd.concat(pred_dfs, ignore_index=True).to_csv(preds_path, index=False)
    print(f"Saved out-of-fold predictions to {preds_path}")
