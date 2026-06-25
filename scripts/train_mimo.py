"""Train a DNN to predict the MIMO SDP relaxation value via k-fold CV.

Usage:
    python scripts/train_mimo.py --n-samples 5000 --n-epochs 1000
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from problems.mimo_detection.generate_data import DATA_DIR, _B_COLS
from problems.mimo_detection.problem import M_RECEIVERS
from nn.models import DNN
from nn.training import cross_validate, save_model

MODELS_DIR = pathlib.Path(__file__).resolve().parents[1] / "models" / "mimo_detection"
RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results" / "mimo_detection"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--n-splits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    data_path = DATA_DIR / f"training_MIMO_{args.n_samples}samples.csv"
    df = pd.read_csv(data_path)

    input_dim = 2 * M_RECEIVERS
    X = df[_B_COLS].values
    y = df["Cost"].values

    fold_results = cross_validate(
        X, y,
        model_fn=lambda: DNN(input_dim=input_dim, hidden_dims=tuple(args.hidden_dims)),
        n_splits=args.n_splits,
        batch_size=args.batch_size,
        random_state=args.seed,
        train_kwargs=dict(
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            verbose=not args.quiet,
        ),
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pred_dfs = []
    for fold in fold_results:
        save_path = MODELS_DIR / f"dnn_MIMO_{args.n_samples}samples_fold{fold['fold']}.pth"
        save_model(fold["model"], save_path)
        print(f"Fold {fold['fold']}: saved model to {save_path}")
        pred_dfs.append(pd.DataFrame({
            "fold": fold["fold"],
            "Cost": fold["y_true"],
            "Pred": fold["y_pred"],
        }))

    df_preds = pd.concat(pred_dfs, ignore_index=True)
    preds_path = RESULTS_DIR / f"cv_predictions_MIMO_{args.n_samples}samples.csv"
    df_preds.to_csv(preds_path, index=False)
    print(f"Saved out-of-fold predictions to {preds_path}")
