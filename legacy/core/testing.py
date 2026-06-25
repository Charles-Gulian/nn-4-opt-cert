import numpy as np
import pandas as pd

from ieee_case import IEEECase
from neural_networks import *

### Experiment ###
# Compare NN, ICNN, and OPF solver on test data set

def main(case_name, input_dim, num_samples, SAVE=True):

    # Input case name, input dimension, number of samples
    print(f"Case: {case_name} | Input Dimension: {input_dim} | Num. Samples: {num_samples}")

    # Define IEEE case
    case = IEEECase(case_name)

    # Define number of load buses
    num_load_buses = len(case.load_buses)

    # Load data
    df_X, df_y = case.load_training_data(input_dim, num_samples)
    X_columns = [f"Pd{i}" for i in case.load_buses] + [f"Qd{i}" for i in case.load_buses]

    # Get input data/labels
    X, y = df_X.values, df_y.values.reshape(-1, 1)

    # Create training/testing data sets
    X_train, y_train, X_test, y_test = create_train_test_datasets(X, y)

    # Initialize models
    icnn_model = ICNN(input_dim=2 * num_load_buses)
    nn_model = FCNN(input_dim=2 * num_load_buses)

    # Save paths for models
    nn_save_path = case.dir_str.model_dir / f"nn_{case_name}_{input_dim}d_{num_samples}samples.pth"
    icnn_save_path = case.dir_str.model_dir / f"icnn_{case_name}_{input_dim}d_{num_samples}samples.pth"

    # Load models
    print("Loading models...")
    load_model(nn_model, nn_save_path)
    load_model(icnn_model, icnn_save_path)
    print("Done.")

    # Create test data tensor
    test_loader = create_batches(X_test, y_test, batch_size=1, shuffle=False)

    # Get NN, ICNN predictions
    print("Getting NN / ICNN predictions...")
    nn_pred_arr = pred(nn_model, test_loader)
    icnn_pred_arr = pred(icnn_model, test_loader)
    print("Done.")

    # Get local search solver OPF predictions
    print("Getting local search OPF solver predictions...")
    Pd_test, Qd_test = X_test[:, :num_load_buses], X_test[:, num_load_buses:]
    lss_pred_arr = np.array([case.run_opf(Pd_test[i, :], Qd_test[i, :]) for i in range(X_test.shape[0])])
    print("Done.")

    # Save results
    df_results = pd.DataFrame(columns=X_columns + ["Cost", "LSS Pred", "NN Pred", "ICNN Pred"])
    df_results[X_columns] = X_test
    df_results["Cost"] = y_test
    df_results["LSS Pred"] = lss_pred_arr
    df_results["NN Pred"] = nn_pred_arr
    df_results["ICNN Pred"] = icnn_pred_arr

    if SAVE:
        df_results.to_csv(case.dir_str.results_dir / f"results_{case_name}_{input_dim}d_{num_samples}samples.csv")

    return df_results

if __name__ == "__main__":
    df_results = main(case_name="case9", input_dim=1, num_samples=5000, SAVE=False)