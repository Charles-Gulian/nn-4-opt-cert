import numpy as np
import pandas as pd
import pathlib

from pypower import case9, case14, case30, case57, case118, case300
cases = {
    "case9": case9,
    "case14": case14,
    "case30": case30,
    "case57": case57,
    "case118": case118,
    "case300": case300,
}

def create_input_dataframe(case_name, input_dim, num_samples):
    # Get case data
    case_data = cases[case_name].__dict__[case_name]()
    bus_data = case_data["bus"]

    # Convert to dataframe
    bus_columns = [
        "bus_ID",
        "bus_type",
        "Pd",
        "Qd",
        "Gs",
        "Bs",
        "area_number",
        "Vm",
        "Va",
        "basekV",
        "zone",
        "maxVm",
        "minVm",
    ]
    assert len(bus_columns) == bus_data.shape[1]

    # Organize data
    df_bus = pd.DataFrame(data=bus_data, columns=bus_columns)
    int_columns = ["bus_ID", "bus_type", "area_number", "zone"]
    df_bus[int_columns] = df_bus[int_columns].astype(np.int64)
    df_bus = df_bus.set_index("bus_ID")

    # Load buses
    load_buses = list(df_bus.index[(df_bus[["Pd", "Qd"]] != 0.0).any(axis=1)])
    df_load = df_bus.loc[load_buses, ["Pd", "Qd"]]

    # Create columns
    Pd_columns = [f"Pd{k}" for k in load_buses]
    Qd_columns = [f"Qd{k}" for k in load_buses]

    # Create scale vector
    scale_vector = np.array([df_load.loc[k, "Pd"] for k in load_buses] + [df_load.loc[k, "Qd"] for k in load_buses])

    ### Generate random linear transform ###

    np.random.seed(10)  # Random seed
    m = input_dim  # Input dimension
    n = len(Pd_columns) + len(Qd_columns)  # Output dimension
    assert m <= n

    # Random matrix
    A = np.random.rand(n, m)

    # Normalize each row to sum to 1
    A /= A.sum(axis=1, keepdims=True)

    ### Generate random input data ###

    np.random.seed(11)  # Random seed
    N = num_samples  # Number of samples

    # Random input
    z = np.random.rand(m, N)

    # Random output
    b = (A @ z).T
    b = scale_vector * b

    ### Create final input data frame ###
    df_input = pd.DataFrame(columns=Pd_columns + Qd_columns, data=b)

    return df_input

experiment_matrix = {
    "case9": {
        1: 5000,
        3: 5000,
        6: 10000,
    },
    "case14": {
        3: 5000,
        6: 10000,
        10: 20000,
    },
    "case30": {
        6: 10000,
        10: 20000,
        25: 50000,
    },
    "case118": {
        10: 20000,
        25: 50000,
    },
    "case300": {
        60: 30000,
    },
}

if __name__ == "__main__":
    # Define current directory and input data directory
    base_dir = pathlib.Path(".") / ".."
    input_data_dir = base_dir / "input_data"
    # Create directories to catch training data
    for case_name in experiment_matrix:
        training_dir = base_dir / "training_data" / case_name
        training_dir.mkdir(exist_ok=True)
    # Create input data
    for case_name in experiment_matrix:
        case_dir = input_data_dir / case_name
        case_dir.mkdir(exist_ok=True)
        for input_dim, num_samples in experiment_matrix[case_name].items():
            print(case_name, input_dim, num_samples)
            df_data = create_input_dataframe(case_name, input_dim, num_samples)
            df_data.to_csv(case_dir / f"input_{case_name}_{input_dim}d_{num_samples}samples.csv")