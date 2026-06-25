import copy
import numpy as np
import pandas as pd
import pathlib

from pypower.api import runopf, ppoption
ppopt = ppoption(VERBOSE=0, OUT_ALL=0, OPT={'OPF_ALG': 560})
from pypower import case9, case14, case30, case57, case118, case300
cases = {
    "case9": case9,
    "case14": case14,
    "case30": case30,
    "case57": case57,
    "case118": case118,
    "case300": case300,
}

class DirectoryStructure:

    def __init__(self, case_name):
        # Define directory structure
        self.base_dir = pathlib.Path(".") / ".."
        self.data_dir = self.base_dir / "training_data" / case_name
        self.model_dir = self.base_dir / "models" / case_name
        self.model_dir.mkdir(exist_ok=True)
        self.results_dir = self.base_dir / "results" / case_name
        self.results_dir.mkdir(exist_ok=True)

class IEEECase:

    def __init__(self, name):
        self.name = name
        self.case = cases[name].__dict__[name]()
        self.dir_str = DirectoryStructure(name)

    @property
    def load_buses(self):
        # Get bus data
        bus_data = self.case["bus"]
        # Convert to dataframe
        bus_columns = ["bus_ID", "bus_type", "Pd", "Qd", "Gs", "Bs", "area_number", "Vm", "Va", "basekV", "zone",
                       "maxVm", "minVm"]
        assert len(bus_columns) == bus_data.shape[1]
        # Organize data
        df_bus = pd.DataFrame(data=bus_data, columns=bus_columns)
        df_bus[["bus_ID", "bus_type", "area_number", "zone"]] = df_bus[
            ["bus_ID", "bus_type", "area_number", "zone"]].astype(np.int64)
        df_bus = df_bus.set_index("bus_ID")
        # Load buses
        load_buses = list(df_bus.index[(df_bus[["Pd", "Qd"]] != 0.0).any(axis=1)])
        return load_buses

    def load_training_data(self, input_dim, num_samples):
        # Load data
        df_data = pd.read_csv(self.dir_str.data_dir / f"training_{self.name}_{input_dim}d_{num_samples}samples.csv")

        # Get feasible data points only
        df_data = df_data.loc[(df_data["Feas_Flag"] == 1) & (df_data["Global_Opt"] == 1)]
        print("Number of valid samples: ", len(df_data))

        # Organize data
        num_load_buses = int((len(df_data.columns) - 3) / 2)
        print("Number of load buses: ", num_load_buses)
        assert num_load_buses == len(self.load_buses)
        X_columns = [f"Pd{i}" for i in self.load_buses] + [f"Qd{i}" for i in self.load_buses]
        assert set(X_columns).issubset(set(df_data.columns))
        assert "Cost" in df_data.columns
        df_X = df_data[X_columns]
        df_y = df_data["Cost"]
        return df_X, df_y

    def run_opf(self, Pd, Qd):
        # Create copy of case
        case = copy.deepcopy(self.case)
        load_buses_idx = [bus - 1 for bus in self.load_buses]

        # Update case load vector
        case["bus"][load_buses_idx, 2] = Pd
        case["bus"][load_buses_idx, 3] = Qd

        # Solve case
        results = runopf(case, ppopt)
        if results["success"]:
            cost = results["f"]
        else:
            cost = np.nan

        # Cleanup
        del case

        return cost