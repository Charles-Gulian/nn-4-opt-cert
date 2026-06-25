from ieee_case import IEEECase
from neural_networks import *

nn_training_parameters = {
    "case9": {
        1: dict(
            learning_rate=0.01,
            scheduler_step_size=500,
            scheduler_gamma=0.9,
            max_norm=5.0
        ),
        2: dict(
            learning_rate=0.00001,
            scheduler_step_size=500,
            scheduler_gamma=0.9,
            max_norm=1.0
        )
    },
    "case14": {
        1: dict(
            learning_rate=0.01,
            scheduler_step_size=500,
            scheduler_gamma=0.9,
            max_norm=3.0
        ),
        2: dict(
            learning_rate=0.000001,
            scheduler_step_size=500,
            scheduler_gamma=0.9,
            max_norm=1.0
        )
    },
    "case30": {
        1: dict(
            learning_rate=0.01,
            scheduler_step_size=500,
            scheduler_gamma=0.9,
            max_norm=3.0
        ),
        2: dict(
            learning_rate=0.000001,
            scheduler_step_size=500,
            scheduler_gamma=0.9,
            max_norm=1.0
        )
    },
    "case118": {
        1: dict(
            learning_rate=0.001,
            scheduler_step_size=500,
            scheduler_gamma=0.9,
            max_norm=5.0
        ),
        2: dict(
            learning_rate=0.00001,
            scheduler_step_size=200,
            scheduler_gamma=0.7,
            max_norm=1.0
        )
    },
}

icnn_training_parameters = {
    "case9": {
        1: dict(
            learning_rate=0.001,
            scheduler_step_size=500,
            scheduler_gamma=0.9,
            max_norm=5.0,
        ),
        2: dict(
            learning_rate=0.0001,
            scheduler_step_size=200,
            scheduler_gamma=0.7,
            max_norm=1.0
        )
    },
    "case14": {
        1: dict(
            learning_rate=0.001,
            scheduler_step_size=500,
            scheduler_gamma=0.9,
            max_norm=5.0,
        ),
        2: dict(
            learning_rate=0.0001,
            scheduler_step_size=200,
            scheduler_gamma=0.7,
            max_norm=1.0
        )
    },
    "case30": {
        1: dict(
            learning_rate=0.001,
            scheduler_step_size=500,
            scheduler_gamma=0.9,
            max_norm=5.0,
        ),
        2: dict(
            learning_rate=0.0001,
            scheduler_step_size=200,
            scheduler_gamma=0.7,
            max_norm=1.0
        )
    },
    "case118": {
        1: dict(
            learning_rate=0.001,
            scheduler_step_size=500,
            scheduler_gamma=0.9,
            max_norm=5.0,
        ),
        2: dict(
            learning_rate=0.00001,
            scheduler_step_size=200,
            scheduler_gamma=0.7,
            max_norm=1.0
        )
    },
}

# -----------------------------
# Training for IEEE Case Data
# -----------------------------

def main(case_name, input_dim, num_samples, SAVE=True):
    # Input case name, input dimension, number of samples
    print(f"Case: {case_name} | Input Dimension: {input_dim} | Num. Samples: {num_samples}")

    # Define IEEE case
    case = IEEECase(case_name)

    # Define number of load buses (for NN/ICNN structure input dimension)
    num_load_buses = len(case.load_buses)

    ### Load IEEE case SDP OPF cost training data for NN and ICNN

    # Load data
    df_X, df_y = case.load_training_data(input_dim, num_samples)

    # Get input data/labels
    X, y = df_X.values, df_y.values.reshape(-1, 1)

    ### Train Neural Network(s)

    # Create training/testing data sets
    X_train, y_train, X_test, y_test = create_train_test_datasets(X, y)
    train_loader1 = create_batches(X_train, y_train, batch_size=200)
    train_loader2 = create_batches(X_train, y_train, batch_size=50)
    test_loader = create_batches(X_test, y_test, batch_size=100)

    # Initialize models
    icnn_model = ICNN(input_dim=2 * num_load_buses)
    nn_model = FCNN(input_dim=2 * num_load_buses)

    # Save paths for models
    nn_save_path = case.dir_str.model_dir / f"nn_{case_name}_{input_dim}d_{num_samples}samples.pth"
    icnn_save_path = case.dir_str.model_dir / f"icnn_{case_name}_{input_dim}d_{num_samples}samples.pth"

    # Train models

    # Phase 1: SGD
    nn_model, nn_train_history_phase1, nn_test_history_phase1 = train_model(
        nn_model,
        train_loader1,
        test_loader,
        optimizer="SGD",
        n_epochs=10000,
        learning_rate=nn_training_parameters[case_name][1]["learning_rate"],
        weight_decay=1e-9,
        scheduler_step_size=nn_training_parameters[case_name][1]["scheduler_step_size"],
        scheduler_gamma=nn_training_parameters[case_name][1]["scheduler_gamma"],
        max_norm=nn_training_parameters[case_name][1]["max_norm"]
    )

    # Phase 2: Adam
    nn_model, nn_train_history_phase2, nn_test_history_phase2 = train_model(
        nn_model,
        train_loader2,
        test_loader,
        optimizer="Adam",
        n_epochs=10000,
        learning_rate=nn_training_parameters[case_name][2]["learning_rate"],
        weight_decay=1e-9,
        scheduler_step_size=nn_training_parameters[case_name][2]["scheduler_step_size"],
        scheduler_gamma=nn_training_parameters[case_name][2]["scheduler_gamma"],
        max_norm=nn_training_parameters[case_name][2]["max_norm"]
    )
    nn_train_history = nn_train_history_phase1 + nn_train_history_phase2
    nn_test_history = nn_test_history_phase1 + nn_test_history_phase2

    # Save model (optional)
    if SAVE:
        save_model(nn_model, nn_save_path)

    # Phase 1: SGD
    icnn_model, icnn_train_history_phase1, icnn_test_history_phase1 = train_model(
        icnn_model,
        train_loader1,
        test_loader,
        optimizer="SGD",
        n_epochs=10000,
        learning_rate=icnn_training_parameters[case_name][1]["learning_rate"],
        weight_decay=1e-9,
        scheduler_step_size=icnn_training_parameters[case_name][1]["scheduler_step_size"],
        scheduler_gamma=icnn_training_parameters[case_name][1]["scheduler_gamma"],
        max_norm=icnn_training_parameters[case_name][1]["max_norm"]
    )

    # Phase 2: Adam
    icnn_model, icnn_train_history_phase2, icnn_test_history_phase2 = train_model(
        icnn_model,
        train_loader2,
        test_loader,
        optimizer="Adam",
        n_epochs=10000,
        learning_rate=icnn_training_parameters[case_name][2]["learning_rate"],
        weight_decay=1e-9,
        scheduler_step_size=icnn_training_parameters[case_name][2]["scheduler_step_size"],
        scheduler_gamma=icnn_training_parameters[case_name][2]["scheduler_gamma"],
        max_norm=icnn_training_parameters[case_name][2]["max_norm"]
    )
    icnn_train_history = icnn_train_history_phase1 + icnn_train_history_phase2
    icnn_test_history = icnn_test_history_phase1 + icnn_test_history_phase2

    # Save model (optional)
    if SAVE:
        save_model(icnn_model, icnn_save_path)

    return nn_train_history, nn_test_history, icnn_train_history, icnn_test_history

if __name__ == "__main__":
    nn_train_history, nn_test_history, icnn_train_history, icnn_test_history = (
        main(case_name="case9", input_dim=1, num_samples=5000, SAVE=False) # Change SAVE=True to save/over-write models
    )

    # Show training/testing error
    plt.title("ICNN Train / Test Error")
    plt.plot(np.sqrt(np.array(icnn_train_history))[10:], label="Train")
    plt.plot(np.sqrt(np.array(icnn_test_history))[10:], label="Test")
    plt.legend()
    plt.axis([0, None, 0, None])
    plt.show()

    plt.title("NN Train / Test Error")
    plt.plot(np.sqrt(np.array(nn_train_history))[10:], label="Train")
    plt.plot(np.sqrt(np.array(nn_test_history))[10:], label="Test")
    plt.legend()
    plt.axis([0, None, 0, None])
    plt.show()