import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data as data_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

from tqdm import tqdm
from sklearn.model_selection import train_test_split

# -----------------------------
# Standard Feedforward NN Model
# -----------------------------
class FCNN(nn.Module):
    def __init__(self, input_dim):
        super(FCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Input Convex Neural Network
# -----------------------------
class ICNN(nn.Module):
    def __init__(self, input_dim):
        super(ICNN, self).__init__()
        self.flatten = nn.Flatten()

        # First hidden layer
        self.first_hidden_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU()
        )

        # ICNN hidden layers with non-negative W_z
        self.second_layer_linear_prim = nn.Linear(512, 512)
        self.second_layer_linear_skip = nn.Linear(input_dim, 512)
        self.second_layer_act = nn.ReLU()

        self.third_layer_linear_prim = nn.Linear(512, 256)
        self.third_layer_linear_skip = nn.Linear(input_dim, 256)
        self.third_layer_act = nn.ReLU()

        self.fourth_layer_linear_prim = nn.Linear(256, 64)
        self.fourth_layer_linear_skip = nn.Linear(input_dim, 64)
        self.fourth_layer_act = nn.ReLU()

        self.fifth_layer_linear_prim = nn.Linear(64, 16)
        self.fifth_layer_linear_skip = nn.Linear(input_dim, 16)
        self.fifth_layer_act = nn.ReLU()

        self.output_layer_linear_prim = nn.Linear(16, 1)
        self.output_layer_linear_skip = nn.Linear(input_dim, 1)

        # Non-negative weights
        self.nonneg_layers = [layer for name, layer in self.named_modules() if
                              isinstance(layer, nn.Linear) and "prim" in name]

        # Enforce non-negative weights at initialization
        self.clamp_nonneg_weights()

    def clamp_nonneg_weights(self):
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear) and "prim" in name:
                layer.weight.data.clamp_(min=0)

    def forward(self, x):
        x = self.flatten(x)
        z1 = self.first_hidden_layer(x)
        z2 = self.second_layer_act(self.second_layer_linear_prim(z1) + self.second_layer_linear_skip(x))
        z3 = self.third_layer_act(self.third_layer_linear_prim(z2) + self.third_layer_linear_skip(x))
        z4 = self.fourth_layer_act(self.fourth_layer_linear_prim(z3) + self.fourth_layer_linear_skip(x))
        z5 = self.fifth_layer_act(self.fifth_layer_linear_prim(z4) + self.fifth_layer_linear_skip(x))
        out = self.output_layer_linear_prim(z5) + self.output_layer_linear_skip(x)
        return out


# -----------------------------
# Training Functions
# -----------------------------

def train(model, dataloader, optimizer, loss_fn, scheduler=None, max_norm=None):
    # Prepare model for training
    model.train()
    loss_arr = []
    for X, y in dataloader:
        # Get prediction / loss
        pred = model(X)
        loss = loss_fn(pred, y)
        loss_arr.append(loss.item())
        # Do backpropagation
        optimizer.zero_grad()
        loss.backward()
        # Gradient clip (optional)
        if max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        # Update weights
        optimizer.step()
        # Optional clamping (for ICNN)
        if isinstance(model, ICNN):
            model.clamp_nonneg_weights()
    return np.mean(loss_arr)


def test(model, dataloader, loss_fn):
    model.eval()
    loss_arr = []
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)
            loss_arr.append(loss.item())
    return np.mean(loss_arr)


def pred(model, dataloader):
    model.eval()
    pred_arr = []
    with torch.no_grad():
        for X, _ in dataloader:
            pred = model(X)
            pred_arr.append(pred.item())
    return np.array(pred_arr)


def train_loop(model, train_loader, test_loader, loss_fn, optimizer, scheduler=None, epochs=1000, max_norm=None):
    # Loop through epochs of training / testing
    train_history, test_history = [], []
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        train_loss = train(model, train_loader, optimizer, loss_fn, scheduler=scheduler, max_norm=max_norm)
        train_history.append(train_loss)
        test_loss = test(model, test_loader, loss_fn)
        test_history.append(test_loss)
        # Scheduler step
        if scheduler:
            scheduler.step()
    print("Done!")
    return model, train_history, test_history


def create_train_test_datasets(X, y):
    # Create train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

    return X_train, y_train, X_test, y_test


def create_batches(X, y, batch_size, shuffle=True):
    # Convert to tensors
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    # Get training / testing data loaders
    data = data_utils.TensorDataset(X, y)
    data_loader = data_utils.DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    return data_loader


def train_model(model, train_loader, test_loader, optimizer="SGD", n_epochs=5000, learning_rate=0.1, weight_decay=0.0,
                scheduler_step_size=None, scheduler_gamma=None, max_norm=None):

    # Loss function
    loss_fn = nn.MSELoss()  # Mean square error
    if optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Scheduler
    if (scheduler_step_size is not None) or (scheduler_gamma is not None):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    else:
        scheduler = None

    # Train model
    print("Training...")
    model, train_history, test_history = train_loop(model, train_loader, test_loader, loss_fn, optimizer,
                                                    epochs=n_epochs, scheduler=scheduler, max_norm=max_norm)

    return model, train_history, test_history


def save_model(model, save_path):
    # Save model weights
    torch.save(model.state_dict(), save_path)


def load_model(model, save_path):
    # Load saved model weights
    model.load_state_dict(torch.load(save_path, weights_only=True))

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # -----------------------------
    # Create Training Data
    # -----------------------------

    # Generate training data for f(x) = x^2 - alpha x^4 (non-convex)
    np.random.seed(0)
    torch.manual_seed(0)

    def f(x, alpha=0.):
        return x ** 2 - alpha * x ** 4


    n_samples = 5000
    X = 4 * np.random.randn(n_samples).reshape(-1, 1)
    y = f(X)

    # -----------------------------
    # Train Both Models
    # -----------------------------

    # Create training/testing data sets
    X_train, y_train, X_test, y_test = create_train_test_datasets(X, y)
    train_loader1 = create_batches(X_train, y_train, batch_size=200)
    train_loader2 = create_batches(X_train, y_train, batch_size=50)
    test_loader = create_batches(X_test, y_test, batch_size=50)

    # Initialize models
    nn_model = FCNN(input_dim=1)
    icnn_model = ICNN(input_dim=1)

    # Train models

    # Phase 1: SGD
    nn_model, nn_train_history_phase1, nn_test_history_phase1 = train_model(
        nn_model,
        train_loader1,
        test_loader,
        optimizer="SGD",
        n_epochs=500,
        learning_rate=0.01,
        weight_decay=1e-9,
        scheduler_step_size=64,
        scheduler_gamma=0.8,
        max_norm=3.0
    )

    # Phase 2: Adam
    nn_model, nn_train_history_phase2, nn_test_history_phase2 = train_model(
        nn_model,
        train_loader2,
        test_loader,
        optimizer="Adam",
        n_epochs=500,
        learning_rate=0.0001,
        weight_decay=1e-9,
        scheduler_step_size=64,
        scheduler_gamma=0.8,
        max_norm=3.0
    )
    nn_train_history = nn_train_history_phase1 + nn_train_history_phase2
    nn_test_history = nn_test_history_phase1 + nn_test_history_phase2

    # Phase 1: SGD
    icnn_model, icnn_train_history_phase1, icnn_test_history_phase1 = train_model(
        icnn_model,
        train_loader1,
        test_loader,
        optimizer="SGD",
        n_epochs=500,
        learning_rate=0.1,
        weight_decay=1e-9,
        scheduler_step_size=64,
        scheduler_gamma=0.8,
        max_norm=3.0
    )

    # Phase 2: Adam
    icnn_model, icnn_train_history_phase2, icnn_test_history_phase2 = train_model(
        icnn_model,
        train_loader2,
        test_loader,
        optimizer="Adam",
        n_epochs=500,
        learning_rate=0.0001,
        weight_decay=1e-9,
        scheduler_step_size=64,
        scheduler_gamma=0.8,
        max_norm=3.0
    )
    icnn_train_history = icnn_train_history_phase1 + icnn_train_history_phase2
    icnn_test_history = icnn_test_history_phase1 + icnn_test_history_phase2

    # -----------------------------
    # Plotting
    # -----------------------------

    x_test = torch.linspace(-15, 15, 200).reshape(-1, 1)
    y_true = f(x_test)

    with torch.no_grad():
        y_nn_pred = nn_model(x_test)
        y_icnn_pred = icnn_model(x_test)

    plt.figure(figsize=(10, 5))
    plt.plot(x_test, y_true, label='True $f(x) = x^2 - \\alpha x^4$', color='black')
    plt.plot(x_test, y_nn_pred, label='Standard NN', linestyle='--')
    plt.plot(x_test, y_icnn_pred, label='ICNN', linestyle='-.')
    plt.legend()
    plt.show()

    # Test model
    plt.title("ICNN Train / Test Error")
    plt.plot(np.sqrt(np.array(icnn_train_history))[10:], label="Train")
    plt.plot(np.sqrt(np.array(icnn_test_history))[10:], label="Test")
    plt.legend()
    # plt.yscale('log')
    plt.axis([0, None, 0, 10])
    plt.show()

    plt.title("NN Train / Test Error")
    plt.plot(np.sqrt(np.array(nn_train_history))[10:], label="Train")
    plt.plot(np.sqrt(np.array(nn_test_history))[10:], label="Test")
    plt.legend()
    # plt.yscale('log')
    plt.axis([0, None, 0, 10])
    plt.show()