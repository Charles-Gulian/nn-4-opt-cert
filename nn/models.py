import torch.nn as nn


class DNN(nn.Module):
    """Standard feedforward network for regression onto a scalar value function."""

    def __init__(self, input_dim, hidden_dims=(64, 64)):
        super(DNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers += [nn.Linear(prev_dim, dim), nn.ReLU()]
            prev_dim = dim
        layers += [nn.Linear(prev_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
