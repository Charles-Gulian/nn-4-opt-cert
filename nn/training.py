"""Problem-agnostic training utilities: works on plain (X, y) arrays."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data as data_utils
from sklearn.model_selection import KFold
from tqdm import tqdm


def to_loader(X, y, batch_size, shuffle=True):
    X = torch.tensor(np.asarray(X), dtype=torch.float32)
    y = torch.tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
    dataset = data_utils.TensorDataset(X, y)
    return data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    losses = []
    for X, y in loader:
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def evaluate(model, loader, loss_fn):
    model.eval()
    losses = []
    with torch.no_grad():
        for X, y in loader:
            pred = model(X)
            losses.append(loss_fn(pred, y).item())
    return float(np.mean(losses))


def predict(model, X):
    model.eval()
    X = torch.tensor(np.asarray(X), dtype=torch.float32)
    with torch.no_grad():
        return model(X).numpy().reshape(-1)


def train_model(model, train_loader, test_loader, n_epochs=1000, learning_rate=1e-3,
                 weight_decay=0.0, verbose=True):
    """Train with AdamW and cosine-annealed learning rate -- standard defaults
    that work well across problems without per-problem tuning."""
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    train_history, test_history = [], []
    epoch_iter = tqdm(range(n_epochs), desc="Training") if verbose else range(n_epochs)
    for _ in epoch_iter:
        train_history.append(train_one_epoch(model, train_loader, optimizer, loss_fn))
        test_history.append(evaluate(model, test_loader, loss_fn))
        scheduler.step()

    return model, train_history, test_history


def train_model_two_phase(
    model, X_train, X_test, y_train, y_test,
    pretrain_epochs=500, pretrain_lr=1e-3, pretrain_batch_size=512,
    finetune_epochs=200, finetune_lr=1e-4, finetune_batch_size=32,
    weight_decay=0.0, verbose=True,
):
    """Two-phase training: large-batch/large-LR pre-training then small-batch/small-LR fine-tuning.

    Phase 1 (pre-training): big steps over the full loss landscape to reach a good basin quickly.
    Phase 2 (fine-tuning): noisier small-batch gradients + low LR to settle into a flat minimum.

    Returns (model, train_history, test_history) where the histories are concatenated across both
    phases. A vertical marker at index `pretrain_epochs` separates the two phases when plotting.
    """
    loss_fn = nn.MSELoss()

    def _run_phase(loader_train, loader_test, n_epochs, lr, label):
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        train_hist, test_hist = [], []
        epoch_iter = tqdm(range(n_epochs), desc=label) if verbose else range(n_epochs)
        for _ in epoch_iter:
            train_hist.append(train_one_epoch(model, loader_train, optimizer, loss_fn))
            test_hist.append(evaluate(model, loader_test, loss_fn))
            scheduler.step()
        return train_hist, test_hist

    pretrain_train_loader = to_loader(X_train, y_train, batch_size=pretrain_batch_size)
    pretrain_test_loader  = to_loader(X_test,  y_test,  batch_size=pretrain_batch_size, shuffle=False)
    finetune_train_loader = to_loader(X_train, y_train, batch_size=finetune_batch_size)
    finetune_test_loader  = to_loader(X_test,  y_test,  batch_size=finetune_batch_size, shuffle=False)

    tr1, te1 = _run_phase(pretrain_train_loader, pretrain_test_loader, pretrain_epochs, pretrain_lr,  "Pre-training")
    tr2, te2 = _run_phase(finetune_train_loader, finetune_test_loader, finetune_epochs, finetune_lr, "Fine-tuning")

    return model, tr1 + tr2, te1 + te2


def cross_validate(X, y, model_fn, n_splits=2, batch_size=50, random_state=0, train_kwargs=None):
    """K-fold cross-validation. Returns a list of dicts, one per fold, with keys:
    'model', 'train_history', 'test_history', 'y_true', 'y_pred' (held-out fold predictions).
    """
    train_kwargs = train_kwargs or {}
    X, y = np.asarray(X), np.asarray(y).reshape(-1, 1)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_loader = to_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
        test_loader = to_loader(X_test, y_test, batch_size=batch_size, shuffle=False)

        model = model_fn()
        model, train_history, test_history = train_model(model, train_loader, test_loader, **train_kwargs)

        y_pred = predict(model, X_test)
        fold_results.append(dict(
            fold=fold,
            model=model,
            train_history=train_history,
            test_history=test_history,
            y_true=y_test.reshape(-1),
            y_pred=y_pred,
        ))
    return fold_results


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path, weights_only=True))
