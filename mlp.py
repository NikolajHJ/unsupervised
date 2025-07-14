#!/usr/bin/env python3
"""
torch_mlp_simple.py
────────────────────
A *tiny* two‑hidden‑layer MLP for binary classification **plus** a
bare‑bones scikit‑learn wrapper so it can plug straight into tools such
as `SequentialFeatureSelector`, grid‑search, pipelines, etc.

Changes from your original
~~~~~~~~~~~~~~~~~~~~~~~~~~
* Added a ``score()`` method – scikit‑learn calls this when you don’t
  supply an explicit *scoring* argument.
* Added optional **early stopping** (``early_stopping=True`` &
  ``n_iter_no_change``) using a 10 % validation split.
* Added ``weight_decay`` argument for L2 regularisation.
* Wrapper’s ``__init__`` exposes **all hyper‑parameters** so it behaves
  nicely with `clone()` (needed for feature‑selection).
* Uses the same random seed for reproducibility when you pass
  ``random_state``.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------
# 1. Network definition ------------------------------------------------
# ---------------------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, h1: int = 32, h2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),   # logits
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # (batch,)

# ---------------------------------------------------------------------
# 2. Training helper ---------------------------------------------------
# ---------------------------------------------------------------------

def _train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cpu",
    early_stopping: bool = False,
    n_iter_no_change: int = 10,
    verbose: int = 0,
):
    model.to(device)
    crit = nn.BCEWithLogitsLoss()
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            optimiser.step()

        if val_loader is not None and early_stopping:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += crit(model(xb), yb).item() * yb.size(0)
            val_loss /= len(val_loader.dataset)

            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                no_improve = 0
            else:
                no_improve += 1

            if verbose:
                print(f"epoch {epoch+1:3d}/{epochs}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}")

            if no_improve >= n_iter_no_change:
                if verbose:
                    print("Early stopping – no improvement for", n_iter_no_change, "epochs.")
                break

# ---------------------------------------------------------------------
# 3. sklearn‑compatible wrapper ---------------------------------------
# ---------------------------------------------------------------------
class TorchMLPWrapper(BaseEstimator, ClassifierMixin):
    """scikit‑learn wrapper around *SimpleMLP* suitable for feature‑selection."""

    def __init__(
        self,
        h1: int = 32,
        h2: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 200,
        batch_size: int = 64,
        early_stopping: bool = True,
        n_iter_no_change: int = 10,
        device: str = "cpu",
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.h1 = h1
        self.h2 = h2
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # -------------------------------------------------------------
    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()  # shape (N,)

        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

        # 10 % validation split if early stopping is on
        if self.early_stopping and len(ds) > 10:
            val_size = max(1, int(0.1 * len(ds)))
            train_size = len(ds) - val_size
            train_ds, val_ds = random_split(ds, [train_size, val_size])
            val_loader = DataLoader(val_ds, batch_size=self.batch_size)
        else:
            train_ds, val_loader = ds, None

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        self.model_ = SimpleMLP(X.shape[1], self.h1, self.h2)
        _train_loop(
            self.model_,
            train_loader,
            val_loader,
            epochs=self.epochs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            device=self.device,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change,
            verbose=self.verbose,
        )
        return self

    # -------------------------------------------------------------
    @torch.no_grad()
    def predict(self, X):
        check_is_fitted(self, "model_")
        X = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        self.model_.eval()
        preds = (torch.sigmoid(self.model_(X)) >= 0.5).cpu().numpy()
        return preds.astype(int).ravel()

    # -------------------------------------------------------------
    def score(self, X, y):
        """Return accuracy score to integrate smoothly with sklearn."""
        return accuracy_score(y, self.predict(X))
