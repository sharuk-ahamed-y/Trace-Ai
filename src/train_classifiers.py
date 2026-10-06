"""
train_classifiers.py — MLP classifiers for direction and transformation prediction.

Architecture
------------
  Input → Linear(2305, 256) → ReLU → Dropout(0.3)
        → Linear(256, 128)  → ReLU → Dropout(0.3)
        → Linear(128, n_classes)
"""

import pickle
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    MLP_HIDDEN_DIM, MLP_DROPOUT, MLP_LR, MLP_EPOCHS, MLP_BATCH_SIZE,
    MODELS_DIR, RANDOM_SEED, TRANSFORMATION_NAMES,
)
from src.dataset import DIRECTION_NAMES, TRANSFORM_NAMES
from src.pair_features import PAIR_FEATURE_DIM


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Two-hidden-layer MLP classifier."""

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int,
                 dropout: float = 0.30) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def _accuracy(model: nn.Module, loader: DataLoader,
              device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total if total > 0 else 0.0


def train_classifier(
    train_dataset,
    val_dataset,
    test_dataset,
    n_classes: int,
    label: str,
    device: torch.device,
    seed: int = RANDOM_SEED,
) -> Tuple[MLP, float, float]:
    """
    Train a single MLP and return (model, val_accuracy, test_accuracy).
    """
    torch.manual_seed(seed)

    model = MLP(PAIR_FEATURE_DIM, MLP_HIDDEN_DIM, n_classes, MLP_DROPOUT).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    criterion = nn.CrossEntropyLoss()

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=MLP_BATCH_SIZE,
                              shuffle=True,  generator=g)
    val_loader   = DataLoader(val_dataset,   batch_size=MLP_BATCH_SIZE,
                              shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=MLP_BATCH_SIZE,
                              shuffle=False)

    best_val_acc = 0.0
    best_state   = None

    print(f"\n[Train] {label} classifier  ({n_classes} classes, "
          f"{MLP_EPOCHS} epochs)")
    print(f"  train={len(train_dataset)}  val={len(val_dataset)}  "
          f"test={len(test_dataset)}")

    for epoch in range(1, MLP_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            val_acc = _accuracy(model, val_loader, device)
            print(f"  Epoch {epoch:3d} | loss={epoch_loss / max(1, len(train_loader)):.4f} "
                  f"| val_acc={val_acc:.3f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    val_acc  = _accuracy(model, val_loader,  device)
    test_acc = _accuracy(model, test_loader, device)
    print(f"\n  [{label}] Best val acc = {val_acc:.4f} | Test acc = {test_acc:.4f}")
    return model, val_acc, test_acc


# ─────────────────────────────────────────────────────────────────────────────
# Save / load helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_direction_model(model: MLP) -> None:
    path = MODELS_DIR / "direction_mlp.pt"
    torch.save(model.state_dict(), path)
    meta = {"class_names": DIRECTION_NAMES, "n_classes": 2,
            "input_dim": PAIR_FEATURE_DIM}
    with open(MODELS_DIR / "direction_meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"[Save] Direction model -> {path}")


def save_transform_model(model: MLP) -> None:
    path = MODELS_DIR / "transform_mlp.pt"
    torch.save(model.state_dict(), path)
    meta = {"class_names": TRANSFORM_NAMES, "n_classes": len(TRANSFORM_NAMES),
            "input_dim": PAIR_FEATURE_DIM}
    with open(MODELS_DIR / "transform_meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"[Save] Transform model -> {path}")


def load_direction_model(device: torch.device) -> Tuple[MLP, list]:
    with open(MODELS_DIR / "direction_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    model = MLP(meta["input_dim"], MLP_HIDDEN_DIM, meta["n_classes"], MLP_DROPOUT)
    model.load_state_dict(torch.load(MODELS_DIR / "direction_mlp.pt",
                                     map_location=device))
    model.to(device).eval()
    return model, meta["class_names"]


def load_transform_model(device: torch.device) -> Tuple[MLP, list]:
    with open(MODELS_DIR / "transform_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    model = MLP(meta["input_dim"], MLP_HIDDEN_DIM, meta["n_classes"], MLP_DROPOUT)
    model.load_state_dict(torch.load(MODELS_DIR / "transform_mlp.pt",
                                     map_location=device))
    model.to(device).eval()
    return model, meta["class_names"]
