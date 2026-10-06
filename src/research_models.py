"""
src/research_models.py — PyTorch Neural Classifier Architectures for Image Provenance Analysis.

Includes:
  1. RelatedClassifier: Binary classifier for pair relatedness prediction P(related)
  2. DirectionClassifier: Binary classifier for direction prediction P(A -> B)
  3. TransformationClassifier: Multi-class classifier for 12 transformation classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelatedClassifier(nn.Module):
    """Binary classifier to determine if Image A and Image B are provenance-related."""
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DirectionClassifier(nn.Module):
    """Binary classifier to determine relationship direction (A -> B vs B -> A)."""
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TransformationClassifier(nn.Module):
    """Multi-class classifier predicting one of 12 transformation categories."""
    def __init__(self, in_dim: int, num_classes: int = 12, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
