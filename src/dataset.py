"""
dataset.py — PyTorch Dataset classes for direction and transformation classifiers.

Splitting is done by ORIGINAL image identity so that child images derived from
the same original never appear in different splits.
"""

import json
import random
from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import TRANSFORMATION_NAMES, RANDOM_SEED
from src.pair_features import build_pair_feature


# ─────────────────────────────────────────────────────────────────────────────
# Label maps
# ─────────────────────────────────────────────────────────────────────────────

DIRECTION_LABELS: Dict[str, int] = {
    "parent_to_child": 0,
    "child_to_parent": 1,
}
DIRECTION_NAMES: List[str] = ["parent_to_child", "child_to_parent"]

TRANSFORM_LABEL_MAP: Dict[str, int] = {
    name: idx for idx, name in enumerate(TRANSFORMATION_NAMES)
}
TRANSFORM_NAMES: List[str] = TRANSFORMATION_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Split by original identity
# ─────────────────────────────────────────────────────────────────────────────

def split_by_original(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into train/val/test ensuring every child of the same original
    ends up in the same split (no data leakage).

    Columns expected: parent_path, child_path, ...
    """
    originals = sorted(df["parent_path"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(originals)

    n = len(originals)
    n_train = max(1, int(n * train_frac))
    n_val   = max(1, int(n * val_frac))

    train_origs = originals[:n_train]
    val_origs   = originals[n_train : n_train + n_val]
    test_origs  = originals[n_train + n_val :]

    # Fall-back for small datasets (e.g. 1 or 2 originals)
    if not val_origs:
        val_origs = train_origs
    if not test_origs:
        test_origs = train_origs

    train_df = df[df["parent_path"].isin(train_origs)].reset_index(drop=True)
    val_df   = df[df["parent_path"].isin(val_origs)].reset_index(drop=True)
    test_df  = df[df["parent_path"].isin(test_origs)].reset_index(drop=True)

    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Direction dataset
# ─────────────────────────────────────────────────────────────────────────────

class DirectionDataset(Dataset):
    """
    For each CSV edge we create TWO samples:
      - (parent_emb, child_emb) → label 0  (parent_to_child)
      - (child_emb, parent_emb) → label 1  (child_to_parent)

    This ensures both classes are always present.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: Dict[str, torch.Tensor],
    ) -> None:
        self.samples: List[Tuple[torch.Tensor, int]] = []

        for _, row in df.iterrows():
            p = str(row["parent_path"])
            c = str(row["child_path"])

            if p not in embeddings or c not in embeddings:
                continue   # skip if embedding missing

            emb_p = embeddings[p]
            emb_c = embeddings[c]

            feat_fwd = build_pair_feature(emb_p, emb_c)
            feat_rev = build_pair_feature(emb_c, emb_p)

            self.samples.append((feat_fwd, DIRECTION_LABELS["parent_to_child"]))
            self.samples.append((feat_rev, DIRECTION_LABELS["child_to_parent"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.samples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Transformation dataset
# ─────────────────────────────────────────────────────────────────────────────

class TransformationDataset(Dataset):
    """
    One sample per CSV edge:  (parent_emb, child_emb) → transform label.
    The pair is always in parent→child order (ground truth direction).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: Dict[str, torch.Tensor],
    ) -> None:
        self.samples: List[Tuple[torch.Tensor, int]] = []

        for _, row in df.iterrows():
            p = str(row["parent_path"])
            c = str(row["child_path"])
            t = row["transformation"]

            if p not in embeddings or c not in embeddings:
                continue
            if t not in TRANSFORM_LABEL_MAP:
                continue

            emb_p = embeddings[p]
            emb_c = embeddings[c]
            feat  = build_pair_feature(emb_p, emb_c)
            label = TRANSFORM_LABEL_MAP[t]
            self.samples.append((feat, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.samples[idx]
