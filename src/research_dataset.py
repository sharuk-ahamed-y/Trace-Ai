"""
src/research_dataset.py — Clean, leakage-free dataset loader for Research Image Provenance analysis.

Loads:
  1. 55-D .npy visual embeddings per image
  2. 19-D forensic features from dataset/features/forensic_features.csv
  3. Structured train/val/test pair metadata (pairs_train.csv, pairs_val.csv, pairs_test.csv)
  4. Ground truth DAG graphs (graphs_train.json, graphs_val.json, graphs_test.json)
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import ROOT

DATASET_DIR = ROOT / "dataset"
FEATURES_DIR = DATASET_DIR / "features"
FORENSIC_CSV = FEATURES_DIR / "forensic_features.csv"
METADATA_DIR = DATASET_DIR / "metadata"

# 12 Target transformation classes in dataset
TRANSFORMATION_CLASSES = [
    "crop",
    "resize",
    "jpeg",
    "blur",
    "brightness",
    "contrast",
    "sharpen",
    "perspective",
    "rotation",
    "color",
    "noise",
    "quality_degradation",
]

TFM2IDX = {c: i for i, c in enumerate(TRANSFORMATION_CLASSES)}
IDX2TFM = {i: c for i, c in enumerate(TRANSFORMATION_CLASSES)}


def load_image_features(dataset_dir: Path = DATASET_DIR) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Loads 55-D npy embeddings and 17-D numerical forensic features per image.
    Returns:
      features_dict: map of image_id -> 72-D combined numpy vector (float32)
      forensic_feature_names: list of forensic column names
    """
    features_dir = dataset_dir / "features"
    forensic_csv = features_dir / "forensic_features.csv"
    
    # Load forensic features
    if not forensic_csv.exists():
        raise FileNotFoundError(f"Forensic CSV not found at {forensic_csv}")
    
    df_f = pd.read_csv(forensic_csv)
    # Non-numeric columns to exclude
    exclude_cols = {"image_id", "file_name"}
    forensic_cols = [c for c in df_f.columns if c not in exclude_cols]
    
    # Create forensic dictionary
    forensic_map = {}
    for _, row in df_f.iterrows():
        img_id = str(row["image_id"]).strip()
        vec = row[forensic_cols].values.astype(np.float32)
        forensic_map[img_id] = vec
        
    # Standardize forensic features across dataset
    all_f_vecs = np.array(list(forensic_map.values()), dtype=np.float32)
    f_mean = np.mean(all_f_vecs, axis=0, keepdims=True)
    f_std = np.std(all_f_vecs, axis=0, keepdims=True) + 1e-6
    
    for img_id in list(forensic_map.keys()):
        forensic_map[img_id] = (forensic_map[img_id] - f_mean[0]) / f_std[0]

    # Combine 55D npy + 17D forensic
    combined_dict = {}
    npy_files = list(features_dir.glob("*.npy"))
    if not npy_files:
        # Check subdirectories
        npy_files = list(features_dir.rglob("*.npy"))
        
    for npy_path in npy_files:
        img_id = npy_path.stem.strip()
        npy_vec = np.load(npy_path).astype(np.float32)
        if npy_vec.ndim > 1:
            npy_vec = npy_vec.flatten()
            
        f_vec = forensic_map.get(img_id, np.zeros(len(forensic_cols), dtype=np.float32))
        combined_vec = np.concatenate([npy_vec, f_vec], axis=0)
        combined_dict[img_id] = combined_vec
        
    return combined_dict, forensic_cols


def build_pair_feature_vector(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    feature_mode: str = "combined"
) -> np.ndarray:
    """
    Constructs symmetric/asymmetric pair feature representation for Image A and Image B.
    feature_mode:
      - 'npy_only': use first 55 dims
      - 'forensic_only': use last 17 dims
      - 'combined': use all 72 dims
    """
    if feature_mode == "npy_only":
        v_a = vec_a[:55]
        v_b = vec_b[:55]
    elif feature_mode == "forensic_only":
        v_a = vec_a[55:]
        v_b = vec_b[55:]
    else:
        v_a = vec_a
        v_b = vec_b

    diff = np.abs(v_a - v_b)
    prod = v_a * v_b
    
    norm_a = np.linalg.norm(v_a) + 1e-8
    norm_b = np.linalg.norm(v_b) + 1e-8
    cos_sim = np.array([np.dot(v_a, v_b) / (norm_a * norm_b)], dtype=np.float32)

    return np.concatenate([v_a, v_b, diff, prod, cos_sim], axis=0).astype(np.float32)


class ResearchPairsDataset(Dataset):
    """
    PyTorch Dataset for pair-level training.
    Provides:
      - pair_vec: Pairwise feature vector
      - related_target: 1.0 (related) or 0.0 (unrelated)
      - direction_target: 1.0 (A_TO_B) or 0.0 (B_TO_A)
      - transform_target: class index 0..11
    """
    def __init__(
        self,
        pairs_csv: Path,
        image_features: Dict[str, np.ndarray],
        feature_mode: str = "combined"
    ):
        self.df = pd.read_csv(pairs_csv)
        self.image_features = image_features
        self.feature_mode = feature_mode
        self.samples = []

        for _, row in self.df.iterrows():
            img_a = str(row["image_a"]).strip()
            img_b = str(row["image_b"]).strip()
            
            if img_a not in image_features or img_b not in image_features:
                continue
                
            vec_a = image_features[img_a]
            vec_b = image_features[img_b]
            pair_vec = build_pair_feature_vector(vec_a, vec_b, feature_mode)
            
            related = float(row["related"])
            dir_str = str(row["direction"]).strip()
            direction = 1.0 if dir_str == "A_TO_B" else 0.0
            
            tfm_str = str(row["transformation"]).strip()
            tfm_idx = TFM2IDX.get(tfm_str, 0)
            
            self.samples.append({
                "image_a": img_a,
                "image_b": img_b,
                "pair_vec": torch.from_numpy(pair_vec),
                "related": torch.tensor(related, dtype=torch.float32),
                "direction": torch.tensor(direction, dtype=torch.float32),
                "transformation": torch.tensor(tfm_idx, dtype=torch.long),
                "tfm_str": tfm_str
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
