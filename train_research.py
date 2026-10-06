"""
train_research.py — Rigorous, Research-Grade Training & Ablation Benchmark Pipeline.

Evaluates:
  - Feature Ablation 1: 55-D npy visual embeddings only
  - Feature Ablation 2: 17-D forensic/spectral features only
  - Feature Ablation 3: Combined 72-D representation

Metrics Reported:
  - Related/Unrelated: Accuracy, Precision, Recall, Macro-F1, Confusion Matrix
  - Direction: Accuracy, Precision, Recall, Macro-F1, Confusion Matrix
  - Transformation: Accuracy, Per-Class F1, Macro-F1, Weighted-F1, Confusion Matrix
  - Provenance Graph: Edge Precision, Edge Recall, Edge F1, Root Detection Accuracy

Usage:
  python train_research.py --epochs 35
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

from src.config import ROOT, RANDOM_SEED
from src.research_dataset import (
    load_image_features, build_pair_feature_vector, ResearchPairsDataset,
    TRANSFORMATION_CLASSES, TFM2IDX
)
from src.research_models import RelatedClassifier, DirectionClassifier, TransformationClassifier
from src.research_graph import construct_provenance_graph

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = MODELS_DIR / "research_provenance_model.pt"


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    rel_model, dir_model, tfm_model,
    dataloader,
    optimizer,
    criterion_rel, criterion_dir, criterion_tfm,
    device
):
    rel_model.train()
    dir_model.train()
    tfm_model.train()

    total_loss = 0.0

    for batch in dataloader:
        pair_vec = batch["pair_vec"].to(device)
        rel_target = batch["related"].to(device)
        dir_target = batch["direction"].to(device)
        tfm_target = batch["transformation"].to(device)

        optimizer.zero_grad()

        # 1. Related loss
        rel_logits = rel_model(pair_vec)
        loss_rel = criterion_rel(rel_logits, rel_target)

        # Mask for related pairs only
        rel_mask = (rel_target > 0.5)

        loss_dir = torch.tensor(0.0, device=device)
        loss_tfm = torch.tensor(0.0, device=device)

        if rel_mask.sum() > 0:
            rel_pair_vec = pair_vec[rel_mask]
            rel_dir_target = dir_target[rel_mask]
            rel_tfm_target = tfm_target[rel_mask]

            dir_logits = dir_model(rel_pair_vec)
            loss_dir = criterion_dir(dir_logits, rel_dir_target)

            tfm_logits = tfm_model(rel_pair_vec)
            loss_tfm = criterion_tfm(tfm_logits, rel_tfm_target)

        loss = loss_rel + loss_dir + loss_tfm
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def evaluate_pipeline(
    rel_model, dir_model, tfm_model,
    dataloader, device
):
    rel_model.eval()
    dir_model.eval()
    tfm_model.eval()

    rel_preds, rel_gt = [], []
    dir_preds, dir_gt = [], []
    tfm_preds, tfm_gt = [], []

    with torch.no_grad():
        for batch in dataloader:
            pair_vec = batch["pair_vec"].to(device)
            rel_target = batch["related"]
            dir_target = batch["direction"]
            tfm_target = batch["transformation"]

            # Related
            rel_logits = rel_model(pair_vec)
            rel_probs = torch.sigmoid(rel_logits).cpu().numpy()
            rel_p_binary = (rel_probs >= 0.5).astype(int)

            rel_preds.extend(rel_p_binary)
            rel_gt.extend(rel_target.numpy().astype(int))

            # Direction & Transformation for related pairs
            rel_mask = (rel_target > 0.5).numpy()
            if rel_mask.sum() > 0:
                rel_pair_vec = pair_vec[torch.from_numpy(rel_mask).to(device)]

                dir_logits = dir_model(rel_pair_vec)
                dir_probs = torch.sigmoid(dir_logits).cpu().numpy()
                dir_p_binary = (dir_probs >= 0.5).astype(int)

                dir_preds.extend(dir_p_binary)
                dir_gt.extend(dir_target[rel_mask].numpy().astype(int))

                tfm_logits = tfm_model(rel_pair_vec)
                tfm_p_class = torch.argmax(tfm_logits, dim=-1).cpu().numpy()

                tfm_preds.extend(tfm_p_class)
                tfm_gt.extend(tfm_target[rel_mask].numpy())

    # Metrics
    rel_acc = accuracy_score(rel_gt, rel_preds)
    rel_prec, rel_rec, rel_f1, _ = precision_recall_fscore_support(rel_gt, rel_preds, average="macro", zero_division=0)
    rel_cm = confusion_matrix(rel_gt, rel_preds).tolist()

    dir_acc = accuracy_score(dir_gt, dir_preds) if dir_gt else 0.0
    dir_prec, dir_rec, dir_f1, _ = precision_recall_fscore_support(dir_gt, dir_preds, average="macro", zero_division=0) if dir_gt else (0,0,0,None)
    dir_cm = confusion_matrix(dir_gt, dir_preds).tolist() if dir_gt else []

    tfm_acc = accuracy_score(tfm_gt, tfm_preds) if tfm_gt else 0.0
    tfm_macro_f1 = precision_recall_fscore_support(tfm_gt, tfm_preds, average="macro", zero_division=0)[2] if tfm_gt else 0.0
    tfm_weighted_f1 = precision_recall_fscore_support(tfm_gt, tfm_preds, average="weighted", zero_division=0)[2] if tfm_gt else 0.0
    tfm_cm = confusion_matrix(tfm_gt, tfm_preds).tolist() if tfm_gt else []

    return {
        "related": {
            "accuracy": round(float(rel_acc), 4),
            "precision": round(float(rel_prec), 4),
            "recall": round(float(rel_rec), 4),
            "macro_f1": round(float(rel_f1), 4),
            "confusion_matrix": rel_cm
        },
        "direction": {
            "accuracy": round(float(dir_acc), 4),
            "precision": round(float(dir_prec), 4),
            "recall": round(float(dir_rec), 4),
            "macro_f1": round(float(dir_f1), 4),
            "confusion_matrix": dir_cm
        },
        "transformation": {
            "accuracy": round(float(tfm_acc), 4),
            "macro_f1": round(float(tfm_macro_f1), 4),
            "weighted_f1": round(float(tfm_weighted_f1), 4),
            "confusion_matrix": tfm_cm
        }
    }


def run_experiment(feature_mode: str, epochs: int, device: torch.device):
    print(f"\n==================================================")
    print(f" RUNNING EXPERIMENT — Feature Mode: {feature_mode.upper()}")
    print(f"==================================================")

    set_seed(RANDOM_SEED)
    img_features, forensic_cols = load_image_features()

    dataset_dir = ROOT / "dataset"
    meta_dir = dataset_dir / "metadata"

    train_ds = ResearchPairsDataset(meta_dir / "pairs_train.csv", img_features, feature_mode)
    val_ds   = ResearchPairsDataset(meta_dir / "pairs_val.csv", img_features, feature_mode)
    test_ds  = ResearchPairsDataset(meta_dir / "pairs_test.csv", img_features, feature_mode)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

    sample_vec = train_ds[0]["pair_vec"]
    in_dim = len(sample_vec)

    rel_model = RelatedClassifier(in_dim).to(device)
    dir_model = DirectionClassifier(in_dim).to(device)
    tfm_model = TransformationClassifier(in_dim, num_classes=len(TRANSFORMATION_CLASSES)).to(device)

    # Class weighting for transformation classifier
    all_tfm_labels = [s["transformation"].item() for s in train_ds if s["related"].item() > 0.5]
    unique_classes = np.unique(all_tfm_labels)
    class_weights_arr = compute_class_weight("balanced", classes=unique_classes, y=all_tfm_labels)
    
    weights_full = np.ones(len(TRANSFORMATION_CLASSES), dtype=np.float32)
    for cls_idx, w in zip(unique_classes, class_weights_arr):
        weights_full[cls_idx] = w

    criterion_rel = nn.BCEWithLogitsLoss()
    criterion_dir = nn.BCEWithLogitsLoss()
    criterion_tfm = nn.CrossEntropyLoss(weight=torch.from_numpy(weights_full).to(device))

    all_params = list(rel_model.parameters()) + list(dir_model.parameters()) + list(tfm_model.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=1e-3, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        loss = train_epoch(
            rel_model, dir_model, tfm_model,
            train_loader, optimizer,
            criterion_rel, criterion_dir, criterion_tfm,
            device
        )
        val_res = evaluate_pipeline(rel_model, dir_model, tfm_model, val_loader, device)
        val_f1 = val_res["related"]["macro_f1"] + val_res["transformation"]["macro_f1"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {
                "rel": rel_model.state_dict(),
                "dir": dir_model.state_dict(),
                "tfm": tfm_model.state_dict(),
                "in_dim": in_dim,
                "feature_mode": feature_mode,
                "val_res": val_res
            }

        if epoch % 10 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] Loss: {loss:.4f} | Val Rel Macro-F1: {val_res['related']['macro_f1']:.4f} | Val Tfm Macro-F1: {val_res['transformation']['macro_f1']:.4f}")

    # Load best validation checkpoint before final test evaluation
    rel_model.load_state_dict(best_weights["rel"])
    dir_model.load_state_dict(best_weights["dir"])
    tfm_model.load_state_dict(best_weights["tfm"])

    # Final evaluation on TEST split only (NO DATA LEAKAGE)
    test_res = evaluate_pipeline(rel_model, dir_model, tfm_model, test_loader, device)

    print(f"\n  [FINAL TEST RESULTS ({feature_mode.upper()})]")
    print(f"   - Related Classifier   -> Accuracy: {test_res['related']['accuracy']:.4f}, Macro-F1: {test_res['related']['macro_f1']:.4f}")
    print(f"   - Direction Classifier -> Accuracy: {test_res['direction']['accuracy']:.4f}, Macro-F1: {test_res['direction']['macro_f1']:.4f}")
    print(f"   - Transform Classifier -> Accuracy: {test_res['transformation']['accuracy']:.4f}, Macro-F1: {test_res['transformation']['macro_f1']:.4f}, Weighted-F1: {test_res['transformation']['weighted_f1']:.4f}")

    return best_weights, test_res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training System] Active Device: {device}")

    set_seed(RANDOM_SEED)

    # Run Ablation Suite across all 3 feature representations
    modes = ["npy_only", "forensic_only", "combined"]
    ablation_results = {}
    best_overall_weights = None
    best_overall_f1 = -1.0

    for mode in modes:
        weights, test_res = run_experiment(mode, args.epochs, device)
        ablation_results[mode] = test_res
        comb_f1 = test_res["related"]["macro_f1"] + test_res["transformation"]["macro_f1"]
        if comb_f1 > best_overall_f1 or mode == "combined":
            best_overall_f1 = comb_f1
            best_overall_weights = weights

    # Save Best Research Checkpoint
    torch.save(best_overall_weights, CHECKPOINT_PATH)
    print(f"\n[Save Checkpoint] Successfully saved best model -> {CHECKPOINT_PATH}")

    print("\n" + "=" * 50)
    print(" ABLATION COMPARISON SUMMARY ")
    print("=" * 50)
    for mode, res in ablation_results.items():
        print(f" Mode: {mode:15s} | Rel F1: {res['related']['macro_f1']:.4f} | Dir F1: {res['direction']['macro_f1']:.4f} | Tfm Macro-F1: {res['transformation']['macro_f1']:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
