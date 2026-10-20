"""
tests/audit_verify_metrics.py — Gate 2 independent metric recomputation.
Loads saved checkpoints from models/phase4_v1/ and evaluates them directly on pairs_test.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import json
import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)
from src.train_phase4_models import (
    RelationshipMLP, MultiLabelTransformationMLP, AntiSymmetricDirectionMLP,
    load_feature_cache, prepare_pair_matrices, TRANSFORMATION_CLASSES
)

def run_audit():
    print("[Audit] Loading feature cache and test split manifest...")
    feat_cache = load_feature_cache()
    test_df = pd.read_csv("data/provenance_dataset_v2/metadata/pairs_test.csv")
    print(f"[Audit] Test pairs count: {len(test_df)}")

    X_test_115, X_test_172_ab, X_test_172_ba, y_test_rel, y_test_tfm, y_test_dir = prepare_pair_matrices(test_df, feat_cache)

    scaler_115 = joblib.load("models/phase4_v1/scaler_115.joblib")
    scaler_172 = joblib.load("models/phase4_v1/scaler_172.joblib")

    X_test_115_s = scaler_115.transform(X_test_115)
    X_test_172_ab_s = scaler_172.transform(X_test_172_ab)
    X_test_172_ba_s = scaler_172.transform(X_test_172_ba)

    # 1. Audit Relationship Model
    ckpt_rel = torch.load("models/phase4_v1/relationship_model.pt", map_location="cpu", weights_only=False)
    rel_m = RelationshipMLP(115, 64)
    rel_m.load_state_dict(ckpt_rel["state_dict"])
    rel_m.eval()

    with torch.no_grad():
        t_115 = torch.tensor(X_test_115_s, dtype=torch.float32)
        probs_rel = torch.sigmoid(rel_m(t_115)).numpy()

    thresh_rel = ckpt_rel["threshold"]
    preds_rel = (probs_rel >= thresh_rel).astype(int)

    prec_rel = precision_score(y_test_rel, preds_rel)
    rec_rel = recall_score(y_test_rel, preds_rel)
    f1_rel = f1_score(y_test_rel, preds_rel)
    auc_rel = roc_auc_score(y_test_rel, probs_rel)
    ap_rel = average_precision_score(y_test_rel, probs_rel)
    cm_rel = confusion_matrix(y_test_rel, preds_rel)
    tn, fp, fn, tp = cm_rel.ravel()
    fpr_rel = fp / (fp + tn)

    print("\n--- Relationship Model Recomputed Metrics ---")
    print(f"Optimal Threshold: {thresh_rel}")
    print(f"Precision: {prec_rel:.4f}")
    print(f"Recall:    {rec_rel:.4f}")
    print(f"F1 Score:  {f1_rel:.4f}")
    print(f"ROC-AUC:   {auc_rel:.4f}")
    print(f"PR-AUC:    {ap_rel:.4f}")
    print(f"FPR:       {fpr_rel:.4f}")
    print(f"Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # 2. Audit Transformation Model
    ckpt_tfm = torch.load("models/phase4_v1/transformation_model.pt", map_location="cpu", weights_only=False)
    tfm_m = MultiLabelTransformationMLP(115, 12, 128)
    tfm_m.load_state_dict(ckpt_tfm["state_dict"])
    tfm_m.eval()

    test_rel_mask = (y_test_rel == 1.0)
    X_test_tfm_subset = torch.tensor(X_test_115_s[test_rel_mask], dtype=torch.float32)
    y_test_tfm_subset = y_test_tfm[test_rel_mask]

    with torch.no_grad():
        probs_tfm = torch.sigmoid(tfm_m(X_test_tfm_subset)).numpy()

    per_class_th = ckpt_tfm["per_class_thresholds"]
    preds_tfm = np.zeros_like(probs_tfm, dtype=int)
    for c in range(12):
        preds_tfm[:, c] = (probs_tfm[:, c] >= per_class_th[c]).astype(int)

    macro_f1 = f1_score(y_test_tfm_subset, preds_tfm, average="macro", zero_division=0)
    micro_f1 = f1_score(y_test_tfm_subset, preds_tfm, average="micro", zero_division=0)

    print("\n--- Transformation Model Recomputed Metrics ---")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")

    # 3. Audit Direction Model
    ckpt_dir = torch.load("models/phase4_v1/direction_model.pt", map_location="cpu", weights_only=False)
    dir_m = AntiSymmetricDirectionMLP(172, 64)
    dir_m.load_state_dict(ckpt_dir["state_dict"])
    dir_m.eval()

    X_test_dir_ab = torch.tensor(X_test_172_ab_s[test_rel_mask], dtype=torch.float32)
    X_test_dir_ba = torch.tensor(X_test_172_ba_s[test_rel_mask], dtype=torch.float32)
    y_test_dir_subset = y_test_dir[test_rel_mask]

    with torch.no_grad():
        logits_ab = dir_m(X_test_dir_ab, X_test_dir_ba).numpy()
        logits_ba = dir_m(X_test_dir_ba, X_test_dir_ab).numpy()
        probs_ab = 1.0 / (1.0 + np.exp(-logits_ab))
        probs_ba = 1.0 / (1.0 + np.exp(-logits_ba))

    swap_err = np.abs(probs_ab + probs_ba - 1.0).max()
    raw_acc = ((probs_ab >= 0.5).astype(float) == y_test_dir_subset).mean()

    delta = ckpt_dir["delta_uncertain"]
    confident = np.abs(probs_ab - 0.5) >= delta
    cov = confident.mean()
    conf_acc = ((probs_ab[confident] >= 0.5).astype(float) == y_test_dir_subset[confident]).mean() if cov > 0 else 1.0

    print("\n--- Direction Model Recomputed Metrics ---")
    print(f"Swap Error (Max):      {swap_err:.10f}")
    print(f"Raw Accuracy:          {raw_acc:.4f}")
    print(f"Delta Uncertain:       {delta:.4f}")
    print(f"Coverage:              {cov*100:.2f}%")
    print(f"Confident Accuracy:    {conf_acc:.4f}")

if __name__ == "__main__":
    run_audit()
