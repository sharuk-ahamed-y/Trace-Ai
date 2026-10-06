"""
metrics.py — Evaluation utilities for direction and transformation classifiers.

Provides accuracy, per-class F1, and a confusion-matrix printer.
"""

from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    title: str = "Evaluation",
) -> float:
    """
    Run inference on *loader* and print a classification report.

    Returns overall accuracy.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y.tolist())

    if not all_labels:
        print(f"[{title}] No samples to evaluate.")
        return 0.0

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc = correct / len(all_labels)

    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")
    print(f"  Accuracy: {acc:.4f}  ({correct}/{len(all_labels)})\n")
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0,
    ))

    cm = confusion_matrix(all_labels, all_preds)
    print("  Confusion matrix:")
    header = "  " + "  ".join(f"{n[:6]:>6}" for n in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:6d}" for v in row)
        print(f"  {class_names[i][:6]:>6}  {row_str}")
    print()

    return acc
