"""
train.py — End-to-end training pipeline.

Steps
-----
1. Generate dataset (if not already done).
2. Extract ViT embeddings (with caching).
3. Train direction MLP.
4. Train transformation MLP.
5. Save models to models/.

Usage
-----
    python train.py
    python train.py --overwrite   # regenerate child images
    python train.py --epochs 100  # override epoch count
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch

from src.config import (
    EDGES_CSV, MODELS_DIR, RANDOM_SEED, MLP_EPOCHS, TRANSFORMATION_NAMES,
)
from src.generate_dataset import generate_dataset
from src.feature_extractor import extract_embeddings_cached
from src.dataset import (
    DirectionDataset, TransformationDataset,
    split_by_original, DIRECTION_NAMES, TRANSFORM_NAMES,
)
from src.train_classifiers import (
    train_classifier,
    save_direction_model, save_transform_model,
    MLP,
)
from src.metrics import evaluate_classifier
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="Train provenance classifiers")
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate child images even if they exist")
    p.add_argument("--epochs", type=int, default=MLP_EPOCHS,
                   help=f"Training epochs (default {MLP_EPOCHS})")
    p.add_argument("--seed", type=int, default=RANDOM_SEED,
                   help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # ── 1. Generate dataset ───────────────────────────────────────────────────
    print("\n[Step 1] Generating dataset …")
    generate_dataset(seed=args.seed, overwrite=args.overwrite)

    # ── 2. Load edges CSV ─────────────────────────────────────────────────────
    print("\n[Step 2] Loading edges …")
    df = pd.read_csv(EDGES_CSV)
    print(f"  {len(df)} edges loaded.")

    # ── 3. Extract embeddings ───────────────────────────────────────────────ject
    all_paths = list(set(
        [Path(p) for p in df["parent_path"].tolist()] +
        [Path(p) for p in df["child_path"].tolist()]
    ))
    embeddings = extract_embeddings_cached(all_paths, device=device)

    # ── 4. Split ──────────────────────────────────────────────────────────────
    print("\n[Step 4] Splitting by original identity …")
    train_df, val_df, test_df = split_by_original(df, seed=args.seed)
    print(f"  Train originals : {train_df['parent_path'].nunique()}")
    print(f"  Val originals   : {val_df['parent_path'].nunique()}")
    print(f"  Test originals  : {test_df['parent_path'].nunique()}")

    # ── 5. Build datasets ─────────────────────────────────────────────────────
    dir_train = DirectionDataset(train_df, embeddings)
    dir_val   = DirectionDataset(val_df,   embeddings)
    dir_test  = DirectionDataset(test_df,  embeddings)

    tfm_train = TransformationDataset(train_df, embeddings)
    tfm_val   = TransformationDataset(val_df,   embeddings)
    tfm_test  = TransformationDataset(test_df,  embeddings)

    # Sanity check
    if len(dir_train) == 0:
        print("[ERROR] Direction training set is empty — check that embeddings "
              "were generated and paths match.")
        sys.exit(1)

    # ── 6. Train direction classifier ─────────────────────────────────────────
    print("\n[Step 5] Training direction classifier …")
    import src.config as cfg
    cfg.MLP_EPOCHS = args.epochs   # honour CLI override

    dir_model, dir_val_acc, dir_test_acc = train_classifier(
        dir_train, dir_val, dir_test,
        n_classes=2,
        label="Direction",
        device=device,
        seed=args.seed,
    )
    save_direction_model(dir_model)

    # ── 7. Train transformation classifier ────────────────────────────────────
    print("\n[Step 6] Training transformation classifier …")
    tfm_model, tfm_val_acc, tfm_test_acc = train_classifier(
        tfm_train, tfm_val, tfm_test,
        n_classes=len(TRANSFORMATION_NAMES),
        label="Transformation",
        device=device,
        seed=args.seed,
    )
    save_transform_model(tfm_model)

    # ── 8. Detailed evaluation ────────────────────────────────────────────────
    from src.train_classifiers import MLP
    from src.config import MLP_HIDDEN_DIM, MLP_DROPOUT
    from src.pair_features import PAIR_FEATURE_DIM

    dir_test_loader = DataLoader(dir_test,  batch_size=32, shuffle=False)
    tfm_test_loader = DataLoader(tfm_test,  batch_size=32, shuffle=False)

    evaluate_classifier(dir_model, dir_test_loader, DIRECTION_NAMES,
                        device, "Direction — Test Set")
    evaluate_classifier(tfm_model, tfm_test_loader, TRANSFORM_NAMES,
                        device, "Transformation — Test Set")

    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55)
    print(f"  Direction    val={dir_val_acc:.4f}  test={dir_test_acc:.4f}")
    print(f"  Transform    val={tfm_val_acc:.4f}  test={tfm_test_acc:.4f}")
    print(f"  Models saved -> {MODELS_DIR}")
    print()


if __name__ == "__main__":
    main()
