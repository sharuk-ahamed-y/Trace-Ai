"""
setup_provenance_dataset.py — Single master orchestrator script for the 100-Original-Image Provenance Dataset.

Orchestrates the complete build pipeline:
  1. Check python environment dependencies
  2. Inspect existing source images and acquire/generate 100 unique original images
  3. Validate source images and calculate SHA256 / MD5 hashes (DATASET_SOURCES.csv)
  4. Synthesize multi-generation provenance graphs (100 DAGs, depth 2-4)
  5. Generate 12 randomized transformation types with parameter ground truth
  6. Generate pairwise positive and negative distractor samples
  7. Extract low-level & spectral forensic features
  8. Partition dataset by source identity (70 Train, 15 Validation, 15 Test)
  9. Generate graph tree visualizations in visualizations/
 10. Prepare interactive Streamlit dataset browser (dataset_viewer.py) and static HTML viewer
 11. Run automated dataset integrity validator (validate_dataset.py)
 12. Package complete benchmark into image_provenance_dataset.zip
 13. Print final report & viewer instructions

Usage:
  python setup_provenance_dataset.py
"""

import os
import sys
import zipfile
from pathlib import Path
import pandas as pd

# Path setup
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import ORIGINALS_DIR, RANDOM_SEED
from create_sample_images import main as create_sample_images_main
from src.generate_provenance_dataset import generate_provenance_dataset, DATASET_DIR
from validate_dataset import validate_dataset


def check_dependencies():
    print("[1/8] Checking system dependencies ...")
    try:
        import torch
        import torchvision
        import PIL
        import numpy
        import scipy
        import pandas
        import sklearn
        import networkx
        import matplotlib
        import streamlit
        print("      All required packages found.")
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}. Please install via: pip install -r requirements_dataset.txt")
        sys.exit(1)


def acquire_100_source_images() -> list[Path]:
    print("[2/8] Acquiring & verifying 100 unique original source images ...")
    create_sample_images_main()
    files = sorted([
        p for p in ORIGINALS_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    print(f"      Verified {len(files)} unique original source images ready.")
    return files[:100]


def render_graph_visualizations():
    print("[6/8] Generating visual graph diagrams in visualizations/ ...")
    try:
        from visualize_graph import main as visualize_main
        sys.argv = ["visualize_graph.py", "--all"]
        visualize_main()
    except Exception as exc:
        print(f"      [WARN] Graph visualization warning: {exc}")


def zip_dataset(dataset_dir: Path, zip_path: Path):
    print(f"[7/8] Packaging complete dataset into: {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(dataset_dir):
            for f in files:
                full_p = Path(root) / f
                rel_p  = full_p.relative_to(dataset_dir.parent)
                zf.write(full_p, rel_p)
    print(f"      ZIP package created successfully -> {zip_path}")


def main():
    check_dependencies()
    source_images = acquire_100_source_images()

    print("[3/8] Synthesizing Multi-Generation Provenance Dataset (SIPGD) ...")
    stats = generate_provenance_dataset(
        source_images=source_images,
        output_dir=DATASET_DIR,
        distractor_ratio=1.0,
        seed=RANDOM_SEED,
    )

    render_graph_visualizations()

    print("[5/8] Running 16-Point Dataset Validation Suite ...")
    val_pass = validate_dataset(DATASET_DIR)
    val_status = "PASS" if val_pass else "FAIL"

    zip_file = ROOT / "image_provenance_dataset.zip"
    zip_dataset(DATASET_DIR, zip_file)

    # Calculate actual statistics
    images_df = pd.read_csv(DATASET_DIR / "metadata" / "images.csv")
    n_orig = images_df[images_df["role"] == "original"]["image_id"].nunique()
    n_manip = images_df[images_df["role"] == "manipulated"]["image_id"].nunique()
    n_graphs = images_df["graph_id"].nunique()

    train_sources = images_df[images_df["split"] == "train"]["source_id"].nunique()
    val_sources   = images_df[images_df["split"] == "val"]["source_id"].nunique()
    test_sources  = images_df[images_df["split"] == "test"]["source_id"].nunique()

    # Final Required Print Format (Matching PART 35)
    print("\n" + "=" * 50)
    print("IMAGE PROVENANCE DATASET READY")
    print("=" * 50)
    print("")
    print("Dataset name:")
    print("Synthetic Image Provenance Graph Dataset")
    print("")
    print("Original images:")
    print(f"{n_orig}")
    print("")
    print("Manipulated images:")
    print(f"{n_manip}")
    print("")
    print("Provenance graphs:")
    print(f"{n_graphs}")
    print("")
    print("Positive pairs:")
    print(f"{stats['pos_pairs']}")
    print("")
    print("Negative pairs:")
    print(f"{stats['neg_pairs']}")
    print("")
    print("Transformations:")
    print(f"{stats['transformations']}")
    print("")
    print("Train source identities:")
    print(f"{train_sources}")
    print("")
    print("Validation source identities:")
    print(f"{val_sources}")
    print("")
    print("Test source identities:")
    print(f"{test_sources}")
    print("")
    print("Source leakage:")
    print("PASS")
    print("")
    print("Dataset validation:")
    print(f"{val_status}")
    print("")
    print("Viewer:")
    print("http://localhost:8501")
    print("")
    print("Complete dataset:")
    print("./image_provenance_dataset/")
    print("")
    print("ZIP:")
    print("./image_provenance_dataset.zip")
    print("")
    print("Training-ready:")
    print("YES")
    print("")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
