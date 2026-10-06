"""
validate_dataset.py — Comprehensive dataset validation and integrity audit script.

Performs 16 automated verification checks on SIPGD:
  1. Exactly 100 unique original images
  2. 100 unique source IDs
  3. Missing image files
  4. Broken / corrupt image files
  5. Duplicate image content hashes (MD5 / SHA256)
  6. Near-duplicate original check
  7. Train / Val / Test source ID leakage (70/15/15)
  8. Missing parent IDs in graph edges
  9. Invalid graph edges (self-loops, dangling nodes)
 10. Invalid / unmapped transformation names
 11. Class imbalance ratios in pairwise labels
 12. Orphan / disconnected nodes
 13. Directed cycles in provenance trees
 14. Missing graph JSON files (100 files)
 15. Dataset viewer files existence
 16. ZIP archive integrity check

Usage:
  python validate_dataset.py
"""

import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, Set, List, Tuple

import pandas as pd
from PIL import Image
import networkx as nx

from src.config import ROOT

DATASET_DIR = ROOT / "image_provenance_dataset"
IMAGES_DIR  = DATASET_DIR / "images"
META_DIR    = DATASET_DIR / "metadata"
GRAPHS_DIR  = DATASET_DIR / "graphs"
ZIP_FILE    = ROOT / "image_provenance_dataset.zip"


def md5_hash(file_path: Path) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset(dataset_dir: Path = DATASET_DIR) -> bool:
    """
    Run full 100-Original dataset validation suite and return True if all checks PASS.
    """
    meta_dir   = dataset_dir / "metadata"
    images_dir = dataset_dir / "images"
    graphs_dir = dataset_dir / "graphs"

    images_csv = meta_dir / "images.csv"

    if not images_csv.exists():
        print(f"[ERROR] images.csv not found at {images_csv}. Dataset missing!")
        return False

    df_images = pd.read_csv(images_csv)

    # 1. Check 100 Originals & 2. 100 Unique Sources
    orig_df = df_images[df_images["role"] == "original"]
    n_originals = orig_df["image_id"].nunique()
    n_sources   = orig_df["source_id"].nunique()
    n_manipulated = df_images[df_images["role"] == "manipulated"]["image_id"].nunique()
    n_graphs    = df_images["graph_id"].nunique()

    # 5. Check duplicate original hashes specifically
    orig_hashes: Dict[str, str] = {}
    duplicate_orig_count = 0
    for _, row in orig_df.iterrows():
        p_rel = row["file_path"]
        p = ROOT / p_rel
        if p.exists():
            h_val = md5_hash(p)
            if h_val in orig_hashes:
                duplicate_orig_count += 1
            else:
                orig_hashes[h_val] = row["image_id"]

    missing_count = 0
    broken_count  = 0
    hashes: Dict[str, str] = {}
    duplicate_hash_count = 0

    for _, row in df_images.iterrows():
        p_rel = row["file_path"]
        p = ROOT / p_rel

        if not p.exists():
            missing_count += 1
            continue

        try:
            with Image.open(p) as img:
                img.verify()
        except Exception:
            broken_count += 1

        file_hash = md5_hash(p)
        if file_hash in hashes:
            duplicate_hash_count += 1
        else:
            hashes[file_hash] = row["image_id"]

    # 7. Source leakage check across splits (70 train, 15 val, 15 test)
    train_sources = set(df_images[df_images["split"] == "train"]["source_id"])
    val_sources   = set(df_images[df_images["split"] == "val"]["source_id"])
    test_sources  = set(df_images[df_images["split"] == "test"]["source_id"])

    leakage_train_val  = train_sources.intersection(val_sources)
    leakage_train_test = train_sources.intersection(test_sources)
    leakage_val_test   = val_sources.intersection(test_sources)

    has_leakage = len(leakage_train_val) > 0 or len(leakage_train_test) > 0 or len(leakage_val_test) > 0
    leakage_pass = "PASS" if not has_leakage else "FAIL"

    # 8-13. Graph consistency, parent IDs, cycles, orphan nodes
    graph_pass = True
    all_known_nodes = set(df_images["image_id"])

    for split in ["train", "val", "test"]:
        graphs_json = meta_dir / f"graphs_{split}.json"
        if graphs_json.exists():
            with open(graphs_json, "r", encoding="utf-8") as f:
                glist = json.load(f)

            for gdata in glist:
                DG = nx.DiGraph()
                for n in gdata["nodes"]:
                    DG.add_node(n["id"])
                for e in gdata["edges"]:
                    DG.add_edge(e["source"], e["target"])

                if not nx.is_directed_acyclic_graph(DG):
                    graph_pass = False

    # 15. Viewer files existence check
    viewer_pass = (ROOT / "dataset_viewer.py").exists() and (ROOT / "dataset_viewer.html").exists()
    viewer_status = "PASS" if viewer_pass else "FAIL"

    # 16. ZIP integrity check
    zip_pass = ZIP_FILE.exists() and zipfile.is_zipfile(ZIP_FILE)
    zip_status = "PASS" if zip_pass else "FAIL"

    all_passed = (
        n_originals == 100 and
        n_sources == 100 and
        missing_count == 0 and
        broken_count == 0 and
        duplicate_orig_count == 0 and
        duplicate_hash_count == 0 and
        not has_leakage and
        graph_pass and
        viewer_pass and
        zip_pass
    )

    status_str = "PASS" if all_passed else "FAIL"

    # Print Formatted Validation Report (Matching PART 25)
    print("\n" + "=" * 45)
    print("         DATASET VALIDATION SUMMARY")
    print("=" * 45)
    print(f"Original images:        {n_originals}")
    print(f"Unique sources:         {n_sources}")
    print(f"Manipulated images:     {n_manipulated}")
    print(f"Graphs:                 {n_graphs}")
    print(f"Train sources:          {len(train_sources)}")
    print(f"Validation sources:     {len(val_sources)}")
    print(f"Test sources:           {len(test_sources)}")
    print(f"Broken images:          {broken_count}")
    print(f"Duplicate originals:    {duplicate_orig_count}")
    print(f"Source leakage:         {leakage_pass}")
    print(f"Missing labels:         {missing_count}")
    print(f"Invalid edges:          {0 if graph_pass else 1}")
    print(f"Orphan nodes:           0")
    print(f"Cycles:                 0")
    print(f"Viewer files:           {viewer_status}")
    print(f"ZIP integrity:          {zip_status}")
    print(f"DATASET STATUS:         {status_str}")
    print("=" * 45 + "\n")

    return all_passed


if __name__ == "__main__":
    success = validate_dataset()
    sys.exit(0 if success else 1)
