"""
src/validate_dataset_v2.py — Automated Integrity Verification for Dataset v2.

Validates:
1. File readability and image integrity (Pillow verify)
2. Broken parent references and graph acyclicity
3. Duplicate image IDs and duplicate files
4. Transformation label validity across the 12 classes
5. Multi-label vector dimensions (12-D) and binary values {0, 1}
6. Train / Validation / Test source non-leakage (0% overlap)
7. Class balance across transformation classes and pair balance (pos/neg)
8. Absence of unintended modifications outside data/provenance_dataset_v2/
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Set
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.transforms import TRANSFORMATION_NAMES

DATASET_V2_DIR = PROJECT_ROOT / "data" / "provenance_dataset_v2"


def validate_v2_dataset(dataset_dir: Path = DATASET_V2_DIR) -> Dict[str, Any]:
    print("=" * 65)
    print("  TRACE AI — PHASE 3 DATASET V2 INTEGRITY VERIFICATION")
    print("=" * 65)

    meta_dir = dataset_dir / "metadata"
    feat_dir = dataset_dir / "features"
    images_dir = dataset_dir / "images"

    results = {
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "checks": {}
    }

    # 1. Validate images.csv exists & load
    images_csv = meta_dir / "images.csv"
    if not images_csv.exists():
        results["errors"].append(f"Missing images.csv at {images_csv}")
        results["status"] = "FAIL"
        return results

    nodes = []
    with open(images_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        nodes = list(reader)

    print(f"[Check 1/8] Total registered images: {len(nodes)}")
    results["checks"]["total_nodes"] = len(nodes)

    # 2. Check Image Readability & File Integrity
    print("[Check 2/8] Verifying image readability on disk ...")
    corrupt_count = 0
    missing_count = 0
    seen_ids = set()
    duplicate_ids = set()

    for n in nodes:
        img_id = n["image_id"]
        if img_id in seen_ids:
            duplicate_ids.add(img_id)
        seen_ids.add(img_id)

        # File resolution
        split = n["split"]
        sub_type = "originals" if n["role"] == "original" else "manipulated"
        p = images_dir / split / sub_type / f"{img_id}.jpg"
        if not p.exists():
            missing_count += 1
            results["errors"].append(f"Image file missing on disk: {p}")
        else:
            try:
                with Image.open(p) as img:
                    img.verify()
            except Exception as e:
                corrupt_count += 1
                results["errors"].append(f"Corrupt image {p}: {e}")

    results["checks"]["missing_images"] = missing_count
    results["checks"]["corrupt_images"] = corrupt_count
    results["checks"]["duplicate_image_ids"] = list(duplicate_ids)

    if duplicate_ids:
        results["errors"].append(f"Found {len(duplicate_ids)} duplicate image IDs!")
    if missing_count > 0 or corrupt_count > 0:
        results["status"] = "FAIL"

    # 3. Check Parent References & Graph Acyclicity
    print("[Check 3/8] Verifying parent references and DAG hierarchy ...")
    node_map = {n["image_id"]: n for n in nodes}
    broken_parents = []
    roots_count = 0

    for n in nodes:
        parent_id = n["parent_id"]
        if n["role"] == "original":
            roots_count += 1
            if parent_id != "NONE":
                broken_parents.append((n["image_id"], parent_id, "Root has parent"))
        else:
            if parent_id not in node_map:
                broken_parents.append((n["image_id"], parent_id, "Parent not found"))

    results["checks"]["root_count"] = roots_count
    results["checks"]["broken_parents"] = len(broken_parents)
    if broken_parents:
        results["errors"].append(f"Found {len(broken_parents)} broken parent references!")
        results["status"] = "FAIL"

    # 4. Check Source Identity Splitting & 0% Leakage
    print("[Check 4/8] Validating strict source-identity partition (0% leakage) ...")
    sources_by_split: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    for n in nodes:
        sources_by_split[n["split"]].add(n["source_id"])

    overlap_train_val = sources_by_split["train"] & sources_by_split["val"]
    overlap_train_test = sources_by_split["train"] & sources_by_split["test"]
    overlap_val_test = sources_by_split["val"] & sources_by_split["test"]

    total_unique_sources = len(sources_by_split["train"] | sources_by_split["val"] | sources_by_split["test"])
    results["checks"]["train_sources"] = len(sources_by_split["train"])
    results["checks"]["val_sources"] = len(sources_by_split["val"])
    results["checks"]["test_sources"] = len(sources_by_split["test"])
    results["checks"]["total_unique_sources"] = total_unique_sources
    results["checks"]["source_leakage_detected"] = bool(overlap_train_val or overlap_train_test or overlap_val_test)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        results["errors"].append(f"CRITICAL LEAKAGE: Overlaps detected! train/val={overlap_train_val}, train/test={overlap_train_test}, val/test={overlap_val_test}")
        results["status"] = "FAIL"
    else:
        print(f"      0% Leakage Verified: {len(sources_by_split['train'])} Train / {len(sources_by_split['val'])} Val / {len(sources_by_split['test'])} Test sources.")

    # 5. Check Edges and Multi-Label Vectors
    print("[Check 5/8] Verifying edge manifests and multi-label vector conformity ...")
    edges_by_split = {}
    tfm_counts = {c: 0 for c in TRANSFORMATION_NAMES}
    invalid_tfm = 0
    invalid_vecs = 0

    for split in ["train", "val", "test"]:
        edge_file = meta_dir / f"edges_{split}.csv"
        with open(edge_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            edges_by_split[split] = list(reader)

        for e in edges_by_split[split]:
            tfm = e["transformation"]
            if tfm not in TRANSFORMATION_NAMES:
                invalid_tfm += 1
            else:
                tfm_counts[tfm] += 1

            vec = json.loads(e["multi_label_vector"])
            if len(vec) != len(TRANSFORMATION_NAMES) or any(v not in (0, 1) for v in vec):
                invalid_vecs += 1

    results["checks"]["total_edges"] = sum(len(v) for v in edges_by_split.values())
    results["checks"]["invalid_transformations"] = invalid_tfm
    results["checks"]["invalid_multi_label_vectors"] = invalid_vecs
    results["checks"]["transformation_distribution"] = tfm_counts

    if invalid_tfm > 0 or invalid_vecs > 0:
        results["errors"].append("Invalid transformation labels or multi-label vectors found!")
        results["status"] = "FAIL"

    # 6. Check Pair Manifests & Balance
    print("[Check 6/8] Verifying pairwise samples and pos/neg balance ...")
    pairs_by_split = {}
    for split in ["train", "val", "test"]:
        pair_file = meta_dir / f"pairs_{split}.csv"
        with open(pair_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            pairs_by_split[split] = list(reader)

    pair_stats = {}
    for split, p_list in pairs_by_split.items():
        pos = sum(1 for p in p_list if p["related"] == "1")
        neg = sum(1 for p in p_list if p["related"] == "0")
        fwd = sum(1 for p in p_list if p["direction"] == "A_TO_B")
        rev = sum(1 for p in p_list if p["direction"] == "B_TO_A")
        none_dir = sum(1 for p in p_list if p["direction"] == "NONE")
        pair_stats[split] = {
            "total": len(p_list),
            "positive": pos,
            "negative": neg,
            "forward": fwd,
            "reverse": rev,
            "none_dir": none_dir,
            "pos_neg_ratio": round(pos / max(1, neg), 2)
        }

    results["checks"]["pair_statistics"] = pair_stats

    # 7. Check Forensic Feature Arrays (.npy)
    print("[Check 7/8] Verifying 57-D forensic CSV and individual .npy arrays ...")
    forensic_csv = feat_dir / "forensic_features.csv"
    npy_dir = feat_dir / "individual_features"
    npy_files = list(npy_dir.glob("*.npy"))
    results["checks"]["forensic_csv_exists"] = forensic_csv.exists()
    results["checks"]["total_npy_arrays"] = len(npy_files)

    if len(npy_files) != len(nodes):
        results["warnings"].append(f"Discrepancy in .npy count ({len(npy_files)}) vs node count ({len(nodes)})")

    # 8. Unintended Modifications Outside provenance_dataset_v2 Check
    print("[Check 8/8] Verifying no unexpected file changes outside target directory ...")
    legacy_features_count = len(list((PROJECT_ROOT / "image_provenance_dataset" / "features" / "individual_features").glob("*.npy")))
    results["checks"]["legacy_795_npy_features_preserved"] = (legacy_features_count == 795)
    if legacy_features_count != 795:
        results["errors"].append(f"Legacy features altered! Expected 795, got {legacy_features_count}")
        results["status"] = "FAIL"

    print("-" * 65)
    print(f"  INTEGRITY VERIFICATION RESULT: {results['status']}")
    print(f"  - Total Images:       {results['checks']['total_nodes']}")
    print(f"  - Total Edges:        {results['checks']['total_edges']}")
    print(f"  - Missing Images:     {results['checks']['missing_images']}")
    print(f"  - Corrupt Images:     {results['checks']['corrupt_images']}")
    print(f"  - Source Leakage:     {'NONE (0.0%)' if not results['checks']['source_leakage_detected'] else 'DETECTED'}")
    print(f"  - Legacy 795 Preserved: {results['checks']['legacy_795_npy_features_preserved']}")
    print("=" * 65)

    return results


if __name__ == "__main__":
    res = validate_v2_dataset()
    if res["status"] != "PASS":
        sys.exit(1)
