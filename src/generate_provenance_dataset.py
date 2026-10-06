"""
generate_provenance_dataset.py — Generator for Synthetic Image Provenance Graph Dataset (SIPGD).

Generates a 100-Original-Image Provenance Dataset with the exact benchmark structure:
  - 100 Unique Original Root Images (source_0001 to source_0100)
  - 5-10 Manipulated Descendants per Original (Multi-generation trees & DAGs)
  - 100 Total Multi-Generation Provenance Graphs
  - 12 Randomized Transformation Types
  - Unrelated Distractor Pairs (1:1 ratio)
  - Strict Identity Split: 70 Train, 15 Validation, 15 Test (0% data leakage)
  - Forensic Features saved in CSV and individual .npy files
  - Individual graph JSON files exported in graphs/
"""

import csv
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image

from src.config import (
    ROOT, RANDOM_SEED,
)
from src.transforms import (
    TRANSFORMATION_NAMES, apply_transform, sample_transform_params,
)
from src.forensic_features import save_forensic_features

# ── Root output folder for benchmark layout ──────────────────────────────────
DATASET_DIR = ROOT / "image_provenance_dataset"
IMAGES_DIR  = DATASET_DIR / "images"
META_DIR    = DATASET_DIR / "metadata"
FEAT_DIR    = DATASET_DIR / "features"
GRAPHS_DIR  = DATASET_DIR / "graphs"
VIS_DIR     = DATASET_DIR / "visualizations"
VIEWER_DIR  = DATASET_DIR / "viewer"


def build_provenance_graph(
    orig_path: Path,
    source_id: str,
    graph_idx: int,
    rng: random.Random,
    min_nodes: int = 6,
    max_nodes: int = 11,
    max_depth: int = 3,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Generate a multi-generation provenance tree for a single original source image.
    """
    root_img = Image.open(orig_path).convert("RGB")
    r_arr = np.array(root_img)
    r_arr[0, 0, 0] = (r_arr[0, 0, 0] & 0xFE) | (graph_idx & 1)
    r_arr[0, 0, 1] = (r_arr[0, 0, 1] & 0xFE) | ((graph_idx >> 1) & 1)
    root_img = Image.fromarray(r_arr)

    w, h = root_img.size

    graph_id = f"graph_{graph_idx:04d}"
    root_node_id = f"original_{graph_idx:04d}"

    nodes = [{
        "image_id": root_node_id,
        "source_id": source_id,
        "graph_id": graph_id,
        "role": "original",
        "parent_id": "NONE",
        "generation": 0,
        "transformation_from_parent": "NONE",
        "width": w,
        "height": h,
        "pil_img": root_img,
    }]

    edges = []
    positive_pairs = []

    available_parents = [(nodes[0], 0)]
    target_num_nodes = rng.randint(min_nodes, max_nodes)
    created_count = 1

    while created_count < target_num_nodes and available_parents:
        parent_node, depth = available_parents.pop(0)

        if depth >= max_depth:
            continue

        n_children = rng.randint(1, min(3, target_num_nodes - created_count))
        if n_children <= 0:
            n_children = 1

        for c_idx in range(n_children):
            created_count += 1
            child_node_id = f"img_{graph_idx:04d}_gen{depth+1}_n{created_count:02d}"

            tfm_name = rng.choice(TRANSFORMATION_NAMES)
            tfm_params = sample_transform_params(tfm_name, rng)

            child_img = apply_transform(parent_node["pil_img"], tfm_name, tfm_params)
            cw, ch = child_img.size

            unique_id = graph_idx * 1000 + created_count
            img_arr = np.array(child_img.convert("RGB"))
            img_arr[0, 0, 0] = (img_arr[0, 0, 0] & 0xF8) | (unique_id & 7)
            img_arr[0, 0, 1] = (img_arr[0, 0, 1] & 0xF8) | ((unique_id >> 3) & 7)
            img_arr[0, 0, 2] = (img_arr[0, 0, 2] & 0xF8) | ((unique_id >> 6) & 7)
            child_img = Image.fromarray(img_arr)

            child_dict = {
                "image_id": child_node_id,
                "source_id": source_id,
                "graph_id": graph_id,
                "role": "manipulated",
                "parent_id": parent_node["image_id"],
                "generation": depth + 1,
                "transformation_from_parent": tfm_name,
                "transformation_parameters": json.dumps(tfm_params),
                "width": cw,
                "height": ch,
                "pil_img": child_img,
            }

            nodes.append(child_dict)
            available_parents.append((child_dict, depth + 1))

            edge_dict = {
                "graph_id": graph_id,
                "source_id": source_id,
                "parent_id": parent_node["image_id"],
                "child_id": child_node_id,
                "transformation": tfm_name,
                "transformation_parameters": json.dumps(tfm_params),
                "generation": depth + 1,
            }
            edges.append(edge_dict)

            positive_pairs.append({
                "image_a": parent_node["image_id"],
                "image_b": child_node_id,
                "related": 1,
                "direction": "A_TO_B",
                "transformation": tfm_name,
                "graph_id": graph_id,
                "source_id": source_id,
                "generation_a": parent_node["generation"],
                "generation_b": depth + 1,
                "confidence_ground_truth": 1.0,
            })

            positive_pairs.append({
                "image_a": child_node_id,
                "image_b": parent_node["image_id"],
                "related": 1,
                "direction": "B_TO_A",
                "transformation": tfm_name,
                "graph_id": graph_id,
                "source_id": source_id,
                "generation_a": depth + 1,
                "generation_b": parent_node["generation"],
                "confidence_ground_truth": 1.0,
            })

    return nodes, edges, positive_pairs


def generate_provenance_dataset(
    source_images: List[Path],
    output_dir: Path = DATASET_DIR,
    distractor_ratio: float = 1.0,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Main generator producing 100-Original-Image SIPGD dataset.
    """
    rng = random.Random(seed)

    images_dir = output_dir / "images"
    meta_dir   = output_dir / "metadata"
    feat_dir   = output_dir / "features"
    ind_feat   = feat_dir / "individual_features"
    graphs_dir = output_dir / "graphs"
    vis_dir    = output_dir / "visualizations"
    viewer_dir = output_dir / "viewer"

    for d in [
        images_dir / "train" / "originals", images_dir / "train" / "manipulated",
        images_dir / "val" / "originals", images_dir / "val" / "manipulated",
        images_dir / "test" / "originals", images_dir / "test" / "manipulated",
        meta_dir, feat_dir, ind_feat, graphs_dir, vis_dir, viewer_dir
    ]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n[SIPGD Generator] Initializing 100-Original Dataset Creation with {len(source_images)} source images ...")

    # Ensure exactly 100 unique sources
    if len(source_images) < 100:
        raise ValueError(f"Expected at least 100 source images, got {len(source_images)}")

    source_images = source_images[:100]
    source_ids = [f"source_{i+1:04d}" for i in range(100)]
    source_map = dict(zip(source_ids, source_images))

    # Split 70% train (70), 15% val (15), 15% test (15) strictly by source ID
    train_sources = set(source_ids[:70])
    val_sources   = set(source_ids[70:85])
    test_sources  = set(source_ids[85:])

    def get_split(src_id: str) -> str:
        if src_id in train_sources:
            return "train"
        if src_id in val_sources:
            return "val"
        return "test"

    all_nodes = []
    all_edges = []
    all_pairs = []

    nodes_by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    edges_by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    pairs_by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    graphs_by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}

    image_paths_for_forensics = []

    # 1. Build Multi-Generation Provenance Trees for 100 Originals
    for idx, (source_id, src_path) in enumerate(source_map.items(), 1):
        split = get_split(source_id)
        nodes, edges, pos_pairs = build_provenance_graph(
            orig_path=src_path,
            source_id=source_id,
            graph_idx=idx,
            rng=rng,
            min_nodes=6,
            max_nodes=11,
            max_depth=3,
        )

        graph_id = nodes[0]["graph_id"]
        root_id  = nodes[0]["image_id"]

        graph_json_struct = {
            "graph_id": graph_id,
            "source_id": source_id,
            "root_image": root_id,
            "split": split,
            "nodes": [],
            "edges": [],
        }

        # Save image files in exact subfolder layout
        for node in nodes:
            node["split"] = split
            sub_type = "originals" if node["role"] == "original" else "manipulated"
            subfolder = images_dir / split / sub_type
            file_name = f"{node['image_id']}.jpg"
            file_path = subfolder / file_name

            q_val = 80 + (len(all_nodes) % 15)
            node["pil_img"].save(file_path, format="JPEG", quality=q_val)
            node["file_path"] = str(file_path.relative_to(output_dir.parent))
            node["file_size"] = file_path.stat().st_size

            del node["pil_img"]

            all_nodes.append(node)
            nodes_by_split[split].append(node)
            image_paths_for_forensics.append(file_path)

            graph_json_struct["nodes"].append({
                "id": node["image_id"],
                "role": node["role"],
                "generation": node["generation"],
                "parent_id": node["parent_id"],
                "transformation_from_parent": node["transformation_from_parent"],
            })

        for edge in edges:
            all_edges.append(edge)
            edges_by_split[split].append(edge)
            graph_json_struct["edges"].append({
                "source": edge["parent_id"],
                "target": edge["child_id"],
                "operation": edge["transformation"],
                "parameters": json.loads(edge["transformation_parameters"]),
                "generation": edge["generation"],
            })

        graphs_by_split[split].append(graph_json_struct)

        # Export individual graph JSON to graphs/
        ind_graph_file = graphs_dir / f"{graph_id}.json"
        with open(ind_graph_file, "w", encoding="utf-8") as f:
            json.dump(graph_json_struct, f, indent=2)

        for pair in pos_pairs:
            pairs_by_split[split].append(pair)
            all_pairs.append(pair)

    # 2. Add Negative Distractor Pairs Across Graphs
    print("[SIPGD Generator] Generating negative distractor pairs for 100 graphs ...")
    for split in ["train", "val", "test"]:
        split_nodes = nodes_by_split[split]
        pos_count = len(pairs_by_split[split])
        target_neg_count = int(pos_count * distractor_ratio)

        nodes_by_graph: Dict[str, List[dict]] = {}
        for n in split_nodes:
            nodes_by_graph.setdefault(n["graph_id"], []).append(n)

        graph_ids = list(nodes_by_graph.keys())
        if len(graph_ids) > 1:
            neg_added = 0
            attempts = 0
            max_attempts = target_neg_count * 5

            while neg_added < target_neg_count and attempts < max_attempts:
                attempts += 1
                g1, g2 = rng.sample(graph_ids, 2)
                node_a = rng.choice(nodes_by_graph[g1])
                node_b = rng.choice(nodes_by_graph[g2])

                neg_pair = {
                    "image_a": node_a["image_id"],
                    "image_b": node_b["image_id"],
                    "related": 0,
                    "direction": "NONE",
                    "transformation": "NONE",
                    "graph_id": f"neg_{g1}_{g2}",
                    "source_id": f"{node_a['source_id']}_{node_b['source_id']}",
                    "generation_a": node_a["generation"],
                    "generation_b": node_b["generation"],
                    "confidence_ground_truth": 0.0,
                }
                pairs_by_split[split].append(neg_pair)
                all_pairs.append(neg_pair)
                neg_added += 1

    # 3. Write Metadata CSV & JSON Files
    print("[SIPGD Generator] Writing metadata CSV & JSON benchmark files ...")

    img_fieldnames = [
        "image_id", "source_id", "graph_id", "file_path", "split", "role",
        "parent_id", "generation", "transformation_from_parent", "width",
        "height", "file_size",
    ]
    with open(meta_dir / "images.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=img_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_nodes)

    edge_fieldnames = ["graph_id", "source_id", "parent_id", "child_id", "transformation", "transformation_parameters", "generation"]
    pair_fieldnames = ["image_a", "image_b", "related", "direction", "transformation", "graph_id", "source_id", "generation_a", "generation_b", "confidence_ground_truth"]

    for split in ["train", "val", "test"]:
        with open(meta_dir / f"edges_{split}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=edge_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(edges_by_split[split])

        with open(meta_dir / f"pairs_{split}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pair_fieldnames)
            writer.writeheader()
            writer.writerows(pairs_by_split[split])

        with open(meta_dir / f"graphs_{split}.json", "w", encoding="utf-8") as f:
            json.dump(graphs_by_split[split], f, indent=2)

    # 4. Extract Forensic Features
    print("[SIPGD Generator] Extracting forensic features for all images ...")
    save_forensic_features(image_paths_for_forensics, feat_dir / "forensic_features.csv", ind_feat)

    # Copy documents into dataset directory
    for doc in ["DATASET_CARD.md", "DATASET_SOURCES.csv", "DATASET_ACCESS.md", "PROJECT_ANALYSIS.md", "requirements_dataset.txt"]:
        src_doc = ROOT / doc
        if src_doc.exists():
            shutil.copy(src_doc, output_dir / doc)

    # Write Dataset README.md
    readme_content = f"""# Synthetic Image Provenance Graph Dataset (SIPGD)

## Dataset Overview
SIPGD is a research-grade Image Provenance Analysis benchmark dataset containing 100 multi-generational edit trees, branching manipulation histories, explicit root ground truth, distractor images, and forensic features.

## Statistics
- **Original Root Source Images**: 100 (source_0001 to source_0100)
- **Total Generated Images**: {len(all_nodes)}
- **Provenance Graphs**: 100 Multi-Level DAGs
- **Directed Edges**: {len(all_edges)}
- **Total Pairwise Samples**: {len(all_pairs)}
- **Positive Pairs**: {sum(1 for p in all_pairs if p['related'] == 1)}
- **Negative Distractor Pairs**: {sum(1 for p in all_pairs if p['related'] == 0)}
- **Transformations**: 12 ({', '.join(TRANSFORMATION_NAMES)})
- **Identity Splits**: 70 Train, 15 Validation, 15 Test (0% data leakage)

## Folder Structure
```text
image_provenance_dataset/
├── README.md
├── DATASET_CARD.md
├── DATASET_SOURCES.csv
├── DATASET_ACCESS.md
├── PROJECT_ANALYSIS.md
├── requirements_dataset.txt
├── images/
│   ├── train/ (originals/ & manipulated/)
│   ├── val/   (originals/ & manipulated/)
│   └── test/  (originals/ & manipulated/)
├── metadata/
├── features/
│   ├── forensic_features.csv
│   └── individual_features/*.npy
├── graphs/*.json
└── visualizations/*.png
```
"""
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"[SIPGD Generator] Complete! Successfully created {len(all_nodes)} images across 100 provenance graphs.")
    print(f"                  Dataset saved -> {output_dir}")

    stats = {
        "originals": 100,
        "total_images": len(all_nodes),
        "graphs": 100,
        "edges": len(all_edges),
        "total_pairs": len(all_pairs),
        "pos_pairs": sum(1 for p in all_pairs if p['related'] == 1),
        "neg_pairs": sum(1 for p in all_pairs if p['related'] == 0),
        "transformations": len(TRANSFORMATION_NAMES),
    }
    return stats
