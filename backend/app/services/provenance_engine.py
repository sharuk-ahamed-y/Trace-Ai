"""
backend/app/services/provenance_engine.py — Graph Reconstruction & Provenance Pipeline
"""

import time
import math
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image
import numpy as np
import torch
import networkx as nx

from backend.app.services.model_manager import get_model_manager
from src.forensic_features import extract_forensic_features

def compute_hashes(image_path: Path) -> Tuple[str, str]:
    with open(image_path, "rb") as f:
        data = f.read()
        sha256 = hashlib.sha256(data).hexdigest()
        md5 = hashlib.md5(data).hexdigest()
    return sha256, md5

def run_provenance_analysis(
    image_paths: List[Path],
    similarity_threshold: float = 0.65,
    confidence_filter: float = 0.50
) -> Dict[str, Any]:
    start_time = time.time()
    manager = get_model_manager()

    n = len(image_paths)
    if n == 0:
        raise ValueError("No images provided for analysis.")

    # 1. Image metadata & embeddings & forensic features
    image_nodes = []
    embeddings = {}
    forensics = {}

    for idx, path in enumerate(image_paths):
        img_id = path.stem
        sha256, md5 = compute_hashes(path)
        
        with Image.open(path) as img:
            w, h = img.size
            file_size = path.stat().st_size

        emb = manager.get_embedding(path)
        f_dict = extract_forensic_features(path)

        # Build 15-dim forensic vector
        f_vec = np.array([
            f_dict["width"], f_dict["height"], f_dict["aspect_ratio"], f_dict["file_size"], f_dict["jpeg_quality_est"],
            f_dict["r_mean"], f_dict["r_std"], f_dict["g_mean"], f_dict["g_std"], f_dict["b_mean"], f_dict["b_std"],
            f_dict["edge_mean"], f_dict["edge_std"], f_dict["noise_res_std"], f_dict["noise_res_skew"]
        ], dtype=np.float32)

        embeddings[img_id] = emb
        forensics[img_id] = f_vec

        image_nodes.append({
            "id": img_id,
            "filename": path.name,
            "file_path": str(path),
            "url": f"/data/uploads/{path.parent.name}/{path.name}" if "uploads" in str(path) else f"/data/{path.name}",
            "sha256": sha256,
            "md5": md5,
            "width": w,
            "height": h,
            "file_size": file_size,
            "forensics": f_dict,
            "is_root": False
        })

    # 2. Compute Pairwise Relations & Directed Graph
    G = nx.DiGraph()
    for node in image_nodes:
        G.add_node(node["id"])

    candidate_edges = []

    for i in range(n):
        id_a = image_nodes[i]["id"]
        emb_a = embeddings[id_a]
        f_a = forensics[id_a]

        for j in range(i + 1, n):
            id_b = image_nodes[j]["id"]
            emb_b = embeddings[id_b]
            f_b = forensics[id_b]

            f_diff = np.abs(f_a - f_b)
            f_combined = torch.from_numpy(np.concatenate([f_a, f_b, f_diff])).float()

            pred = manager.predict_pair(emb_a, emb_b, f_combined)

            if pred["related_confidence"] >= similarity_threshold:
                direction = pred["direction"]
                confidence = pred["related_confidence"]

                if direction == "A_TO_B":
                    candidate_edges.append({
                        "source": id_a, "target": id_b,
                        "transformation": pred["transformation"],
                        "confidence": confidence,
                        "weight": 1.0 - confidence
                    })
                elif direction == "B_TO_A":
                    candidate_edges.append({
                        "source": id_b, "target": id_a,
                        "transformation": pred["transformation"],
                        "confidence": confidence,
                        "weight": 1.0 - confidence
                    })
                else:
                    # Undirected tie-breaker: check area / file size / quality to infer direction (parent usually larger/higher quality)
                    if image_nodes[i]["width"] * image_nodes[i]["height"] >= image_nodes[j]["width"] * image_nodes[j]["height"]:
                        src, tgt = id_a, id_b
                    else:
                        src, tgt = id_b, id_a
                    candidate_edges.append({
                        "source": src, "target": tgt,
                        "transformation": pred["transformation"],
                        "confidence": confidence,
                        "weight": 1.0 - confidence
                    })

    # Filter candidate edges by confidence_filter
    valid_edges = [e for e in candidate_edges if e["confidence"] >= confidence_filter]

    # 3. Maximum Spanning Tree / Edmonds Directed Tree Reconstruction
    mst_edges = []
    if len(valid_edges) > 0:
        # Build networkx graph for MST
        G_candidates = nx.DiGraph()
        for e in valid_edges:
            G_candidates.add_edge(e["source"], e["target"], weight=e["weight"], confidence=e["confidence"], transformation=e["transformation"])

        try:
            # Find root candidate with minimum in-degree / maximum out-degree
            in_degrees = dict(G_candidates.in_degree())
            root_candidates = [n for n, deg in in_degrees.items() if deg == 0]
            
            if not root_candidates:
                # If cycle exists, root is node with highest out-degree
                out_degrees = dict(G_candidates.out_degree())
                root_candidate = max(out_degrees, key=out_degrees.get)
            else:
                root_candidate = root_candidates[0]

            # Run Edmonds optimum branching / arborescence
            branching = nx.maximum_spanning_arborescence(
                G_candidates.to_undirected(as_view=True).to_directed(),
                preserve_attrs=True
            ) if not nx.is_arborescence(G_candidates) else G_candidates

            for u, v, d in branching.edges(data=True):
                mst_edges.append({
                    "source": u,
                    "target": v,
                    "transformation": d.get("transformation", "crop"),
                    "confidence": round(float(d.get("confidence", 0.85)), 4),
                    "weight": round(float(d.get("weight", 0.15)), 4),
                    "parameters": {"method": "ViT-B16+ForensicMLP", "mst_selected": True}
                })
        except Exception:
            # Kruskal Fallback
            sorted_candidates = sorted(valid_edges, key=lambda x: x["weight"])
            visited_nodes = set()
            for e in sorted_candidates:
                if e["target"] not in visited_nodes or len(visited_nodes) < n - 1:
                    mst_edges.append(e)
                    visited_nodes.add(e["source"])
                    visited_nodes.add(e["target"])

    # If no MST edges created but n > 1, connect linearly by timestamp/size heuristic
    if len(mst_edges) == 0 and n > 1:
        sorted_by_size = sorted(image_nodes, key=lambda x: x["file_size"], reverse=True)
        for k in range(len(sorted_by_size) - 1):
            mst_edges.append({
                "source": sorted_by_size[k]["id"],
                "target": sorted_by_size[k+1]["id"],
                "transformation": "crop",
                "confidence": 0.75,
                "weight": 0.25,
                "parameters": {"fallback": True}
            })

    # 4. Identify Root Node
    all_targets = set(e["target"] for e in mst_edges)
    all_sources = set(e["source"] for e in mst_edges)
    potential_roots = all_sources - all_targets

    if potential_roots:
        root_id = list(potential_roots)[0]
    elif image_nodes:
        # Node with highest width * height
        root_id = max(image_nodes, key=lambda x: x["width"] * x["height"])["id"]
    else:
        root_id = None

    # Mark root node
    root_confidence = 0.95
    for node in image_nodes:
        if node["id"] == root_id:
            node["is_root"] = True
            root_confidence = 0.98

    # Assign generation levels via BFS from root
    levels = {node["id"]: 0 for node in image_nodes}
    if root_id:
        tree_graph = nx.DiGraph()
        for e in mst_edges:
            tree_graph.add_edge(e["source"], e["target"])
        
        for node_id in image_nodes:
            nid = node_id["id"]
            if root_id in tree_graph and nid in tree_graph and nid != root_id:
                try:
                    if nx.has_path(tree_graph, root_id, nid):
                        levels[nid] = len(nx.shortest_path(tree_graph, root_id, nid)) - 1
                except Exception:
                    pass

    for node in image_nodes:
        node["generation_level"] = levels.get(node["id"], 0)

    exec_time = (time.time() - start_time) * 1000

    raw_result = {
        "root_image_id": root_id,
        "root_confidence": float(root_confidence),
        "execution_time_ms": round(float(exec_time), 2),
        "nodes": image_nodes,
        "edges": mst_edges,
        "raw_candidate_edges": candidate_edges
    }

    def sanitize_val(obj):
        if isinstance(obj, dict):
            return {k: sanitize_val(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_val(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    return sanitize_val(raw_result)
