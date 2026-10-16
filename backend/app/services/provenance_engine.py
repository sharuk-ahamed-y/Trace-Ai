"""
backend/app/services/provenance_engine.py — Component-Aware Graph Reconstruction & MST Engine

Implements the forensic image provenance pipeline:
1. Cryptographic hashing (SHA-256, MD5) and dimension validation.
2. ViT-B/16 visual embedding & 17-D forensic feature extraction.
3. Pairwise relationship, direction, and transformation inference via ModelManager.
4. Component-aware Minimum Spanning Forest (Kruskal/Prim) over distance weights:
   - weight = 1.0 - P(related) (lower weight = higher confidence)
   - Disconnected components are preserved as distinct trees in a forest.
   - No false edges are invented between disconnected components.
5. Edge orientation performed strictly by the directional classifier, separate from MST.
6. Component root detection via in-degree and directional evidence scoring.
7. BFS generational hierarchy assignment.
"""

import time
import math
import hashlib
from pathlib import Path
from collections import deque
from typing import List, Dict, Any, Tuple
from PIL import Image
import numpy as np
import networkx as nx

from backend.app.services.model_manager import get_model_manager


def compute_hashes(image_path: Path) -> Tuple[str, str]:
    """Computes SHA-256 and MD5 cryptographic hashes."""
    sha256_hash = hashlib.sha256()
    md5_hash = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)
            md5_hash.update(chunk)
    return sha256_hash.hexdigest(), md5_hash.hexdigest()


def run_provenance_analysis(
    image_paths: List[Path],
    similarity_threshold: float = 0.55,
    confidence_filter: float = 0.40,
    adaptive_penalty: bool = True,
    lambda_max: float = 0.05,
    tau_cycle: float = 0.05
) -> Dict[str, Any]:
    """
    Executes end-to-end component-aware provenance graph reconstruction.
    """
    start_time = time.time()
    manager = get_model_manager()

    n = len(image_paths)
    if n < 1:
        raise ValueError("At least 1 image is required for provenance analysis.")

    print(f"[ProvenanceEngine] Uploaded image count: {n}")

    # 1. Extract metadata, real ViT embeddings, and forensic features
    nodes = []
    path_map = {}

    for idx, path in enumerate(image_paths):
        img_id = path.stem.strip()
        sha256, md5 = compute_hashes(path)

        try:
            with Image.open(path) as img:
                w, h = img.size
                img_format = img.format or "UNKNOWN"
        except Exception:
            w, h = 512, 512
            img_format = path.suffix.upper().replace(".", "")

        f_size = path.stat().st_size if path.exists() else 0
        raw_forensics = manager.get_forensic_features(path)
        # Filter out numpy ndarray for clean JSON serialization
        clean_forensics = {
            k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
            for k, v in raw_forensics.items()
            if k != "_feature_vector"
        }
        vit_dim = 768
        try:
            vit_feat = manager.get_vit_features(path)
            if vit_feat is not None:
                vit_dim = int(vit_feat.shape[-1])
        except Exception:
            pass
        print(f"[ProvenanceEngine] Extracted feature dimensions for node '{img_id}': ViT={vit_dim}, Forensics={len(clean_forensics)}")

        # Build URL for frontend preview
        if "uploads" in str(path):
            rel_url = f"/data/uploads/{path.parent.name}/{path.name}"
        elif "images" in str(path):
            rel_url = f"/data/dataset-images/{path.name}"
        else:
            rel_url = f"/data/{path.name}"

        node_data = {
            "id": img_id,
            "filename": path.name,
            "file_path": str(path),
            "url": rel_url,
            "sha256": sha256,
            "md5": md5,
            "width": w,
            "height": h,
            "file_size": f_size,
            "format": img_format,
            "forensics": clean_forensics,
            "is_root": False,
            "generation": 0,
            "component_id": 0
        }
        nodes.append(node_data)
        path_map[img_id] = path

    # Single-image corner case
    if n == 1:
        nodes[0]["is_root"] = True
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "root_image_id": nodes[0]["id"],
            "root_confidence": 1.0,
            "root_reason": "Single image analyzed; designated as solitary root.",
            "execution_time_ms": elapsed_ms,
            "nodes": nodes,
            "edges": [],
            "candidate_edges": [],
            "forest_edges": [],
            "components": [{"component_id": 0, "root": nodes[0]["id"], "node_count": 1, "nodes": [nodes[0]["id"]]}],
            "bfs": [{"level": 0, "images": [nodes[0]["id"]]}],
            "summary": {
                "total_images": 1,
                "candidate_edges": 0,
                "forest_edges": 0,
                "connected_components": 1,
                "max_depth": 0,
                "active_model": manager.get_status()["active_relationship_model"]
            }
        }

    # 2. Pairwise Analysis across all pairs (N*(N-1)/2)
    candidate_edges = []
    rejected_pairs = []
    pairwise_pred_matrix = {}  # (u, v) -> P(u -> v)

    for i in range(n):
        id_a = nodes[i]["id"]
        path_a = path_map[id_a]

        for j in range(i + 1, n):
            id_b = nodes[j]["id"]
            path_b = path_map[id_b]

            # Model prediction
            pred = manager.predict_pair(path_a, path_b)
            p_rel = pred["related_confidence"]
            cos_sim = pred["cosine_similarity"]

            # Edge weight for MST: lower distance weight = stronger connection
            dist_weight = round(1.0 - p_rel, 4)

            area_a = nodes[i].get("forensics", {}).get("area", 0)
            area_b = nodes[j].get("forensics", {}).get("area", 0)

            # Record directed probabilities for tournament / ancestry analysis
            if pred["direction"] == "A_TO_B":
                p_atob = pred["direction_confidence"]
            elif pred["direction"] == "B_TO_A":
                p_atob = 1.0 - pred["direction_confidence"]
            else:
                p_atob = 0.50
            pairwise_pred_matrix[(id_a, id_b)] = p_atob
            pairwise_pred_matrix[(id_b, id_a)] = 1.0 - p_atob

            if pred["direction"] == "A_TO_B":
                src, tgt = id_a, id_b
            elif pred["direction"] == "B_TO_A":
                src, tgt = id_b, id_a
            else:
                # Direction uncertainty fallback: larger uncropped area is natural ancestor
                if area_a >= area_b:
                    src, tgt = id_a, id_b
                else:
                    src, tgt = id_b, id_a

            edge_data = {
                "u": id_a,
                "v": id_b,
                "source": src,
                "target": tgt,
                "related_prob": p_rel,
                "direction": pred["direction"],
                "direction_confidence": pred["direction_confidence"],
                "direction_rationale": pred.get("direction_rationale", "Forensic analysis"),
                "transformation": pred["transformation"],
                "transformation_confidence": pred["transformation_confidence"],
                "compound_tags": pred.get("compound_tags", [pred["transformation"]]),
                "cosine_similarity": cos_sim,
                "weight": dist_weight,
                "confidence": round(p_rel * pred["direction_confidence"], 4)
            }

            # Two-stage candidate filtering:
            # Stage 1: Lineage relatedness >= similarity_threshold
            # Stage 2: Joint edge confidence (P(related) * P(direction)) >= confidence_filter
            min_conf = confidence_filter if confidence_filter is not None else 0.0
            accepted = (p_rel >= similarity_threshold and cos_sim >= 0.35 and edge_data["confidence"] >= min_conf)
            print(
                f"[ProvenanceEngine] Pair ({id_a}, {id_b}): P(rel)={p_rel:.4f}, CosSim={cos_sim:.4f} -> "
                f"Relationship decision={'ACCEPTED' if accepted else 'REJECTED'}; "
                f"Direction={pred['direction']} (P_dir={pred['direction_confidence']:.4f}, predicted={src}->{tgt})"
            )
            if accepted:
                candidate_edges.append(edge_data)
            else:
                rejected_pairs.append(edge_data)

    print(f"[ProvenanceEngine] Candidate-edge count: {len(candidate_edges)}, Rejected-pair count: {len(rejected_pairs)}")

    # 3. Construct Undirected Graph for Spanning Forest
    G_candidate = nx.Graph()
    for node in nodes:
        G_candidate.add_node(node["id"])

    # If adaptive cycle ambiguity penalty is enabled, calculate minimax bottleneck margins
    if adaptive_penalty and candidate_edges:
        import heapq
        G_temp = nx.Graph()
        for e in candidate_edges:
            G_temp.add_edge(e["u"], e["v"], base_cost=e["weight"])

        for e in candidate_edges:
            u, v = e["u"], e["v"]
            c_uv = e["weight"]
            cos_sim = e.get("cosine_similarity", 1.0)

            G_sub = G_temp.copy()
            G_sub.remove_edge(u, v)
            if nx.has_path(G_sub, u, v):
                dist = {node_id: float("inf") for node_id in G_sub.nodes()}
                dist[u] = 0.0
                pq = [(0.0, u)]
                visited = set()
                while pq:
                    curr_b, curr_node = heapq.heappop(pq)
                    if curr_node in visited:
                        continue
                    visited.add(curr_node)
                    if curr_node == v:
                        break
                    for nbr in G_sub.neighbors(curr_node):
                        e_c = G_sub[curr_node][nbr]["base_cost"]
                        new_b = max(curr_b, e_c)
                        if new_b < dist[nbr]:
                            dist[nbr] = new_b
                            heapq.heappush(pq, (new_b, nbr))
                b_val = dist[v]
                delta_cycle = b_val - c_uv
                a_cycle = math.exp(-delta_cycle / tau_cycle) if delta_cycle > 0 else 0.0
            else:
                a_cycle = 0.0

            lam_eff = lambda_max * a_cycle
            mst_cost = round(c_uv + lam_eff * (1.0 - cos_sim), 4)
            e["mst_cost"] = mst_cost
            G_candidate.add_edge(e["u"], e["v"], weight=mst_cost, data=e)
    else:
        for e in candidate_edges:
            e["mst_cost"] = e["weight"]
            G_candidate.add_edge(e["u"], e["v"], weight=e["weight"], data=e)

    # 4. Component-by-Component Minimum Spanning Forest Computation
    # Deterministic component ordering: largest component first, then lexicographically
    connected_subgraphs = sorted(
        [sorted(list(comp)) for comp in nx.connected_components(G_candidate)],
        key=lambda c: (-len(c), c[0] if c else "")
    )
    forest_edges_raw = []

    for comp in connected_subgraphs:
        if len(comp) > 1:
            subgraph = G_candidate.subgraph(comp)
            # Compute MST within this connected component
            sub_mst = nx.minimum_spanning_tree(subgraph, weight="weight")
            for u, v, d in sub_mst.edges(data=True):
                forest_edges_raw.append(d["data"])

    print(f"[ProvenanceEngine] MST-edge count: {len(forest_edges_raw)}")

    # 5. Orient Forest Edges into Directed Provenance DAG
    D_forest = nx.DiGraph()
    for node in nodes:
        D_forest.add_node(node["id"])

    directed_forest_edges = []
    for edge in forest_edges_raw:
        src = edge["source"]
        tgt = edge["target"]
        D_forest.add_edge(src, tgt, **edge)
        directed_forest_edges.append({
            "source": src,
            "target": tgt,
            "transformation": edge["transformation"],
            "confidence": edge["confidence"],
            "related_prob": edge["related_prob"],
            "direction_confidence": edge["direction_confidence"],
            "transformation_confidence": edge["transformation_confidence"],
            "weight": edge["weight"],
            "mst_cost": edge.get("mst_cost", edge["weight"]),
            "cosine_similarity": edge["cosine_similarity"]
        })

    print(f"[ProvenanceEngine] Final directed-edge count: {len(directed_forest_edges)}")

    # Cycle Safeguard: If directional classification created any cycle, reverse lowest confidence edge
    try:
        cycles = list(nx.simple_cycles(D_forest))
        for cycle in cycles:
            # Find cycle edge with minimum confidence
            cycle_edges = list(zip(cycle, cycle[1:] + [cycle[0]]))
            min_e = min(cycle_edges, key=lambda ce: D_forest[ce[0]][ce[1]].get("confidence", 0.0))
            edge_data = D_forest[min_e[0]][min_e[1]]
            D_forest.remove_edge(min_e[0], min_e[1])
            # Reverse edge
            D_forest.add_edge(min_e[1], min_e[0], **edge_data)
            for d_edge in directed_forest_edges:
                if d_edge["source"] == min_e[0] and d_edge["target"] == min_e[1]:
                    d_edge["source"] = min_e[1]
                    d_edge["target"] = min_e[0]
    except Exception:
        pass

    # 6. Component-wise Root Identification & BFS Generational Depth
    component_summaries = []
    global_root_id = None
    global_max_score = -999999.0
    global_root_reason = ""
    nodes_dict = {n["id"]: n for n in nodes}

    for comp_idx, comp_nodes in enumerate(connected_subgraphs):
        # Sub-DiGraph
        comp_digraph = D_forest.subgraph(comp_nodes)

        # Multi-factor Ancestral Invariance Root Scoring:
        # A true original root has:
        # 1. Zero pairwise directional losses (no other image predicted to precede it).
        # 2. Maximum tournament dominance flow against component peers.
        # 3. Maximum directed reachability (sub-tree span) in D_forest.
        # 4. Maximum hierarchical BFS depth in D_forest.
        # 5. In-degree 0 in D_forest.
        comp_scores = {}
        for node_id in comp_nodes:
            in_deg = comp_digraph.in_degree(node_id)
            out_deg = comp_digraph.out_degree(node_id)

            # Forest reachability & path metrics
            descendants = nx.descendants(comp_digraph, node_id)
            reach = len(descendants)
            try:
                paths = nx.single_source_shortest_path_length(comp_digraph, node_id)
                max_depth = max(paths.values()) if paths else 0
            except Exception:
                max_depth = 0

            # Ancestral wins and losses across component
            losses = sum(1 for other in comp_nodes if other != node_id and pairwise_pred_matrix.get((other, node_id), 0.50) > 0.50)
            wins = sum(1 for other in comp_nodes if other != node_id and pairwise_pred_matrix.get((node_id, other), 0.50) > 0.50)
            dom_score = sum(2.0 * pairwise_pred_matrix.get((node_id, other), 0.50) - 1.0 for other in comp_nodes if other != node_id)

            score = (
                (dom_score * 2.0)
                + (reach * 2.0)
                + (max_depth * 1.5)
                + (out_deg * 1.0)
                - (losses * 15.0)
                - (in_deg * 8.0)
            )
            comp_scores[node_id] = {
                "in_deg": in_deg,
                "out_deg": out_deg,
                "reach": reach,
                "max_depth": max_depth,
                "losses": losses,
                "wins": wins,
                "dom_score": dom_score,
                "score": score
            }

        # Candidate selection with hierarchical gating:
        # Prioritize nodes with 0 ancestral losses and 0 forest in-degree
        zero_loss_zero_in = sorted([nid for nid in comp_nodes if comp_scores[nid]["losses"] == 0 and comp_scores[nid]["in_deg"] == 0])
        zero_loss = sorted([nid for nid in comp_nodes if comp_scores[nid]["losses"] == 0])
        zero_in = sorted([nid for nid in comp_nodes if comp_scores[nid]["in_deg"] == 0])

        if zero_loss_zero_in:
            cand_pool = zero_loss_zero_in
        elif zero_loss:
            cand_pool = zero_loss
        elif zero_in:
            cand_pool = zero_in
        else:
            cand_pool = sorted(list(comp_nodes))

        max_score = max(comp_scores[n]["score"] for n in cand_pool)
        tied = [n for n in cand_pool if abs(comp_scores[n]["score"] - max_score) < 1e-6]
        comp_root = sorted(tied)[0]

        # Assign component root status & component ID
        for nid in comp_nodes:
            nodes_dict[nid]["component_id"] = comp_idx

        nodes_dict[comp_root]["is_root"] = True

        # BFS depth within component
        visited = {comp_root}
        queue = deque([(comp_root, 0)])
        comp_levels = {}

        while queue:
            curr, lvl = queue.popleft()
            nodes_dict[curr]["generation"] = lvl
            comp_levels.setdefault(lvl, []).append(curr)

            for nbr in comp_digraph.successors(curr):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, lvl + 1))

        # Handle unvisited nodes in component (e.g. if disconnected internally)
        for unv in set(comp_nodes) - visited:
            nodes_dict[unv]["generation"] = 1

        comp_reason = (
            f"Component #{comp_idx + 1} Root: '{comp_root}' (In-degree={comp_scores[comp_root]['in_deg']}, "
            f"Out-degree={comp_scores[comp_root]['out_deg']}, Ancestral Losses={comp_scores[comp_root]['losses']}, "
            f"Reach={comp_scores[comp_root]['reach']}, span={len(comp_nodes)} nodes)."
        )

        component_summaries.append({
            "component_id": comp_idx,
            "root": comp_root,
            "node_count": len(comp_nodes),
            "nodes": list(comp_nodes),
            "max_depth": max(comp_levels.keys()) if comp_levels else 0,
            "reason": comp_reason
        })

        if comp_scores[comp_root]["score"] > global_max_score or (
            abs(comp_scores[comp_root]["score"] - global_max_score) < 1e-6 and (global_root_id is None or comp_root < global_root_id)
        ):
            global_max_score = comp_scores[comp_root]["score"]
            global_root_id = comp_root
            global_root_reason = comp_reason

    # Default root fallback
    if global_root_id is None and nodes:
        global_root_id = nodes[0]["id"]
        nodes[0]["is_root"] = True

    # Global BFS Structure
    max_global_depth = max(n["generation"] for n in nodes)
    bfs_structure = []
    for d in range(max_global_depth + 1):
        lvl_imgs = [n["id"] for n in nodes if n["generation"] == d]
        if lvl_imgs:
            bfs_structure.append({"level": d, "images": lvl_imgs})

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    outgoing_confs = [e["confidence"] for e in directed_forest_edges if e["source"] == global_root_id]
    computed_root_conf = round(float(np.mean(outgoing_confs)), 4) if outgoing_confs else (1.0 if n == 1 else 0.50)

    print(f"[ProvenanceEngine] Serialized node count: {len(nodes)}, edge count: {len(directed_forest_edges)}")

    return {
        "root_image_id": global_root_id,
        "root_confidence": computed_root_conf,
        "root_reason": global_root_reason,
        "execution_time_ms": elapsed_ms,
        "nodes": nodes,
        "edges": directed_forest_edges,
        "candidate_edges": candidate_edges,
        "rejected_pairs": rejected_pairs,
        "forest_edges": directed_forest_edges,
        "components": component_summaries,
        "bfs": bfs_structure,
        "summary": {
            "total_images": n,
            "candidate_edges": len(candidate_edges),
            "forest_edges": len(directed_forest_edges),
            "rejected_pairs": len(rejected_pairs),
            "connected_components": len(connected_subgraphs),
            "max_depth": max_global_depth,
            "active_model": manager.get_status()["active_relationship_model"],
            "provisional_tree_notice": (
                "Spanning Forest topology is a provisional provenance hypothesis based on "
                "minimal distance weights d = 1 - P(related). Directional orientation is predicted independently."
            )
        }
    }
