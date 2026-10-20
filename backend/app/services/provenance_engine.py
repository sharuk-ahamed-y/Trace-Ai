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
from src.forensic_arbitrator import clear_arbitration_caches


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
    clear_arbitration_caches()
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
    sim_matrix = {}  # (u, v) -> cos_sim

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
            sim_matrix[(id_a, id_b)] = cos_sim
            sim_matrix[(id_b, id_a)] = cos_sim

            # Check for verified physical transformation evidence (crop, resize, identity, blur)
            tfm_name = pred.get("transformation")
            raw_tfm_conf = pred.get("transformation_confidence")
            has_verified_physical_evidence = (
                tfm_name in ["crop", "resize", "identity", "blur"]
                and raw_tfm_conf is not None
                and float(raw_tfm_conf) >= 0.88
            )

            # Effective lineage relatedness incorporates verified physical evidence
            effective_p_rel = max(p_rel, float(raw_tfm_conf)) if has_verified_physical_evidence else p_rel

            # Edge weight for MST: lower distance weight = stronger connection
            dist_weight = round(1.0 - effective_p_rel, 4)

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

            is_dir_uncertain = pred["direction"] in ["UNCERTAIN", "NONE"]
            if pred["direction"] == "A_TO_B":
                src, tgt = id_a, id_b
            elif pred["direction"] == "B_TO_A":
                src, tgt = id_b, id_a
            else:
                # Remove canvas area as authoritative original/derived signal!
                # When direction is uncertain, preserve canonical ordering and mark explicitly as unresolved
                src, tgt = (id_a, id_b) if id_a < id_b else (id_b, id_a)

            # Joint confidence: For verified direction, joint = P(related) * P(direction).
            # For unresolved direction, edge confidence reflects verified pairwise relatedness P(related),
            # while direction is explicitly marked UNCERTAIN and is_uncertain = True.
            edge_conf = round(effective_p_rel * pred["direction_confidence"], 4) if not is_dir_uncertain else round(effective_p_rel, 4)

            edge_data = {
                "u": id_a,
                "v": id_b,
                "source": src,
                "target": tgt,
                "source_id": src,
                "target_id": tgt,
                "related_prob": effective_p_rel,
                "direction": pred["direction"],
                "direction_confidence": pred["direction_confidence"],
                "is_uncertain": is_dir_uncertain,
                "direction_rationale": pred.get("direction_rationale", "Forensic analysis"),
                "transformation": pred["transformation"],
                "transformation_confidence": pred["transformation_confidence"],
                "compound_tags": pred.get("compound_tags", [pred["transformation"]]),
                "cosine_similarity": cos_sim,
                "weight": dist_weight,
                "confidence": edge_conf
            }

            # Two-stage candidate filtering:
            # Stage 1: Lineage relatedness >= similarity_threshold (or verified physical evidence >= 0.88)
            # Stage 2: Joint edge confidence (P(related) * P(direction)) >= confidence_filter
            min_conf = confidence_filter if confidence_filter is not None else 0.0
            lineage_rel_ok = (p_rel >= similarity_threshold) or has_verified_physical_evidence
            accepted = (pred.get("related", False) and lineage_rel_ok and cos_sim >= 0.35 and edge_data["confidence"] >= min_conf)
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

    # 4. Component Identification and Connected Subgraphs
    # Deterministic component ordering: largest component first, then lexicographically
    connected_subgraphs = sorted(
        [sorted(list(comp)) for comp in nx.connected_components(G_candidate)],
        key=lambda c: (-len(c), c[0] if c else "")
    )

    directed_forest_edges = []
    component_summaries = []
    global_root_id = None
    global_max_score = -999999.0
    global_root_reason = ""
    nodes_dict = {n["id"]: n for n in nodes}

    # 5. Component-wise Directed Arborescence & Root Identification
    for comp_idx, comp_nodes in enumerate(connected_subgraphs):
        # Candidate edges within this connected component
        comp_cand_edges = [
            e for e in candidate_edges
            if e["u"] in comp_nodes and e["v"] in comp_nodes
        ]

        # Extract verified directional signals among candidate edges
        verified_edges = []
        for e in comp_cand_edges:
            if not e.get("is_uncertain", False) and e.get("direction") in ["A_TO_B", "B_TO_A"]:
                verified_edges.append((e["source"], e["target"], e))

        comp_has_verified_edges = len(verified_edges) > 0

        # Build candidate verified DAG on component nodes
        verified_dag = nx.DiGraph()
        verified_dag.add_nodes_from(comp_nodes)
        for parent, child, _ in verified_edges:
            verified_dag.add_edge(parent, child)

        # Multi-factor Ancestral Invariance Root Scoring across candidate edges:
        # 1. Zero pairwise directional losses (no other image predicted or verified to precede it).
        # 2. Maximum tournament dominance flow against component peers.
        # 3. In-degree 0 in verified directional relationships.
        # 4. High medoid cosine similarity across component peers.
        # 5. Directed reachability in verified_dag.
        comp_scores = {}
        for node_id in comp_nodes:
            in_deg = verified_dag.in_degree(node_id)
            out_deg = verified_dag.out_degree(node_id)

            reach = len(nx.descendants(verified_dag, node_id))
            try:
                paths = nx.single_source_shortest_path_length(verified_dag, node_id)
                max_depth = max(paths.values()) if paths else 0
            except Exception:
                max_depth = 0

            # Ancestral wins and losses across component
            losses = sum(1 for other in comp_nodes if other != node_id and (
                pairwise_pred_matrix.get((other, node_id), 0.50) > 0.50 or (other, node_id) in verified_dag.edges()
            ))
            wins = sum(1 for other in comp_nodes if other != node_id and (
                pairwise_pred_matrix.get((node_id, other), 0.50) > 0.50 or (node_id, other) in verified_dag.edges()
            ))
            dom_score = sum(2.0 * pairwise_pred_matrix.get((node_id, other), 0.50) - 1.0 for other in comp_nodes if other != node_id)

            sims = [sim_matrix.get((node_id, other), 0.0) for other in comp_nodes if other != node_id]
            avg_sim = float(np.mean(sims)) if sims else 0.0

            score = (
                (dom_score * 2.0)
                + (reach * 3.0)
                + (max_depth * 2.0)
                + (out_deg * 2.0)
                - (losses * 15.0)
                - (in_deg * 12.0)
                + (avg_sim * 2.0)
            )
            comp_scores[node_id] = {
                "in_deg": in_deg,
                "out_deg": out_deg,
                "reach": reach,
                "max_depth": max_depth,
                "losses": losses,
                "wins": wins,
                "dom_score": dom_score,
                "avg_sim": avg_sim,
                "score": score
            }

        # Candidate selection with hierarchical gating:
        zero_loss_zero_in = sorted([nid for nid in comp_nodes if comp_scores[nid]["losses"] == 0 and comp_scores[nid]["in_deg"] == 0])
        zero_in = sorted([nid for nid in comp_nodes if comp_scores[nid]["in_deg"] == 0])
        zero_loss = sorted([nid for nid in comp_nodes if comp_scores[nid]["losses"] == 0])

        if zero_loss_zero_in:
            cand_pool = zero_loss_zero_in
        elif zero_in:
            cand_pool = zero_in
        elif zero_loss:
            cand_pool = zero_loss
        else:
            cand_pool = sorted(list(comp_nodes))

        # Select component root with highest score, tie-breaking on medoid centrality (avg_sim), then ID
        max_score = max(comp_scores[n]["score"] for n in cand_pool)
        tied = [n for n in cand_pool if abs(comp_scores[n]["score"] - max_score) < 1e-6]
        if len(tied) > 1:
            tied.sort(key=lambda nid: (-comp_scores[nid]["avg_sim"], nid))
        comp_root = tied[0]

        # Assign component root status & component ID
        for nid in comp_nodes:
            nodes_dict[nid]["component_id"] = comp_idx

        nodes_dict[comp_root]["is_root"] = True
        nodes_dict[comp_root]["generation"] = 0

        # Build Directed Provenance Tree (Arborescence) from comp_root
        comp_tree_edges = []
        comp_levels = {0: [comp_root]}
        paths = {comp_root: [comp_root]}

        if len(comp_nodes) > 1:
            # Build physical containment DAG exclusively for verified geometric crop/resize derivations
            # Transitive reduction on containment DAG resolves sequential nested crops (A -> B -> C)
            # without false shortcut attribution across non-geometric transformations (e.g. blur, brightness).
            containment_dag = nx.DiGraph()
            containment_dag.add_nodes_from(comp_nodes)
            for p, c, e_data in verified_edges:
                if e_data.get("transformation") in ["crop", "resize"]:
                    containment_dag.add_edge(p, c)

            shortcut_edges = set()
            if nx.is_directed_acyclic_graph(containment_dag) and len(containment_dag.edges()) > 1:
                try:
                    dag_tr = nx.transitive_reduction(containment_dag)
                    shortcut_edges = set(containment_dag.edges()) - set(dag_tr.edges())
                except Exception:
                    shortcut_edges = set()

            # Directed cost graph: traversing forward along verified edges has priority;
            # traversing backward through verified edges (e.g. crop -> source) is heavily penalized.
            G_dir = nx.DiGraph()
            G_dir.add_nodes_from(comp_nodes)

            for e in comp_cand_edges:
                u, v = e["u"], e["v"]
                cos_sim = e.get("cosine_similarity", 0.50)
                base_w = e.get("mst_cost", e.get("weight", 1.0 - cos_sim))
                is_ver = not e.get("is_uncertain", False) and e.get("direction") in ["A_TO_B", "B_TO_A"]

                if is_ver:
                    vp, vc = e["source"], e["target"]
                    is_sc = (vp, vc) in shortcut_edges
                    e["is_shortcut"] = is_sc
                    # Forward verified edge has priority; shortcut bypasses are deprioritized in favor of intermediate parents
                    edge_w = (50.0 + base_w) if is_sc else (base_w * 0.5)
                    G_dir.add_edge(vp, vc, weight=edge_w, data=e)
                    # Backward verified edge is blocked (e.g. crop cannot precede uncropped source)
                    G_dir.add_edge(vc, vp, weight=1000.0 + base_w, data=e)
                else:
                    e["is_shortcut"] = False
                    # Symmetric or uncertain edge can be traversed either way
                    G_dir.add_edge(u, v, weight=base_w, data=e)
                    G_dir.add_edge(v, u, weight=base_w, data=e)

            # Dijkstra shortest path tree from comp_root
            try:
                dist, paths = nx.single_source_dijkstra(G_dir, comp_root, weight="weight")
            except Exception:
                dist, paths = {comp_root: 0.0}, {comp_root: [comp_root]}

            # Fallback for any unreachable nodes using candidate subgraph
            for tgt in comp_nodes:
                if tgt not in paths:
                    try:
                        u_path = nx.shortest_path(G_candidate.subgraph(comp_nodes), comp_root, tgt, weight="weight")
                        paths[tgt] = u_path
                    except Exception:
                        pass

            for tgt in comp_nodes:
                if tgt == comp_root:
                    continue
                if tgt in paths and len(paths[tgt]) >= 2:
                    parent = paths[tgt][-2]
                    lvl = len(paths[tgt]) - 1
                    nodes_dict[tgt]["generation"] = lvl
                    comp_levels.setdefault(lvl, []).append(tgt)

                    # Locate corresponding candidate edge
                    c_edge = None
                    for ce in comp_cand_edges:
                        if (ce["u"] == parent and ce["v"] == tgt) or (ce["u"] == tgt and ce["v"] == parent):
                            c_edge = ce
                            break

                    if c_edge is not None:
                        is_unc = c_edge.get("is_uncertain", False) or c_edge.get("direction") in ["UNCERTAIN", "NONE"]
                        ver_forward = not is_unc and (c_edge["source"] == parent and c_edge["target"] == tgt)
                        dir_label = "A_TO_B" if ver_forward else ("UNCERTAIN" if is_unc else "A_TO_B")

                        edge_entry = {
                            "source": parent,
                            "target": tgt,
                            "source_id": parent,
                            "target_id": tgt,
                            "transformation": c_edge["transformation"],
                            "confidence": c_edge["confidence"],
                            "related_prob": c_edge["related_prob"],
                            "direction_confidence": c_edge["direction_confidence"],
                            "transformation_confidence": c_edge["transformation_confidence"],
                            "weight": c_edge["weight"],
                            "mst_cost": c_edge.get("mst_cost", c_edge["weight"]),
                            "cosine_similarity": c_edge["cosine_similarity"],
                            "direction": dir_label,
                            "is_uncertain": is_unc,
                            "pairwise_direction": c_edge.get("direction", "UNCERTAIN"),
                            "pairwise_direction_confidence": c_edge.get("direction_confidence", 0.50),
                            "tree_lineage": f"{parent} -> {tgt}",
                            "is_tree_edge": True,
                            "is_shortcut": c_edge.get("is_shortcut", False),
                            "direction_rationale": c_edge.get("direction_rationale", "")
                        }
                        comp_tree_edges.append(edge_entry)

        directed_forest_edges.extend(comp_tree_edges)

        # Handle unvisited nodes in component
        for unv in set(comp_nodes) - set(paths.keys()):
            nodes_dict[unv]["generation"] = 1

        if not comp_has_verified_edges and len(comp_nodes) > 1:
            comp_reason = (
                f"Component #{comp_idx + 1}: Direction unresolved across {len(comp_nodes)} images. "
                "No authoritative directional evidence to designate an original root."
            )
        else:
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

    has_any_verified_edges = any(
        not e.get("is_uncertain", False) and e.get("direction") in ["A_TO_B", "B_TO_A"]
        for e in directed_forest_edges
    )
    outgoing_confs = [
        e["confidence"] for e in directed_forest_edges
        if e["source"] == global_root_id and not e.get("is_uncertain", False)
    ]
    computed_root_conf = round(float(np.mean(outgoing_confs)), 4) if outgoing_confs else (1.0 if n == 1 else 0.50)
    if not has_any_verified_edges and n > 1:
        global_root_reason = "Direction unresolved: No authoritative forensic or directional signal to designate an original root."

    print(f"[ProvenanceEngine] Serialized node count: {len(nodes)}, edge count: {len(directed_forest_edges)}")

    return {
        "root_image_id": global_root_id,
        "root_confidence": computed_root_conf,
        "root_reason": global_root_reason,
        "root_verified": has_any_verified_edges or (n == 1),
        "direction_status": "verified" if (has_any_verified_edges or n == 1) else "unresolved",
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
