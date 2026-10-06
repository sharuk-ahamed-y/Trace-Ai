"""
graph_builder.py — Build the undirected MST provenance graph and orient its edges.

Steps
-----
1. Compute pairwise cosine distances from ViT embeddings.
2. Build a complete weighted graph.
3. Run NetworkX minimum_spanning_tree.
4. For every MST edge:
   a. Predict direction (parent_to_child vs child_to_parent).
   b. Predict transformation type + confidence.
   c. Orient the edge accordingly.
5. Cycle safeguard: if orienting an edge creates a directed cycle,
   reverse the lower-confidence edge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pair_features import build_pair_feature
from src.config import MST_WEIGHT_ATTR, CONFIDENCE_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# Graph building
# ─────────────────────────────────────────────────────────────────────────────

def build_mst(
    image_paths: List[Path],
    embeddings:  Dict[str, torch.Tensor],
) -> nx.Graph:
    """
    Build an undirected MST from cosine distances of ViT embeddings.

    Parameters
    ----------
    image_paths : list of image Paths (nodes)
    embeddings  : path-str → (768,) tensor

    Returns
    -------
    mst : nx.Graph with edge attribute 'weight' = cosine distance
    """
    G = nx.Graph()
    paths = [str(p) for p in image_paths]
    G.add_nodes_from(paths)

    n = len(paths)
    for i in range(n):
        for j in range(i + 1, n):
            a = embeddings[paths[i]].float()
            b = embeddings[paths[j]].float()
            cos_sim  = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            cos_dist = 1.0 - cos_sim     # in [0, 2]; 0 = identical
            G.add_edge(paths[i], paths[j], weight=cos_dist)

    mst = nx.minimum_spanning_tree(G, weight=MST_WEIGHT_ATTR)
    return mst


# ─────────────────────────────────────────────────────────────────────────────
# Direction + transformation prediction for a single edge
# ─────────────────────────────────────────────────────────────────────────────

def _predict_edge(
    path_a: str,
    path_b: str,
    embeddings: Dict[str, torch.Tensor],
    dir_model:   nn.Module,
    dir_names:   List[str],
    tfm_model:   nn.Module,
    tfm_names:   List[str],
    device: torch.device,
) -> dict:
    """
    Predict direction and transformation for one undirected edge (A, B).
    Returns result dict with source/target/transformation/confidences.
    """
    emb_a = embeddings[path_a].float()
    emb_b = embeddings[path_b].float()

    feat_ab = build_pair_feature(emb_a, emb_b).unsqueeze(0).to(device)
    feat_ba = build_pair_feature(emb_b, emb_a).unsqueeze(0).to(device)

    with torch.no_grad():
        # Direction: use A→B feature; label 0 = parent_to_child
        dir_logits_ab = dir_model(feat_ab)                 # (1, 2)
        dir_probs_ab  = F.softmax(dir_logits_ab, dim=1).squeeze(0)  # (2,)

        # Probability that A is parent (i.e. A→B is parent_to_child)
        prob_ab_is_parent = dir_probs_ab[0].item()         # class 0
        dir_conf = max(prob_ab_is_parent, 1.0 - prob_ab_is_parent)

        if prob_ab_is_parent >= 0.5:
            source, target = path_a, path_b
            feat_for_tfm   = feat_ab
        else:
            source, target = path_b, path_a
            feat_for_tfm   = feat_ba

        # Transformation
        tfm_logits = tfm_model(feat_for_tfm)
        tfm_probs  = F.softmax(tfm_logits, dim=1).squeeze(0)
        tfm_idx    = tfm_probs.argmax().item()
        tfm_conf   = tfm_probs[tfm_idx].item()
        tfm_name   = tfm_names[tfm_idx]

    return {
        "source":                  source,
        "target":                  target,
        "predicted_transformation": tfm_name,
        "transformation_confidence": round(tfm_conf, 4),
        "direction_confidence":      round(dir_conf, 4),
        "dir_prob_ab_is_parent":     round(prob_ab_is_parent, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cycle safeguard
# ─────────────────────────────────────────────────────────────────────────────

def _has_cycle(edges: List[Tuple[str, str, dict]]) -> bool:
    """Check if the directed edges contain a cycle."""
    DG = nx.DiGraph()
    for s, t, _ in edges:
        DG.add_edge(s, t)
    return not nx.is_directed_acyclic_graph(DG)


def _resolve_cycle(
    edges: List[Tuple[str, str, dict]],
) -> List[Tuple[str, str, dict]]:
    """
    Iteratively reverse the lowest-confidence edge until no cycle remains.
    """
    resolved = list(edges)
    max_iter = len(edges) * 2
    for _ in range(max_iter):
        if not _has_cycle(resolved):
            break
        # Find edge in a cycle with lowest direction confidence
        DG = nx.DiGraph()
        for s, t, d in resolved:
            DG.add_edge(s, t, **d)
        try:
            cycle_nodes = nx.find_cycle(DG)
        except nx.exception.NetworkXNoCycle:
            break

        cycle_edge_pairs = [(u, v) for u, v, _ in cycle_nodes]
        # Find lowest-confidence edge among cycle members
        worst_idx  = None
        worst_conf = 1.1
        for idx, (s, t, d) in enumerate(resolved):
            if (s, t) in cycle_edge_pairs and d["direction_confidence"] < worst_conf:
                worst_conf = d["direction_confidence"]
                worst_idx  = idx

        if worst_idx is not None:
            s, t, d = resolved[worst_idx]
            resolved[worst_idx] = (t, s, d)   # reverse the edge

    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def orient_mst(
    mst: nx.Graph,
    embeddings:  Dict[str, torch.Tensor],
    dir_model:   nn.Module,
    dir_names:   List[str],
    tfm_model:   nn.Module,
    tfm_names:   List[str],
    device: torch.device,
) -> Tuple[nx.DiGraph, List[dict]]:
    """
    Orient every MST edge and return the directed provenance graph + edge records.

    Returns
    -------
    DG      : nx.DiGraph with directed, labelled edges
    records : list of dicts (one per edge) for CSV/JSON export
    """
    raw_edges: List[Tuple[str, str, dict]] = []

    for path_a, path_b in mst.edges():
        result = _predict_edge(
            path_a, path_b, embeddings,
            dir_model, dir_names,
            tfm_model, tfm_names,
            device,
        )
        raw_edges.append((result["source"], result["target"], result))

    # Resolve cycles
    resolved = _resolve_cycle(raw_edges)

    # Build directed graph
    DG = nx.DiGraph()
    records: List[dict] = []

    for source, target, data in resolved:
        DG.add_edge(source, target,
                    transformation=data["predicted_transformation"],
                    tfm_conf=data["transformation_confidence"],
                    dir_conf=data["direction_confidence"])
        records.append({
            "source":                    source,
            "target":                    target,
            "predicted_transformation":  data["predicted_transformation"],
            "transformation_confidence": data["transformation_confidence"],
            "direction_confidence":      data["direction_confidence"],
        })

    return DG, records
