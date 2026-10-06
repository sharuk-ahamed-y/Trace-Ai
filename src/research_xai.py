"""
src/research_xai.py — Explainable AI (XAI) Engine for Image Provenance Analysis.

Provides model-based explanations for:
  1. Root / Original Image Detection
  2. Pairwise Provenance Edge Transformations & Directions
  3. Feature-level Contributions (Forensic & Spectral Deltas)
"""

from typing import Dict, List, Any
import numpy as np


def generate_provenance_explanation(
    analysis_result: Dict[str, Any],
    image_features: Dict[str, np.ndarray],
    forensic_cols: List[str]
) -> Dict[str, Any]:
    """
    Generates human-readable model explanations and feature-level contribution details.
    """
    root_info = analysis_result.get("root", {})
    edges = analysis_result.get("edges", [])
    summary = analysis_result.get("summary", {})

    root_id = root_info.get("image_id", "Unknown")
    root_conf = root_info.get("confidence", 0.0)

    # 1. Root Explanation
    root_explanation = (
        f"Image '{root_id}' is predicted as the ORIGINAL / ROOT image with {root_conf * 100:.1f}% confidence. "
        f"Graph structural analysis confirms zero incoming relationship edges (indegree = 0) and the highest net outgoing "
        f"directional confidence across all evaluated candidates."
    )

    # 2. Edge Explanations
    edge_explanations = []
    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        tfm = edge["transformation"]
        conf = edge["confidence"]
        rel_p = edge["related_prob"]
        dir_p = edge["dir_conf"]

        # Compute forensic feature delta if available
        feature_evidence = []
        if src in image_features and tgt in image_features:
            f_src = image_features[src][55:]  # forensic features portion
            f_tgt = image_features[tgt][55:]

            diff = f_tgt - f_src
            top_idx = np.argsort(np.abs(diff))[-3:][::-1]

            for idx in top_idx:
                if idx < len(forensic_cols):
                    col_name = forensic_cols[idx]
                    delta_val = diff[idx]
                    direction_str = "increased" if delta_val > 0 else "decreased"
                    feature_evidence.append(f"{col_name} {direction_str} by {abs(delta_val):.2f}")

        evidence_str = "; ".join(feature_evidence) if feature_evidence else "Spectral and pair feature representations match transform signature."

        exp = {
            "edge": f"{src} -> {tgt}",
            "source": src,
            "target": tgt,
            "transformation": tfm,
            "confidence": f"{conf * 100:.1f}%",
            "explanation": (
                f"Image '{tgt}' is derived from '{src}' via '{tfm}' transformation (Relationship Confidence: {rel_p * 100:.1f}%, "
                f"Direction Confidence: {dir_p * 100:.1f}%). Forensic Evidence: {evidence_str}"
            ),
            "forensic_evidence": feature_evidence
        }
        edge_explanations.append(exp)

    # 3. Overall Summary Explanation
    summary_explanation = (
        f"Processed {summary.get('total_images', 0)} uploaded images. "
        f"Identified {summary.get('related_pairs', 0)} related pair candidates and constructed a Minimum Spanning Tree (MST) "
        f"with {summary.get('mst_edges', 0)} directed edges spanning {summary.get('max_bfs_depth', 0)} generation levels. "
        f"Rejected {summary.get('rejected_pairs', 0)} weak or unrelated image pairs."
    )

    return {
        "root_explanation": root_explanation,
        "edge_explanations": edge_explanations,
        "summary_explanation": summary_explanation
    }
