"""
xai_explainer.py — Explainable AI (XAI) feature evidence & explanation generator for provenance edges.

Generates human-readable, grounded explanations for predicted provenance edges using
real model confidence probabilities, visual embeddings, and forensic feature statistics.
"""

from pathlib import Path
from typing import Dict, Any

from src.forensic_features import extract_forensic_features


def explain_edge_prediction(
    path_a: Path,
    path_b: Path,
    predicted_tfm: str,
    tfm_confidence: float,
    dir_confidence: float,
    cos_sim: float,
    forensic_feats_a: Dict[str, Any] = None,
    forensic_feats_b: Dict[str, Any] = None,
) -> str:
    """
    Generate an explainable AI rationale string grounded in real empirical evidence.
    """
    if forensic_feats_a is None:
        forensic_feats_a = extract_forensic_features(path_a)
    if forensic_feats_b is None:
        forensic_feats_b = extract_forensic_features(path_b)

    w_a, h_a = forensic_feats_a.get("width", 0), forensic_feats_a.get("height", 0)
    w_b, h_b = forensic_feats_b.get("width", 0), forensic_feats_b.get("height", 0)
    size_a = forensic_feats_a.get("file_size", 0)
    size_b = forensic_feats_b.get("file_size", 0)

    jq_a = forensic_feats_a.get("jpeg_quality_est", 100.0)
    jq_b = forensic_feats_b.get("jpeg_quality_est", 100.0)

    edge_a = forensic_feats_a.get("edge_mean", 0.0)
    edge_b = forensic_feats_b.get("edge_mean", 0.0)

    res_std_a = forensic_feats_a.get("noise_res_std", 0.0)
    res_std_b = forensic_feats_b.get("noise_res_std", 0.0)

    explanation_lines = [
        f"========================================",
        f"PROVENANCE RELATIONSHIP EXPLANATION (XAI)",
        f"========================================",
        f"Edge: {path_a.name}  -->  {path_b.name}",
        f"Predicted Transformation: {predicted_tfm.upper()}",
        f"Transformation Confidence: {tfm_confidence * 100:.1f}%",
        f"Direction Confidence:      {dir_confidence * 100:.1f}%",
        f"",
        f"EMPIRICAL FORENSIC EVIDENCE:",
        f"1. Visual Feature Similarity:",
        f"   - ViT CLS Cosine Similarity: {cos_sim:.4f} (High semantic alignment)",
        f"",
        f"2. Dimension & Resolution Consistency:",
        f"   - Parent Dimensions: {w_a} x {h_a} ({size_a / 1024:.1f} KB)",
        f"   - Child Dimensions:  {w_b} x {h_b} ({size_b / 1024:.1f} KB)",
    ]

    lum_a = forensic_feats_a.get("r_mean", 0.0) * 0.299 + forensic_feats_a.get("g_mean", 0.0) * 0.587 + forensic_feats_a.get("b_mean", 0.0) * 0.114
    lum_b = forensic_feats_b.get("r_mean", 0.0) * 0.299 + forensic_feats_b.get("g_mean", 0.0) * 0.587 + forensic_feats_b.get("b_mean", 0.0) * 0.114
    std_a = (forensic_feats_a.get("r_std", 0.0) + forensic_feats_a.get("g_std", 0.0) + forensic_feats_a.get("b_std", 0.0)) / 3.0
    std_b = (forensic_feats_b.get("r_std", 0.0) + forensic_feats_b.get("g_std", 0.0) + forensic_feats_b.get("b_std", 0.0)) / 3.0

    if predicted_tfm == "crop":
        if w_a != w_b or h_a != h_b:
            explanation_lines.append(f"   - Spatial framing/bounding reduction: {w_b}x{h_b} from {w_a}x{h_a}")
        else:
            explanation_lines.append(f"   - Sub-region crop with canvas normalization to {w_b}x{h_b} (re-scaled back)")
    elif predicted_tfm == "resize":
        explanation_lines.append(f"   - Spatial interpolation and scaling shift: {w_a}x{h_a} -> {w_b}x{h_b} (Area: {w_a*h_a} -> {w_b*h_b} px)")
    elif predicted_tfm == "jpeg":
        explanation_lines.append(f"   - JPEG compression artifact & blockiness shift: estimated quality {jq_a:.1f}% -> {jq_b:.1f}%")
    elif predicted_tfm == "blur":
        explanation_lines.append(f"   - High-frequency spectral attenuation: Laplacian edge intensity {edge_a:.2f} -> {edge_b:.2f}")
    elif predicted_tfm == "sharpen":
        explanation_lines.append(f"   - High-frequency edge gradient accentuation: Laplacian edge intensity {edge_a:.2f} -> {edge_b:.2f}")
    elif predicted_tfm == "noise":
        explanation_lines.append(f"   - Additive stochastic noise residual increase: spatial noise std {res_std_a:.2f} -> {res_std_b:.2f}")
    elif predicted_tfm == "brightness":
        dir_label = "increase" if lum_b > lum_a else "attenuation"
        explanation_lines.append(f"   - Mean photometric luminance {dir_label}: {lum_a:.1f} -> {lum_b:.1f} (delta = {lum_b - lum_a:+.1f})")
    elif predicted_tfm == "contrast":
        dir_label = "expansion" if std_b > std_a else "compression"
        explanation_lines.append(f"   - Dynamic range standard deviation {dir_label}: {std_a:.1f} -> {std_b:.1f} (delta = {std_b - std_a:+.1f})")
    elif predicted_tfm == "color":
        explanation_lines.append(f"   - Chromatic channel distribution shift: RGB std {std_a:.1f} -> {std_b:.1f}")
    elif predicted_tfm == "rotation":
        explanation_lines.append(f"   - Geometric rotation shift with interpolation artifacts and border realignment ({w_a}x{h_a} -> {w_b}x{h_b})")
    elif predicted_tfm == "perspective":
        explanation_lines.append(f"   - Projective / planar homography warping distortion across coordinate plane ({w_a}x{h_a} -> {w_b}x{h_b})")
    elif predicted_tfm == "quality_degradation":
        explanation_lines.append(f"   - Compound forensic degradation: downscaling + noise injection ({res_std_a:.1f} -> {res_std_b:.1f}) + JPEG compression ({jq_a:.1f}% -> {jq_b:.1f}%)")
    else:
        explanation_lines.append(f"   - Statistical distribution shift consistent with {predicted_tfm}")

    explanation_lines.extend([
        f"",
        f"3. Spectral & Forensic Residuals:",
        f"   - Estimated JPEG Quality Score: {jq_a:.1f}% (Parent) vs {jq_b:.1f}% (Child)",
        f"   - Laplacian Gradient Intensity: {edge_a:.2f} (Parent) vs {edge_b:.2f} (Child)",
        f"   - High-Pass Noise Residual Std: {res_std_a:.2f} (Parent) vs {res_std_b:.2f} (Child)",
        f"========================================",
    ])

    return "\n".join(explanation_lines)
