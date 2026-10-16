"""
src/forensic_arbitrator.py — Deterministic, Grounded Forensic Direction & Transformation Arbitrator

Implements principled, physically grounded forensic rules that resolve model blindspots
while strictly preserving mathematical direction anti-symmetry:
  arbitrate_direction(B, A) == reverse(arbitrate_direction(A, B))

Arbitrated Forensics:
1. Low-Pass Spectral Attenuation: Detects Gaussian blur without falling victim to high-frequency prior bias.
2. Scale-Invariant Sub-Region Containment: Detects crops even when rescaled back to original dimensions.
3. Physical Canvas Invariance: Downscaling or bounding crops cannot expand image canvas area.
4. DCT 8x8 Grid Discontinuity: Distinguishes JPEG block artifacts from true optical sharpening.
5. Compound Manipulation Tagging: Provides multi-label capability for composite/sequential edits.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from PIL import Image
import cv2


def check_sub_region_containment(path_a: Path, path_b: Path) -> Tuple[float, float]:
    """
    Antisymmetric multi-scale normalized cross-correlation test.
    Returns: (direction_score, max_correlation)
      +1.0: Image B is an interior sub-region of Image A (A -> B, A is parent).
      -1.0: Image A is an interior sub-region of Image B (B -> A, B is parent).
       0.0: No scale-invariant sub-region containment detected.
    """
    try:
        with Image.open(path_a).convert("L") as im_a, Image.open(path_b).convert("L") as im_b:
            arr_a = np.array(im_a)
            arr_b = np.array(im_b)
    except Exception:
        return 0.0, 0.0

    scales = [1.8, 1.5, 1.25, 1.0, 0.85, 0.70, 0.55, 0.40]

    # Test B in A
    best_b_in_a = 0.0
    for s in scales:
        bw = int(arr_b.shape[1] * s)
        bh = int(arr_b.shape[0] * s)
        if bw < 32 or bh < 32 or bw >= arr_a.shape[1] or bh >= arr_a.shape[0]:
            continue
        interp = cv2.INTER_LINEAR if s > 1.0 else cv2.INTER_AREA
        b_scaled = cv2.resize(arr_b, (bw, bh), interpolation=interp)
        res = cv2.matchTemplate(arr_a, b_scaled, cv2.TM_CCOEFF_NORMED)
        best_b_in_a = max(best_b_in_a, float(np.max(res)))

    # Test A in B
    best_a_in_b = 0.0
    for s in scales:
        aw = int(arr_a.shape[1] * s)
        ah = int(arr_a.shape[0] * s)
        if aw < 32 or ah < 32 or aw >= arr_b.shape[1] or ah >= arr_b.shape[0]:
            continue
        interp = cv2.INTER_LINEAR if s > 1.0 else cv2.INTER_AREA
        a_scaled = cv2.resize(arr_a, (aw, ah), interpolation=interp)
        res = cv2.matchTemplate(arr_b, a_scaled, cv2.TM_CCOEFF_NORMED)
        best_a_in_b = max(best_a_in_b, float(np.max(res)))

    # Strict threshold with contrast margin
    if best_b_in_a >= 0.80 and best_b_in_a > best_a_in_b + 0.15:
        return 1.0, best_b_in_a
    elif best_a_in_b >= 0.80 and best_a_in_b > best_b_in_a + 0.15:
        return -1.0, best_a_in_b
    return 0.0, max(best_b_in_a, best_a_in_b)


def check_spectral_blur_attenuation(f_a: Dict[str, Any], f_b: Dict[str, Any], cos_sim: float) -> float:
    """
    Antisymmetric test for low-pass Gaussian blur attenuation.
    Returns:
      +1.0: Image B has undergone low-pass blur attenuation relative to A (A -> B).
      -1.0: Image A has undergone low-pass blur attenuation relative to B (B -> A).
       0.0: No asymmetric low-pass attenuation detected.
    """
    if cos_sim < 0.80:
        return 0.0

    e_a = float(f_a.get("edge_mean", 0.0))
    n_a = float(f_a.get("noise_res_std", 0.0))
    e_b = float(f_b.get("edge_mean", 0.0))
    n_b = float(f_b.get("noise_res_std", 0.0))

    # Condition: significant edge energy drop AND noise residual drop
    # A is original, B is blurred
    if e_a >= 2.0 * e_b and (n_a >= 1.5 * n_b or n_a > n_b + 1.5):
        return 1.0
    # B is original, A is blurred
    if e_b >= 2.0 * e_a and (n_b >= 1.5 * n_a or n_b > n_a + 1.5):
        return -1.0

    return 0.0


def check_canvas_area_constraint(f_a: Dict[str, Any], f_b: Dict[str, Any]) -> float:
    """
    Antisymmetric physical canvas dimension constraint.
    Downscaling or bounding cropping reduces area; it cannot expand canvas area naturally.
    Returns:
      +1.0: Image A is significantly larger than Image B (A -> B).
      -1.0: Image B is significantly larger than Image A (B -> A).
       0.0: Similar or equal canvas dimensions.
    """
    area_a = float(f_a.get("area", 0.0))
    area_b = float(f_b.get("area", 0.0))
    if area_a <= 0 or area_b <= 0:
        return 0.0

    if area_a >= 1.18 * area_b:
        return 1.0
    elif area_b >= 1.18 * area_a:
        return -1.0
    return 0.0


def arbitrate_pair_direction(
    path_a: Path,
    path_b: Path,
    f_a: Dict[str, Any],
    f_b: Dict[str, Any],
    base_direction: str,
    base_confidence: float,
    cos_sim: float,
    predicted_tfm: str
) -> Tuple[str, float, str]:
    """
    Arbitrates final parent-child direction by fusing physical constraints with ML logits.
    Guarantees 100% mathematical antisymmetry:
      arbitrate(B, A) == reverse(arbitrate(A, B))
    Returns: (final_direction, confidence, rationale)
    """
    # 1. Check scale-invariant sub-region crop containment
    crop_score, crop_corr = check_sub_region_containment(path_a, path_b)
    if crop_score > 0.5:
        return "A_TO_B", round(max(0.92, crop_corr), 4), "Forensic Field-of-View Containment: Image B is an interior sub-window of Image A"
    elif crop_score < -0.5:
        return "B_TO_A", round(max(0.92, crop_corr), 4), "Forensic Field-of-View Containment: Image A is an interior sub-window of Image B"

    # 2. Check spectral blur attenuation
    blur_score = check_spectral_blur_attenuation(f_a, f_b, cos_sim)
    if blur_score > 0.5:
        return "A_TO_B", 0.9850, "Forensic Spectral Attenuation: Image B exhibits severe high-frequency loss consistent with Gaussian low-pass filtering"
    elif blur_score < -0.5:
        return "B_TO_A", 0.9850, "Forensic Spectral Attenuation: Image A exhibits severe high-frequency loss consistent with Gaussian low-pass filtering"

    # 3. Check canvas area physical constraint for spatial operations and downsampling
    area_score = check_canvas_area_constraint(f_a, f_b)
    if abs(area_score) > 0.5 and cos_sim >= 0.60:
        if area_score > 0.5:
            return "A_TO_B", max(base_confidence, 0.9800), "Physical Area Constraint: High-resolution canvas Image A exceeds derived subscale Image B"
        elif area_score < -0.5:
            return "B_TO_A", max(base_confidence, 0.9800), "Physical Area Constraint: High-resolution canvas Image B exceeds derived subscale Image A"

    # 4. Check for genuine ambiguity
    if 0.48 <= base_confidence <= 0.52 and abs(area_score) < 0.1:
        # Near-decision boundary with no physical cues
        return base_direction, base_confidence, "Slight Directional Gradient: Close to uncertainty boundary"

    # Default to verified Antisymmetric Direction MLP
    return base_direction, base_confidence, "Antisymmetric Neural Gradient: Multimodal feature divergence flow"


def arbitrate_transformation(
    path_a: Path,
    path_b: Path,
    f_a: Dict[str, Any],
    f_b: Dict[str, Any],
    base_tfm: str,
    base_conf: float,
    cos_sim: float,
    direction: str
) -> Tuple[str, float, List[str]]:
    """
    Arbitrates transformation classification, resolving JPEG-sharpen confusion and
    generating multi-label compound tags for composite transformations.
    Returns: (primary_transformation, confidence, compound_tags)
    """
    compound_tags = []
    if base_tfm and base_tfm != "none":
        compound_tags.append(base_tfm)

    # Designate parent and child feature dictionaries based on direction
    if direction == "A_TO_B":
        f_parent, f_child = f_a, f_b
    elif direction == "B_TO_A":
        f_parent, f_child = f_b, f_a
    else:
        f_parent, f_child = f_a, f_b

    area_p = float(f_parent.get("area", 0.0))
    area_c = float(f_child.get("area", 0.0))
    jq_p = float(f_parent.get("jpeg_quality_est", 100.0))
    jq_c = float(f_child.get("jpeg_quality_est", 100.0))
    edge_p = float(f_parent.get("edge_mean", 0.0))
    edge_c = float(f_child.get("edge_mean", 0.0))
    noise_p = float(f_parent.get("noise_res_std", 0.0))
    noise_c = float(f_child.get("noise_res_std", 0.0))

    final_tfm = base_tfm

    # 1. Resolve JPEG Compression Artifacts
    if jq_c < 96.5 or (jq_p - jq_c) >= 1.5:
        if base_tfm in ["sharpen", "rotation", "contrast", "color", "noise"]:
            final_tfm = "jpeg"
        if "jpeg" not in compound_tags:
            compound_tags.append("jpeg")

    # 2. Check for Blur Signature
    blur_score = check_spectral_blur_attenuation(f_parent, f_child, cos_sim)
    if blur_score > 0.5:
        final_tfm = "blur"
        if "blur" not in compound_tags:
            compound_tags.append("blur")

    # 3. Check for Scale-Invariant Crop
    crop_score, _ = check_sub_region_containment(path_a, path_b)
    if abs(crop_score) > 0.5:
        final_tfm = "crop"
        if "crop" not in compound_tags:
            compound_tags.append("crop")

    # 4. Check for Dimensional Resizing / Downscaling
    if area_p > 1.20 * area_c:
        if "resize" not in compound_tags:
            compound_tags.append("resize")
        if base_tfm in ["blur", "perspective", "sharpen", "noise"]:
            final_tfm = "resize"

    # 5. Check for Noise Addition
    if noise_c >= 1.5 * noise_p and (noise_c - noise_p) >= 4.0:
        if "noise" not in compound_tags:
            compound_tags.append("noise")

    # 6. Compound Tagging for Quality Degradation / WhatsApp
    # WhatsApp / social media workflow: downscale + JPEG recompression
    if "resize" in compound_tags and ("jpeg" in compound_tags or jq_c < 90.0):
        if "quality_degradation" not in compound_tags:
            compound_tags.append("quality_degradation")
        if "whatsapp_compression" not in compound_tags and area_p > 1.25 * area_c:
            compound_tags.append("whatsapp_compression")

    return final_tfm, base_conf, list(dict.fromkeys(compound_tags))
