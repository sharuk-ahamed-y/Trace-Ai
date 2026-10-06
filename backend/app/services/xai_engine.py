"""
backend/app/services/xai_engine.py — Empirical XAI Visual Heatmap & Evidence Generator
"""

import base64
import io
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from backend.app.config import VISUALIZATION_DIR
from src.xai_explainer import explain_edge_prediction

def generate_noise_residual_heatmap(
    img_a_path: Path,
    img_b_path: Path,
    output_path: Path
) -> Tuple[str, Dict[str, Any]]:
    img_a = cv2.imread(str(img_a_path))
    img_b = cv2.imread(str(img_b_path))

    if img_a is None or img_b is None:
        raise ValueError("Could not read image files for XAI comparison.")

    # Resize img_b to img_a for pixel diff
    h_a, w_a = img_a.shape[:2]
    img_b_resized = cv2.resize(img_b, (w_a, h_a), interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(img_b_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Compute high-pass noise residuals (3x3 median filter diff)
    med_a = cv2.medianBlur(img_a, 3).astype(np.float32)
    med_b = cv2.medianBlur(img_b_resized, 3).astype(np.float32)

    res_a = np.abs(img_a.astype(np.float32) - med_a)
    res_b = np.abs(img_b_resized.astype(np.float32) - med_b)

    # Spatial difference of noise residuals
    diff = np.abs(res_a - res_b)
    diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # Gaussian blur & normalize heatmap
    diff_smooth = cv2.GaussianBlur(diff_gray, (15, 15), 0)
    norm_heatmap = cv2.normalize(diff_smooth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply Jet colormap
    color_heatmap = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)

    # Overlay with source image (0.6 image + 0.4 heatmap)
    overlay = cv2.addWeighted(img_a, 0.6, color_heatmap, 0.4, 0)

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)

    # Also generate base64 string for direct inline rendering
    _, buffer = cv2.imencode('.png', overlay)
    b64_str = base64.b64encode(buffer).decode('utf-8')

    # Metrics
    mse = float(np.mean((gray_a - gray_b) ** 2))
    psnr = float(10 * np.log10((255.0 ** 2) / (mse + 1e-10)))
    ssim_approx = max(0.0, min(1.0, 1.0 - (mse / (255.0 ** 2))))

    return b64_str, {
        "mse": round(mse, 2),
        "psnr_db": round(psnr, 2),
        "ssim_est": round(ssim_approx, 4),
        "residual_diff_max": float(np.max(diff_gray)),
        "residual_diff_mean": round(float(np.mean(diff_gray)), 3)
    }


def generate_xai_report_for_edge(
    edge: Dict[str, Any],
    source_node: Dict[str, Any],
    target_node: Dict[str, Any],
    analysis_id: str
) -> Dict[str, Any]:
    src_path = Path(source_node["file_path"])
    tgt_path = Path(target_node["file_path"])

    heatmap_filename = f"heatmap_{analysis_id}_{source_node['id']}_{target_node['id']}.png"
    heatmap_path = VISUALIZATION_DIR / heatmap_filename

    b64_img, spatial_metrics = generate_noise_residual_heatmap(src_path, tgt_path, heatmap_path)

    exp_str = explain_edge_prediction(
        path_a=src_path,
        path_b=tgt_path,
        predicted_tfm=edge.get("transformation", "crop"),
        tfm_confidence=float(edge.get("confidence", 0.90)),
        dir_confidence=float(edge.get("confidence", 0.95)),
        cos_sim=float(spatial_metrics.get("ssim_est", 0.88)),
        forensic_feats_a=source_node.get("forensics", None),
        forensic_feats_b=target_node.get("forensics", None)
    )

    findings = [line.strip() for line in exp_str.split("\n") if line.strip().startswith("-") or (len(line.strip()) > 2 and line.strip()[0].isdigit() and line.strip()[1] == ".")]

    evidence = {
        "confidence": float(edge.get("confidence", 0.90)),
        "summary": f"Provenential dependency: {source_node['id']} -> {target_node['id']} via {edge.get('transformation', 'crop').upper()}",
        "empirical_findings": findings if findings else [exp_str],
        "full_text": exp_str
    }

    heatmap_url = f"/data/visualizations/{heatmap_filename}"

    return {
        "source_id": source_node["id"],
        "target_id": target_node["id"],
        "heatmap_url": heatmap_url,
        "heatmap_b64": f"data:image/png;base64,{b64_img}",
        "spatial_metrics": spatial_metrics,
        "evidence": evidence
    }
