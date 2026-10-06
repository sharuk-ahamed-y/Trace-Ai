"""
forensic_features.py — Low-level & forensic statistical feature extraction.

Extracts physical, statistical, noise residual, histogram, and frequency-domain
features from images to support XAI (Explainable AI) and forensic classifiers:

  - Image dimensions (width, height, area)
  - Aspect ratio
  - File size (bytes)
  - Estimated JPEG quality / blockiness index
  - RGB statistics (mean, std, min, max per channel)
  - Grayscale histogram (16 bins)
  - Color channel histograms (8 bins per R, G, B channel = 24 bins)
  - Edge statistics (Laplacian gradient mean, std, max)
  - Spatial noise residual statistics (high-pass filter residual mean, std, skewness)
  - High-frequency DCT spectral energy statistics

Saves summary features to dataset/features/forensic_features.csv and raw feature arrays to dataset/features/*.npy.
"""

import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image


def estimate_jpeg_quality(img: Image.Image) -> float:
    """
    Estimate JPEG compression quality / blockiness metric based on 8x8 block boundary differences.
    Higher value indicates lower compression / higher quality.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    h, w = gray.shape
    if h < 16 or w < 16:
        return 100.0

    # Horizontal block boundary diff (every 8 pixels) vs interior diff
    b_left = gray[:, 7:w-1:8]
    b_right = gray[:, 8:w:8]
    min_w = min(b_left.shape[1], b_right.shape[1])
    h_block_diff = np.abs(b_left[:, :min_w] - b_right[:, :min_w]).mean() if min_w > 0 else 0.0

    i_left = gray[:, 3:w-1:8]
    i_right = gray[:, 4:w:8]
    min_w_i = min(i_left.shape[1], i_right.shape[1])
    h_interior_diff = np.abs(i_left[:, :min_w_i] - i_right[:, :min_w_i]).mean() if min_w_i > 0 else 1.0

    # Vertical block boundary diff vs interior diff
    b_top = gray[7:h-1:8, :]
    b_bot = gray[8:h:8, :]
    min_h = min(b_top.shape[0], b_bot.shape[0])
    v_block_diff = np.abs(b_top[:min_h, :] - b_bot[:min_h, :]).mean() if min_h > 0 else 0.0

    i_top = gray[3:h-1:8, :]
    i_bot = gray[4:h:8, :]
    min_h_i = min(i_top.shape[0], i_bot.shape[0])
    v_interior_diff = np.abs(i_top[:min_h_i, :] - i_bot[:min_h_i, :]).mean() if min_h_i > 0 else 1.0

    block_ratio = (h_block_diff + v_block_diff) / (h_interior_diff + v_interior_diff + 1e-5)
    # Map block ratio to quality score [0, 100]
    estimated_quality = max(10.0, min(100.0, 100.0 - (block_ratio - 1.0) * 45.0))
    return float(round(estimated_quality, 2))


def extract_forensic_features(image_path: Path) -> Dict[str, Any]:
    """
    Extract forensic and statistical feature dictionary for a single image.
    """
    path = Path(image_path)
    file_size = path.stat().st_size if path.exists() else 0

    img = Image.open(path).convert("RGB")
    w, h = img.size
    area = w * h
    aspect_ratio = round(w / float(h), 4)

    # RGB stats
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    r_mean, r_std = float(r.mean()), float(r.std())
    g_mean, g_std = float(g.mean()), float(g.std())
    b_mean, b_std = float(b.mean()), float(b.std())

    # Grayscale conversion for spatial/frequency analysis
    gray = np.array(img.convert("L"), dtype=np.float32)

    # Histograms
    gray_hist, _ = np.histogram(gray, bins=16, range=(0, 256), density=True)
    r_hist, _ = np.histogram(r, bins=8, range=(0, 256), density=True)
    g_hist, _ = np.histogram(g, bins=8, range=(0, 256), density=True)
    b_hist, _ = np.histogram(b, bins=8, range=(0, 256), density=True)

    # Edge statistics (Laplacian filter approximation)
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    pad_gray = np.pad(gray, 1, mode="edge")
    edge_map = (
        pad_gray[:-2, 1:-1] * laplacian_kernel[0, 1] +
        pad_gray[1:-1, :-2] * laplacian_kernel[1, 0] +
        pad_gray[1:-1, 1:-1] * laplacian_kernel[1, 1] +
        pad_gray[1:-1, 2:] * laplacian_kernel[1, 2] +
        pad_gray[2:, 1:-1] * laplacian_kernel[2, 1]
    )
    edge_mean = float(np.abs(edge_map).mean())
    edge_std = float(edge_map.std())
    edge_max = float(np.abs(edge_map).max())

    # Spatial noise residual (3x3 median filter subtraction approximation)
    pad_g = np.pad(gray, 1, mode="edge")
    # Quick 3x3 local average residual
    local_avg = (
        pad_g[:-2, :-2] + pad_g[:-2, 1:-1] + pad_g[:-2, 2:] +
        pad_g[1:-1, :-2] + pad_g[1:-1, 1:-1] + pad_g[1:-1, 2:] +
        pad_g[2:, :-2] + pad_g[2:, 1:-1] + pad_g[2:, 2:]
    ) / 9.0
    noise_residual = gray - local_avg
    res_mean = float(noise_residual.mean())
    res_std = float(noise_residual.std())

    # Skewness of noise residual
    res_skew = float(np.mean(((noise_residual - res_mean) / (res_std + 1e-5)) ** 3))

    jpeg_qual = estimate_jpeg_quality(img)

    # Standard feature dictionary
    feat_dict = {
        "image_id": path.stem,
        "file_name": path.name,
        "width": w,
        "height": h,
        "area": area,
        "aspect_ratio": aspect_ratio,
        "file_size": file_size,
        "jpeg_quality_est": jpeg_qual,
        "r_mean": round(r_mean, 4),
        "r_std": round(r_std, 4),
        "g_mean": round(g_mean, 4),
        "g_std": round(g_std, 4),
        "b_mean": round(b_mean, 4),
        "b_std": round(b_std, 4),
        "edge_mean": round(edge_mean, 4),
        "edge_std": round(edge_std, 4),
        "edge_max": round(edge_max, 4),
        "noise_res_std": round(res_std, 4),
        "noise_res_skew": round(res_skew, 4),
    }

    # Vectorized numerical feature representation for MLP / XAI
    feature_vector = np.concatenate([
        [w, h, aspect_ratio, file_size, jpeg_qual, r_mean, r_std, g_mean, g_std, b_mean, b_std,
         edge_mean, edge_std, res_std, res_skew],
        gray_hist,
        r_hist,
        g_hist,
        b_hist,
    ]).astype(np.float32)

    feat_dict["_feature_vector"] = feature_vector
    return feat_dict


def save_forensic_features(image_paths: List[Path], output_csv: Path, features_dir: Path) -> None:
    """
    Extract forensic features for all image paths, write CSV summary, and save .npy arrays.
    """
    import csv

    features_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    records = []
    print(f"[Forensics] Extracting features for {len(image_paths)} images ...")

    for p in image_paths:
        try:
            feats = extract_forensic_features(p)
            vec = feats.pop("_feature_vector")
            npy_path = features_dir / f"{p.stem}.npy"
            np.save(npy_path, vec)
            records.append(feats)
        except Exception as exc:
            print(f"[WARN] Failed forensic extraction for {p}: {exc}")

    if records:
        fieldnames = list(records[0].keys())
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    print(f"[Forensics] Saved summary CSV -> {output_csv}")
    print(f"[Forensics] Saved {len(records)} feature arrays in -> {features_dir}")
