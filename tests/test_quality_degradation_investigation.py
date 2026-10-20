"""
tests/test_quality_degradation_investigation.py — Controlled Regression Tests for
Quality Degradation vs. Brightness vs. JPEG Compression.

Audits:
1. Unchanged-image controls (PNG and JPEG identity pairs)
2. Brightness-only transformations (increase and decrease in PNG)
3. JPEG-quality-only transformations (PNG -> JPEG and High-Q JPEG -> Low-Q JPEG)
4. Brightness-plus-JPEG compound pairs (brightened and saved as JPEG)
5. True composite quality-degradation controls (downscale + noise + JPEG)
6. Directional antisymmetry and honest uncertainty across both input orders (A, B) and (B, A).
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import cv2
import io

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.transforms import apply_brightness, apply_quality_degradation
from backend.app.services.model_manager import ModelManager


@pytest.fixture(scope="module")
def manager():
    return ModelManager()


@pytest.fixture(scope="module")
def base_scene(tmp_path_factory):
    """Creates a deterministic synthetic scene with geometric objects, gradients, and text."""
    tmp_dir = tmp_path_factory.mktemp("qd_fixtures")
    base = np.zeros((480, 480, 3), dtype=np.uint8)

    # Gradients
    for y in range(240):
        base[y, :, 0] = int(120 + y * 0.3)
        base[y, :, 1] = int(160 + y * 0.2)
        base[y, :, 2] = 230
    for y in range(240, 480):
        base[y, :, 0] = 50
        base[y, :, 1] = int(120 + (y - 240) * 0.2)
        base[y, :, 2] = 40

    cv2.rectangle(base, (80, 160), (240, 380), (190, 80, 70), -1)
    cv2.rectangle(base, (110, 220), (160, 300), (240, 230, 90), -1)
    cv2.circle(base, (360, 110), 45, (255, 230, 60), -1)
    cv2.putText(base, "TRACE INVESTIGATION 2026", (40, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    p_base = tmp_dir / "base_scene.png"
    Image.fromarray(base).save(p_base)
    return p_base


# -----------------------------------------------------------------------------
# 1. Unchanged-Image Controls
# -----------------------------------------------------------------------------

def test_unchanged_image_control_png(manager, base_scene, tmp_path):
    """Identical PNG copies must be identified as 'identity' with UNCERTAIN direction in both orders."""
    p_copy = tmp_path / "base_copy.png"
    Image.open(base_scene).save(p_copy)

    pred_fwd = manager.predict_pair(base_scene, p_copy)
    assert pred_fwd["related"] is True
    assert pred_fwd["transformation"] == "identity"
    assert pred_fwd["direction"] == "UNCERTAIN"
    assert pred_fwd["direction_confidence"] == 0.50

    pred_rev = manager.predict_pair(p_copy, base_scene)
    assert pred_rev["related"] is True
    assert pred_rev["transformation"] == "identity"
    assert pred_rev["direction"] == "UNCERTAIN"
    assert pred_rev["direction_confidence"] == 0.50


def test_unchanged_image_control_jpeg(manager, base_scene, tmp_path):
    """Identical JPEG copies must be identified as 'identity' and NOT as 'jpeg' or 'quality_degradation'."""
    p_jpg1 = tmp_path / "scene_copy1.jpg"
    p_jpg2 = tmp_path / "scene_copy2.jpg"
    im = Image.open(base_scene)
    im.save(p_jpg1, format="JPEG", quality=85)
    im.save(p_jpg2, format="JPEG", quality=85)

    pred_fwd = manager.predict_pair(p_jpg1, p_jpg2)
    assert pred_fwd["related"] is True
    assert pred_fwd["transformation"] == "identity"
    assert pred_fwd["direction"] == "UNCERTAIN"
    assert pred_fwd["direction_confidence"] == 0.50

    pred_rev = manager.predict_pair(p_jpg2, p_jpg1)
    assert pred_rev["related"] is True
    assert pred_rev["transformation"] == "identity"
    assert pred_rev["direction"] == "UNCERTAIN"
    assert pred_rev["direction_confidence"] == 0.50


# -----------------------------------------------------------------------------
# 2. Brightness-Only Cases
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("factor", [1.30, 0.70])
def test_brightness_only(manager, base_scene, tmp_path, factor):
    """Pure brightness change in PNG format must be classified as 'brightness' with UNCERTAIN direction."""
    im = Image.open(base_scene)
    im_bright = apply_brightness(im, {"factor": factor})
    p_bright = tmp_path / f"bright_{factor}.png"
    im_bright.save(p_bright)

    pred_fwd = manager.predict_pair(base_scene, p_bright)
    assert pred_fwd["related"] is True
    assert pred_fwd["transformation"] == "brightness"
    assert pred_fwd["direction"] == "UNCERTAIN"
    assert pred_fwd["direction_confidence"] == 0.50

    pred_rev = manager.predict_pair(p_bright, base_scene)
    assert pred_rev["related"] is True
    assert pred_rev["transformation"] == "brightness"
    assert pred_rev["direction"] == "UNCERTAIN"
    assert pred_rev["direction_confidence"] == 0.50


# -----------------------------------------------------------------------------
# 3. JPEG-Quality-Only Cases
# -----------------------------------------------------------------------------

def test_jpeg_quality_only_png_to_jpeg(manager, base_scene, tmp_path):
    """PNG original saved to JPEG must be classified as 'jpeg' with A_TO_B direction."""
    p_jpg = tmp_path / "scene_q45.jpg"
    Image.open(base_scene).save(p_jpg, format="JPEG", quality=45)

    pred_fwd = manager.predict_pair(base_scene, p_jpg)
    assert pred_fwd["related"] is True
    assert pred_fwd["transformation"] == "jpeg"
    assert pred_fwd["direction"] == "A_TO_B"
    assert pred_fwd["direction_confidence"] >= 0.90

    pred_rev = manager.predict_pair(p_jpg, base_scene)
    assert pred_rev["related"] is True
    assert pred_rev["transformation"] == "jpeg"
    assert pred_rev["direction"] == "B_TO_A"
    assert pred_rev["direction_confidence"] >= 0.90


def test_jpeg_quality_only_high_to_low(manager, base_scene, tmp_path):
    """High-Q JPEG (Q=92) vs Low-Q JPEG (Q=40) must be classified as 'jpeg' with High -> Low direction."""
    p_high = tmp_path / "scene_q92.jpg"
    p_low = tmp_path / "scene_q40.jpg"
    im = Image.open(base_scene)
    im.save(p_high, format="JPEG", quality=92)
    im.save(p_low, format="JPEG", quality=40)

    pred_fwd = manager.predict_pair(p_high, p_low)
    assert pred_fwd["related"] is True
    assert pred_fwd["transformation"] == "jpeg"
    assert pred_fwd["direction"] == "A_TO_B"
    assert pred_fwd["direction_confidence"] >= 0.90

    pred_rev = manager.predict_pair(p_low, p_high)
    assert pred_rev["related"] is True
    assert pred_rev["transformation"] == "jpeg"
    assert pred_rev["direction"] == "B_TO_A"
    assert pred_rev["direction_confidence"] >= 0.90


# -----------------------------------------------------------------------------
# 4. Brightness-Plus-JPEG Cases (The Reported Bug)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("factor", [1.25, 0.75])
def test_brightness_plus_jpeg_not_misclassified_as_quality_degradation(manager, base_scene, tmp_path, factor):
    """Brightness-adjusted image saved as JPEG must NOT be misclassified as 'quality_degradation'."""
    im = Image.open(base_scene)
    im_bright = apply_brightness(im, {"factor": factor})
    p_bright_jpg = tmp_path / f"bright_{factor}.jpg"
    im_bright.save(p_bright_jpg, format="JPEG", quality=85)

    # Case A: PNG original vs Brightened JPEG
    pred_fwd = manager.predict_pair(base_scene, p_bright_jpg)
    assert pred_fwd["related"] is True
    assert pred_fwd["transformation"] != "quality_degradation", "Must not falsely trigger quality_degradation"
    assert pred_fwd["transformation"] in ["brightness", "jpeg"]
    assert "brightness" in pred_fwd["compound_tags"]

    pred_rev = manager.predict_pair(p_bright_jpg, base_scene)
    assert pred_rev["related"] is True
    assert pred_rev["transformation"] != "quality_degradation"
    assert "brightness" in pred_rev["compound_tags"]

    # Case B: JPEG original (Q=85) vs Brightened JPEG (Q=85) (Like WhatsApp transfers)
    p_orig_jpg = tmp_path / "orig_q85.jpg"
    im.save(p_orig_jpg, format="JPEG", quality=85)

    pred_jpg_fwd = manager.predict_pair(p_orig_jpg, p_bright_jpg)
    assert pred_jpg_fwd["related"] is True
    assert pred_jpg_fwd["transformation"] == "brightness", "Brightness between equal-Q JPEGs must be 'brightness'"
    assert pred_jpg_fwd["direction"] == "UNCERTAIN"
    assert pred_jpg_fwd["direction_confidence"] == 0.50

    pred_jpg_rev = manager.predict_pair(p_bright_jpg, p_orig_jpg)
    assert pred_jpg_rev["related"] is True
    assert pred_jpg_rev["transformation"] == "brightness"
    assert pred_jpg_rev["direction"] == "UNCERTAIN"
    assert pred_jpg_rev["direction_confidence"] == 0.50


# -----------------------------------------------------------------------------
# 5. True Quality Degradation Control
# -----------------------------------------------------------------------------

def test_true_quality_degradation_composite(manager, base_scene, tmp_path):
    """True multi-stage quality degradation (downscale + jpeg) must be identified with quality_degradation."""
    im = Image.open(base_scene)
    w, h = im.size
    im_deg = im.resize((int(w * 0.50), int(h * 0.50)), Image.Resampling.LANCZOS)
    p_deg = tmp_path / "true_degraded.jpg"
    im_deg.save(p_deg, format="JPEG", quality=40)

    pred_fwd = manager.predict_pair(base_scene, p_deg)
    assert pred_fwd["related"] is True
    assert "quality_degradation" in pred_fwd["compound_tags"] or pred_fwd["transformation"] == "quality_degradation"
    assert pred_fwd["direction"] == "A_TO_B"

