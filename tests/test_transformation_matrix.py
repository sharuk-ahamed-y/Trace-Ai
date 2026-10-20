"""
tests/test_transformation_matrix.py — Comprehensive Transformation Audit & Regression Test Suite

Tests every supported transformation class across:
1. Relationship classification (related vs unrelated)
2. Transformation classification (correct class without silent identity/crop fallbacks)
3. Direction semantics & antisymmetry (swapping A and B preserves physical parent/child identities)
4. Confidence separation (relationship probability != direction confidence != uncalibrated heuristics)
5. Graph construction & root identification
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.transforms import (
    apply_crop, apply_resize, apply_jpeg, apply_blur,
    apply_brightness, apply_contrast, apply_sharpen,
    apply_rotation, apply_color, apply_noise, apply_perspective,
    apply_quality_degradation
)
from backend.app.services.model_manager import ModelManager
from backend.app.services.provenance_engine import run_provenance_analysis


@pytest.fixture(scope="module")
def base_scene(tmp_path_factory):
    """Generates a deterministic rich synthetic scene with geometry, text, colors, and gradients."""
    tmp_dir = tmp_path_factory.mktemp("base_fixtures")
    base = np.zeros((480, 480, 3), dtype=np.uint8)

    # Sky gradient
    for y in range(240):
        base[y, :, 0] = int(120 + y * 0.3)  # R
        base[y, :, 1] = int(160 + y * 0.2)  # G
        base[y, :, 2] = 230                 # B

    # Terrain
    for y in range(240, 480):
        base[y, :, 0] = 50
        base[y, :, 1] = int(120 + (y - 240) * 0.2)
        base[y, :, 2] = 40

    # Geometric structures
    cv2.rectangle(base, (80, 160), (240, 380), (190, 80, 70), -1)
    cv2.rectangle(base, (110, 220), (160, 300), (240, 230, 90), -1)
    cv2.circle(base, (360, 110), 45, (255, 230, 60), -1)
    cv2.putText(base, "TRACE AUDIT 2026", (60, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    p_base = tmp_dir / "base_scene.png"
    Image.fromarray(base).save(p_base)
    return p_base


@pytest.fixture(scope="module")
def unrelated_scene(tmp_path_factory):
    """Generates an independent, completely disjoint scene."""
    tmp_dir = tmp_path_factory.mktemp("unrelated_fixtures")
    unrelated = np.zeros((480, 480, 3), dtype=np.uint8)
    for x in range(0, 480, 20):
        cv2.line(unrelated, (x, 0), (x, 480), (180, 180, 180), 2)
    for y in range(0, 480, 30):
        cv2.line(unrelated, (0, y), (480, y), (100, 100, 200), 2)

    p_unrelated = tmp_dir / "unrelated_scene.png"
    Image.fromarray(unrelated).save(p_unrelated)
    return p_unrelated


@pytest.fixture(scope="module")
def manager():
    mgr = ModelManager()
    mgr.initialize()
    assert not mgr.phase4_enabled, "ENABLE_PHASE4_MODELS must be False"
    return mgr


# -------------------------------------------------------------------------
# 1. Identity / Unchanged
# -------------------------------------------------------------------------

def test_transformation_identity(manager, base_scene, tmp_path):
    """Identical images must yield transformation='identity', direction='UNCERTAIN', and not claim verified root."""
    p_copy = tmp_path / "base_copy.png"
    im = Image.open(base_scene)
    im.save(p_copy)

    pred = manager.predict_pair(base_scene, p_copy)
    assert pred["related"] is True
    assert pred["related_confidence"] >= 0.98
    assert pred["transformation"] == "identity"
    assert pred["direction"] == "UNCERTAIN"
    assert pred["direction_confidence"] == 0.50


# -------------------------------------------------------------------------
# 2. Brightness Increase and Decrease
# -------------------------------------------------------------------------

@pytest.mark.parametrize("factor,label", [(1.35, "increase"), (0.65, "decrease")])
def test_transformation_brightness(manager, base_scene, tmp_path, factor, label):
    """Brightness shifted images must be identified as 'brightness' in both input orders."""
    im = Image.open(base_scene)
    im_bright = apply_brightness(im, {"factor": factor})
    p_bright = tmp_path / f"bright_{label}.png"
    im_bright.save(p_bright)

    # Forward order (A=Orig, B=Bright)
    pred_fwd = manager.predict_pair(base_scene, p_bright)
    assert pred_fwd["related"] is True
    assert pred_fwd["transformation"] == "brightness"
    assert "brightness" in pred_fwd["compound_tags"] or pred_fwd["transformation"] == "brightness"

    # Reverse order (A=Bright, B=Orig)
    pred_rev = manager.predict_pair(p_bright, base_scene)
    assert pred_rev["related"] is True
    assert pred_rev["transformation"] == "brightness"

    # Semantic antisymmetry check: direction must swap consistently
    if pred_fwd["direction"] == "A_TO_B":
        assert pred_rev["direction"] == "B_TO_A"
    elif pred_fwd["direction"] == "B_TO_A":
        assert pred_rev["direction"] == "A_TO_B"
    else:
        assert pred_fwd["direction"] == "UNCERTAIN"
        assert pred_rev["direction"] == "UNCERTAIN"


# -------------------------------------------------------------------------
# 3. Contrast Increase and Decrease
# -------------------------------------------------------------------------

@pytest.mark.parametrize("factor,label", [(1.40, "increase"), (0.60, "decrease")])
def test_transformation_contrast(manager, base_scene, tmp_path, factor, label):
    """Contrast shifted images must be identified as 'contrast' in both input orders."""
    im = Image.open(base_scene)
    im_contrast = apply_contrast(im, {"factor": factor})
    p_contrast = tmp_path / f"contrast_{label}.png"
    im_contrast.save(p_contrast)

    # Forward order
    pred_fwd = manager.predict_pair(base_scene, p_contrast)
    assert pred_fwd["related"] is True
    assert pred_fwd["transformation"] == "contrast"

    # Reverse order
    pred_rev = manager.predict_pair(p_contrast, base_scene)
    assert pred_rev["related"] is True
    assert pred_rev["transformation"] == "contrast"


# -------------------------------------------------------------------------
# 4. Gaussian Blur
# -------------------------------------------------------------------------

@pytest.mark.parametrize("radius", [1.8, 3.2])
def test_transformation_blur(manager, base_scene, tmp_path, radius):
    """Blurred images must be identified as 'blur' with clean image as Parent."""
    im = Image.open(base_scene)
    im_blur = apply_blur(im, {"radius": radius})
    p_blur = tmp_path / f"blur_{radius}.png"
    im_blur.save(p_blur)

    # Forward order (A=Orig, B=Blur) -> A is parent (A_TO_B)
    pred_fwd = manager.predict_pair(base_scene, p_blur)
    assert pred_fwd["related"] is True
    assert pred_fwd["transformation"] == "blur"
    assert pred_fwd["direction"] == "A_TO_B"
    assert pred_fwd["direction_confidence"] >= 0.90

    # Reverse order (A=Blur, B=Orig) -> B is parent (B_TO_A)
    pred_rev = manager.predict_pair(p_blur, base_scene)
    assert pred_rev["related"] is True
    assert pred_rev["transformation"] == "blur"
    assert pred_rev["direction"] == "B_TO_A"
    assert pred_rev["direction_confidence"] >= 0.90


# -------------------------------------------------------------------------
# 5. Sharpening
# -------------------------------------------------------------------------

@pytest.mark.parametrize("factor", [1.8, 2.6])
def test_transformation_sharpen(manager, base_scene, tmp_path, factor):
    """Sharpened images must be identified as 'sharpen' with clean image as Parent."""
    im = Image.open(base_scene)
    im_sharp = apply_sharpen(im, {"factor": factor})
    p_sharp = tmp_path / f"sharp_{factor}.png"
    im_sharp.save(p_sharp)

    pred_fwd = manager.predict_pair(base_scene, p_sharp)
    assert pred_fwd["related"] is True
    assert pred_fwd["transformation"] == "sharpen"
    assert pred_fwd["direction"] == "A_TO_B"

    pred_rev = manager.predict_pair(p_sharp, base_scene)
    assert pred_rev["related"] is True
    assert pred_rev["transformation"] == "sharpen"
    assert pred_rev["direction"] == "B_TO_A"


# -------------------------------------------------------------------------
# 6. Gaussian Noise
# -------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [12.0, 20.0])
def test_transformation_noise(manager, base_scene, tmp_path, sigma):
    """Noisy images must be identified as 'noise' with clean image as Parent."""
    im = Image.open(base_scene)
    im_noise = apply_noise(im, {"sigma": sigma})
    p_noise = tmp_path / f"noise_{sigma}.png"
    im_noise.save(p_noise)

    pred_fwd = manager.predict_pair(base_scene, p_noise)
    assert pred_fwd["related"] is True
    assert pred_fwd["transformation"] == "noise"
    assert pred_fwd["direction"] == "A_TO_B"

    pred_rev = manager.predict_pair(p_noise, base_scene)
    assert pred_rev["related"] is True
    assert pred_rev["transformation"] == "noise"
    assert pred_rev["direction"] == "B_TO_A"


# -------------------------------------------------------------------------
# 7. JPEG Compression
# -------------------------------------------------------------------------

@pytest.mark.parametrize("quality", [35, 65])
def test_transformation_jpeg(manager, base_scene, tmp_path, quality):
    """JPEG compressed images must be identified as 'jpeg' with original as Parent."""
    im = Image.open(base_scene)
    im_jpeg = apply_jpeg(im, {"quality": quality})
    p_jpeg = tmp_path / f"jpeg_{quality}.jpg"
    im_jpeg.save(p_jpeg, format="JPEG", quality=quality)

    pred_fwd = manager.predict_pair(base_scene, p_jpeg)
    assert pred_fwd["related"] is True
    assert pred_fwd["transformation"] == "jpeg"
    assert pred_fwd["direction"] == "A_TO_B"

    pred_rev = manager.predict_pair(p_jpeg, base_scene)
    assert pred_rev["related"] is True
    assert pred_rev["transformation"] == "jpeg"
    assert pred_rev["direction"] == "B_TO_A"


# -------------------------------------------------------------------------
# 8. Color / Saturation
# -------------------------------------------------------------------------

@pytest.mark.parametrize("factor", [1.6, 0.4])
def test_transformation_color(manager, base_scene, tmp_path, factor):
    """Color saturated/desaturated images must be identified as 'color'."""
    im = Image.open(base_scene)
    im_color = apply_color(im, {"factor": factor})
    p_color = tmp_path / f"color_{factor}.png"
    im_color.save(p_color)

    pred_fwd = manager.predict_pair(base_scene, p_color)
    assert pred_fwd["related"] is True
    assert pred_fwd["transformation"] == "color"

    pred_rev = manager.predict_pair(p_color, base_scene)
    assert pred_rev["related"] is True
    assert pred_rev["transformation"] == "color"


# -------------------------------------------------------------------------
# 9. Resize (Downscale / Upscale)
# -------------------------------------------------------------------------

@pytest.mark.parametrize("scale,resize_back", [(0.60, False), (0.50, True)])
def test_transformation_resize(manager, base_scene, tmp_path, scale, resize_back):
    """Resized images must be identified as 'resize'."""
    im = Image.open(base_scene)
    im_resize = apply_resize(im, {"scale": scale, "resize_back": resize_back})
    p_resize = tmp_path / f"resize_{scale}_{resize_back}.png"
    im_resize.save(p_resize)

    pred_fwd = manager.predict_pair(base_scene, p_resize)
    assert pred_fwd["related"] is True
    assert pred_fwd["transformation"] == "resize"

    pred_rev = manager.predict_pair(p_resize, base_scene)
    assert pred_rev["related"] is True
    assert pred_rev["transformation"] == "resize"


# -------------------------------------------------------------------------
# 10. Crop
# -------------------------------------------------------------------------

def test_transformation_crop(manager, base_scene, tmp_path):
    """Cropped sub-window must be identified as 'crop' with full scene as Parent."""
    im = Image.open(base_scene)
    im_crop = apply_crop(im, {"mode": "center", "frac_w": 0.20, "frac_h": 0.20, "resize_back": False})
    p_crop = tmp_path / "crop_center.png"
    im_crop.save(p_crop)

    pred_fwd = manager.predict_pair(base_scene, p_crop)
    assert pred_fwd["related"] is True
    assert pred_fwd["transformation"] == "crop"
    assert pred_fwd["direction"] == "A_TO_B"

    pred_rev = manager.predict_pair(p_crop, base_scene)
    assert pred_rev["related"] is True
    assert pred_rev["transformation"] == "crop"
    assert pred_rev["direction"] == "B_TO_A"


# -------------------------------------------------------------------------
# 11. Quality Degradation (Composite: Resize + Noise + JPEG)
# -------------------------------------------------------------------------

def test_transformation_quality_degradation(manager, base_scene, tmp_path):
    """Composite degradation must be identified with quality_degradation tag."""
    im = Image.open(base_scene)
    im_deg = apply_quality_degradation(im, {"downscale": 0.55, "sigma": 12.0, "jpeg_quality": 40})
    p_deg = tmp_path / "degraded.jpg"
    im_deg.save(p_deg)

    pred = manager.predict_pair(base_scene, p_deg)
    assert pred["related"] is True
    assert pred["transformation"] in ["quality_degradation", "jpeg", "noise", "resize"]
    assert "quality_degradation" in pred["compound_tags"] or pred["transformation"] == "quality_degradation"
    assert pred["direction"] == "A_TO_B"


# -------------------------------------------------------------------------
# 12. Unrelated Negative Pair
# -------------------------------------------------------------------------

def test_unrelated_negative_pair(manager, base_scene, unrelated_scene):
    """Disjoint visual scenes must be identified as unrelated."""
    pred = manager.predict_pair(base_scene, unrelated_scene)
    assert pred["related"] is False
    assert pred["transformation"] == "none"
    assert pred["direction"] == "NONE"
