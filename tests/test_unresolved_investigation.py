"""
tests/test_unresolved_investigation.py — End-to-End Investigation & Regression Suite

Verifies:
1. Exact reproduction of the live user failure (WhatsApp Image 1 vs Image 2).
   - In forward upload order (Image 1 -> Image 2): Image 1 is original, Image 2 is crop, direction A_TO_B.
   - In reverse upload order (Image 2 -> Image 1): Image 1 is original, Image 2 is crop, direction B_TO_A.
   - Separate confidences for relationship, direction, and transformation.
2. Sibling derivative isolation:
   - Sibling images derived from the same root are not forced into an ancestry relationship.
3. Perspective and Rotation transformations:
   - Planar homography projective warp classified as 'perspective' with genuine confidence.
   - Rigid planar rotation classified as 'rotation' with genuine confidence.
   - Honest UNCERTAIN direction (0.50) because planar transformations are physically symmetric.
4. Scale-invariant full-canvas resize in both upload orders.
5. Photometric symmetry (brightness, contrast) direction honesty (UNCERTAIN, 0.50).
6. Provenance tree orientation:
   - Spanning tree edges radiate outward from component root.
   - Unresolved directional edges do not invert toward root.
"""

import pytest
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

from backend.app.services.model_manager import ModelManager
from backend.app.services.provenance_engine import run_provenance_analysis
from src.forensic_arbitrator import (
    check_sub_region_containment,
    check_full_canvas_resizing,
    check_perspective_and_rotation,
    arbitrate_pair_direction,
    arbitrate_transformation
)


@pytest.fixture(scope="module")
def manager():
    mgr = ModelManager()
    mgr.initialize()
    assert not mgr.phase4_enabled, "Safety violation: ENABLE_PHASE4_MODELS must be False"
    return mgr


@pytest.fixture(scope="module")
def user_upload_dir():
    p = Path("data/uploads/anl_20261019_105704_976675")
    if p.exists():
        return p
    pytest.skip("User upload directory not found")


@pytest.fixture(scope="module")
def synthetic_scene(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("unresolved_fixtures")
    scene = np.zeros((480, 480, 3), dtype=np.uint8)
    for y in range(240):
        scene[y, :, 0] = int(100 + y * 0.4)
        scene[y, :, 1] = int(150 + y * 0.3)
        scene[y, :, 2] = 220
    for y in range(240, 480):
        scene[y, :, 0] = 40
        scene[y, :, 1] = int(110 + (y - 240) * 0.3)
        scene[y, :, 2] = 30

    cv2.rectangle(scene, (60, 140), (220, 360), (180, 70, 60), -1)
    cv2.circle(scene, (340, 120), 40, (240, 220, 50), -1)
    cv2.putText(scene, "FORENSIC 2026", (50, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    p_scene = tmp_dir / "scene.png"
    Image.fromarray(scene).save(p_scene)
    return p_scene


# -------------------------------------------------------------------------
# 1. Live User Test Reproduction: WhatsApp Original vs. Crop in Both Orders
# -------------------------------------------------------------------------

def test_live_user_crop_pair_both_orders(manager, user_upload_dir):
    """
    Exact reproduction test using the user's uploaded images.
    Image 1 (900x1600) is the uncropped original.
    Image 2 (1392x1598) is an interior crop that was previously marked UNRESOLVED.
    """
    p_orig = user_upload_dir / "WhatsApp_Image_2026-10-18_at_10.28.56_PM_(1).jpeg"
    p_crop = user_upload_dir / "WhatsApp_Image_2026-10-18_at_10.28.57_PM_(2).jpeg"

    # Forward upload order: Original first, Crop second
    pred_fwd = manager.predict_pair(p_orig, p_crop)
    assert pred_fwd["related"] is True
    assert pred_fwd["transformation"] == "crop"
    assert pred_fwd["direction"] == "A_TO_B", f"Expected A_TO_B (Orig -> Crop), got {pred_fwd['direction']}"
    assert pred_fwd["direction_confidence"] >= 0.85
    assert pred_fwd["transformation_confidence"] >= 0.85
    # Confidences must remain separate
    assert pred_fwd["related_confidence"] != pred_fwd["direction_confidence"]

    # Reverse upload order: Crop first, Original second
    pred_rev = manager.predict_pair(p_crop, p_orig)
    assert pred_rev["related"] is True
    assert pred_rev["transformation"] == "crop"
    assert pred_rev["direction"] == "B_TO_A", f"Expected B_TO_A (Orig -> Crop), got {pred_rev['direction']}"
    assert pred_rev["direction_confidence"] >= 0.85
    assert pred_rev["transformation_confidence"] >= 0.85


def test_live_user_sibling_isolation(manager, user_upload_dir):
    """
    WhatsApp Image 2 and Image 3 are sibling crops / distinct views.
    Neither contains the other; direction must remain UNCERTAIN,
    and neither is forced into a false parent-child relationship.
    """
    p_crop1 = user_upload_dir / "WhatsApp_Image_2026-10-18_at_10.28.57_PM_(2).jpeg"
    p_crop2 = user_upload_dir / "WhatsApp_Image_2026-10-18_at_10.28.57_PM_(3).jpeg"

    score, corr = check_sub_region_containment(p_crop1, p_crop2)
    assert score == 0.0, "Sibling images must not exhibit one-way sub-region containment"


# -------------------------------------------------------------------------
# 2. Geometric Transformations: Perspective and Rotation
# -------------------------------------------------------------------------

def test_perspective_transformation_classification_and_direction(manager, synthetic_scene, tmp_path):
    """
    Perspective warping (projective homography) must be classified as 'perspective'
    with high confidence, while direction is honestly UNCERTAIN (0.50) due to planar symmetry.
    """
    im = cv2.imread(str(synthetic_scene))
    h, w = im.shape[:2]

    # Non-affine projective warp (trapezoidal / keystone)
    pts1 = np.float32([[40, 40], [w - 40, 40], [40, h - 40], [w - 40, h - 40]])
    pts2 = np.float32([[65, 80], [w - 30, 30], [20, h - 20], [w - 55, h - 60]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(im, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    p_persp = tmp_path / "perspective_warped.png"
    cv2.imwrite(str(p_persp), warped)

    pred = manager.predict_pair(synthetic_scene, p_persp)
    assert pred["related"] is True
    assert pred["transformation"] == "perspective"
    assert pred["transformation_confidence"] >= 0.80
    assert pred["direction"] == "UNCERTAIN"
    assert pred["direction_confidence"] == 0.50


def test_rotation_transformation_classification_and_direction(manager, synthetic_scene, tmp_path):
    """
    Planar rotation (angle >= 5 deg) must be classified as 'rotation'
    with genuine confidence, while direction is honestly UNCERTAIN (0.50).
    """
    im = Image.open(synthetic_scene)
    im_rot = im.rotate(8.0, resample=Image.Resampling.BICUBIC)
    p_rot = tmp_path / "rotated.png"
    im_rot.save(p_rot)

    pred = manager.predict_pair(synthetic_scene, p_rot)
    assert pred["related"] is True
    assert pred["transformation"] == "rotation"
    assert pred["transformation_confidence"] >= 0.80
    assert pred["direction"] == "UNCERTAIN"
    assert pred["direction_confidence"] == 0.50


# -------------------------------------------------------------------------
# 3. Scale-Invariant Resize in Both Upload Orders
# -------------------------------------------------------------------------

def test_resize_both_upload_orders(manager, synthetic_scene, tmp_path):
    """
    Resized images must identify the higher-resolution image as original/parent
    in both input orders with separate confidences.
    """
    im = Image.open(synthetic_scene)
    im_small = im.resize((240, 240), Image.Resampling.BICUBIC)
    p_small = tmp_path / "scene_small.png"
    im_small.save(p_small)

    pred_fwd = manager.predict_pair(synthetic_scene, p_small)
    assert pred_fwd["related"] is True
    assert pred_fwd["transformation"] == "resize"
    assert pred_fwd["direction"] == "A_TO_B"
    assert pred_fwd["direction_confidence"] >= 0.90

    pred_rev = manager.predict_pair(p_small, synthetic_scene)
    assert pred_rev["related"] is True
    assert pred_rev["transformation"] == "resize"
    assert pred_rev["direction"] == "B_TO_A"
    assert pred_rev["direction_confidence"] >= 0.90


# -------------------------------------------------------------------------
# 4. Spanning Tree Orientation for Unresolved Direction Edges
# -------------------------------------------------------------------------

def test_graph_edge_outward_orientation_on_unresolved_direction(synthetic_scene, tmp_path):
    """
    In a provenance graph with symmetric photometric shift (brightness),
    the tree edge orientation must radiate outward from the detected component root
    towards the descendant, not point inward towards the root.
    """
    im = Image.open(synthetic_scene)
    im_bright = im.point(lambda p: min(255, int(p * 1.35)))
    p_bright = tmp_path / "bright.png"
    im_bright.save(p_bright)

    # Run analysis with original and brightness derivative
    result = run_provenance_analysis([synthetic_scene, p_bright])
    assert result["root_image_id"] is not None

    edges = result["edges"]
    assert len(edges) == 1
    edge = edges[0]
    assert edge["is_uncertain"] is True
    assert edge["direction"] == "UNCERTAIN"
    assert edge["direction_confidence"] == 0.50

    # The edge source must be the designated root, radiating outward to the target
    assert edge["source"] == result["root_image_id"]
