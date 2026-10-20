import pytest
from pathlib import Path
import numpy as np
import cv2

from backend.app.services.model_manager import ModelManager
from backend.app.services.provenance_engine import run_provenance_analysis
from src.forensic_arbitrator import check_sub_region_containment, arbitrate_transformation

UPLOAD_DIR = Path("data/uploads/anl_20261019_201200_039ee6")
ORIG_IMG = UPLOAD_DIR / "WhatsApp_Image_2026-10-13_at_1.31.12_AM.jpeg"
CROP_IMG = UPLOAD_DIR / "WhatsApp_Image_2026-10-13_at_1.31.11_AM.jpeg"
CONTRAST_IMG = UPLOAD_DIR / "WhatsApp_Image_2026-10-18_at_10.25.00_PM.jpeg"
BRIGHTNESS_IMG = UPLOAD_DIR / "WhatsApp_Image_2026-10-18_at_8.06.50_PM.jpeg"
BLUR_IMG = UPLOAD_DIR / "WhatsApp_Image_2026-10-19_at_3.36.39_PM.jpeg"

ORIG_ID = "WhatsApp_Image_2026-10-13_at_1.31.12_AM"
CROP_ID = "WhatsApp_Image_2026-10-13_at_1.31.11_AM"
CONTRAST_ID = "WhatsApp_Image_2026-10-18_at_10.25.00_PM"
BRIGHTNESS_ID = "WhatsApp_Image_2026-10-18_at_8.06.50_PM"
BLUR_ID = "WhatsApp_Image_2026-10-19_at_3.36.39_PM"


@pytest.fixture(scope="module")
def model_manager():
    mgr = ModelManager()
    mgr.initialize()
    return mgr


def test_live_incident_exact_images_provenance_tree():
    """
    Mandatory Validation 1, 2, 3:
    Using exact live images from analysis anl_20261019_201200_039ee6:
    Assert the final tree contains:
      - Original -> Crop (crop)
      - Original -> Brightness (brightness)
      - Original -> Contrast (contrast)
      - Original -> Blur (blur)
    Assert NO false Crop <-> Blur tree edge.
    """
    images = [ORIG_IMG, CROP_IMG, CONTRAST_IMG, BRIGHTNESS_IMG, BLUR_IMG]
    for img in images:
        assert img.exists(), f"Image fixture missing: {img}"

    res = run_provenance_analysis(images, similarity_threshold=0.55, confidence_filter=0.40)

    assert res["root_image_id"] == ORIG_ID, f"Expected root {ORIG_ID}, got {res['root_image_id']}"

    tree_edges = {(e["source"], e["target"]): e for e in res["edges"]}
    assert len(tree_edges) == 4, f"Expected exactly 4 tree edges, got {len(tree_edges)}"

    # 1. Original -> Crop
    assert (ORIG_ID, CROP_ID) in tree_edges, f"Missing edge {ORIG_ID} -> {CROP_ID}"
    edge_crop = tree_edges[(ORIG_ID, CROP_ID)]
    assert edge_crop["transformation"] == "crop"
    assert edge_crop["direction"] == "A_TO_B"

    # 2. Original -> Brightness
    assert (ORIG_ID, BRIGHTNESS_ID) in tree_edges, f"Missing edge {ORIG_ID} -> {BRIGHTNESS_ID}"
    edge_bright = tree_edges[(ORIG_ID, BRIGHTNESS_ID)]
    assert edge_bright["transformation"] == "brightness"

    # 3. Original -> Contrast
    assert (ORIG_ID, CONTRAST_ID) in tree_edges, f"Missing edge {ORIG_ID} -> {CONTRAST_ID}"
    edge_contrast = tree_edges[(ORIG_ID, CONTRAST_ID)]
    assert edge_contrast["transformation"] == "contrast"

    # 4. Original -> Blur
    assert (ORIG_ID, BLUR_ID) in tree_edges, f"Missing edge {ORIG_ID} -> {BLUR_ID}"
    edge_blur = tree_edges[(ORIG_ID, BLUR_ID)]
    assert edge_blur["transformation"] == "blur"
    assert edge_blur["direction"] == "A_TO_B"

    # 5. Assert NO false Crop <-> Blur edge in tree
    assert (CROP_ID, BLUR_ID) not in tree_edges, f"False edge {CROP_ID} -> {BLUR_ID} found in tree"
    assert (BLUR_ID, CROP_ID) not in tree_edges, f"False edge {BLUR_ID} -> {CROP_ID} found in tree"


def test_contrast_vs_brightness_order_invariance(model_manager):
    """
    Mandatory Validation Fix 2:
    Assert contrast vs brightness precedence and forward/reverse order invariance.
    """
    # Orig <-> Contrast
    fwd_c = model_manager.predict_pair(ORIG_IMG, CONTRAST_IMG)
    rev_c = model_manager.predict_pair(CONTRAST_IMG, ORIG_IMG)
    assert fwd_c["transformation"] == "contrast"
    assert rev_c["transformation"] == "contrast"
    assert fwd_c["transformation_confidence"] >= 0.88
    assert rev_c["transformation_confidence"] >= 0.88

    # Orig <-> Brightness
    fwd_b = model_manager.predict_pair(ORIG_IMG, BRIGHTNESS_IMG)
    rev_b = model_manager.predict_pair(BRIGHTNESS_IMG, ORIG_IMG)
    assert fwd_b["transformation"] == "brightness"
    assert rev_b["transformation"] == "brightness"
    assert fwd_b["transformation_confidence"] >= 0.88
    assert rev_b["transformation_confidence"] >= 0.88


def test_direction_uncertainty_separation(model_manager):
    """
    Mandatory Validation Fix 4:
    Keep pairwise physical direction as UNCERTAIN for photometric shifts,
    while tree lineage direction remains parent -> child.
    """
    pred_c = model_manager.predict_pair(ORIG_IMG, CONTRAST_IMG)
    pred_b = model_manager.predict_pair(ORIG_IMG, BRIGHTNESS_IMG)

    # Pairwise physical direction must remain UNCERTAIN
    assert pred_c["direction"] == "UNCERTAIN"
    assert pred_b["direction"] == "UNCERTAIN"

    # But full provenance analysis yields parent -> derivative tree edges with lineage metadata
    res = run_provenance_analysis([ORIG_IMG, CONTRAST_IMG, BRIGHTNESS_IMG], similarity_threshold=0.55)
    for edge in res["edges"]:
        assert edge["tree_lineage"] == f"{ORIG_ID} -> {edge['target']}"
        assert edge["is_tree_edge"] is True
        assert edge["is_uncertain"] is True
        assert edge["pairwise_direction"] == "UNCERTAIN"


def test_crop_blur_no_false_containment():
    """
    Mandatory Validation Fix 3:
    Crop <-> Blur shared scene content must not produce a containment edge.
    """
    c_in_b, _ = check_sub_region_containment(CROP_IMG, BLUR_IMG)
    b_in_c, _ = check_sub_region_containment(BLUR_IMG, CROP_IMG)
    assert c_in_b == 0.0, f"Expected 0.0 containment, got {c_in_b}"
    assert b_in_c == 0.0, f"Expected 0.0 containment, got {b_in_c}"


def test_upload_order_permutations():
    """
    Mandatory Validation 4:
    Assert upload-order permutations preserve root selection and tree edges.
    """
    images_rev = [BLUR_IMG, BRIGHTNESS_IMG, CONTRAST_IMG, CROP_IMG, ORIG_IMG]
    res_rev = run_provenance_analysis(images_rev, similarity_threshold=0.55, confidence_filter=0.40)

    assert res_rev["root_image_id"] == ORIG_ID
    tree_pairs = {(e["source"], e["target"]) for e in res_rev["edges"]}
    assert (ORIG_ID, CROP_ID) in tree_pairs
    assert (ORIG_ID, BRIGHTNESS_ID) in tree_pairs
    assert (ORIG_ID, CONTRAST_ID) in tree_pairs
    assert (ORIG_ID, BLUR_ID) in tree_pairs
    assert (CROP_ID, BLUR_ID) not in tree_pairs


def test_nested_crops_and_crop_plus_blur(tmp_path):
    """
    Mandatory Validation 4:
    Test nested crops and genuine crop-plus-blur relationships are preserved.
    """
    # Create parent image
    img = np.zeros((600, 800), dtype=np.uint8)
    cv2.putText(img, "TRACE_AI_ORIGINAL", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
    cv2.circle(img, (400, 300), 100, 200, -1)
    parent_path = tmp_path / "parent.png"
    cv2.imwrite(str(parent_path), img)

    # 1. Nested crop 1: [100:500, 100:700]
    crop1 = img[100:500, 100:700]
    crop1_path = tmp_path / "crop1.png"
    cv2.imwrite(str(crop1_path), crop1)

    # 2. Nested crop 2 (sub-crop of crop 1): [100:300, 100:500] of crop1
    crop2 = crop1[100:300, 100:500]
    crop2_path = tmp_path / "crop2.png"
    cv2.imwrite(str(crop2_path), crop2)

    # 3. Genuine Crop + Blur: crop1 blurred
    crop_blur = cv2.GaussianBlur(crop1, (15, 15), 0)
    crop_blur_path = tmp_path / "crop_blur.png"
    cv2.imwrite(str(crop_blur_path), crop_blur)

    # Assert parent -> crop1 containment
    p_c1, _ = check_sub_region_containment(parent_path, crop1_path)
    assert p_c1 == 1.0, f"Parent -> Crop1 expected 1.0, got {p_c1}"

    # Assert crop1 -> crop2 nested containment
    c1_c2, _ = check_sub_region_containment(crop1_path, crop2_path)
    assert c1_c2 == 1.0, f"Crop1 -> Crop2 expected 1.0, got {c1_c2}"

    # Assert parent -> crop_blur genuine crop+blur containment
    p_cb, _ = check_sub_region_containment(parent_path, crop_blur_path)
    assert p_cb == 1.0, f"Parent -> CropBlur expected 1.0, got {p_cb}"


def test_unrelated_scene_rejection(tmp_path):
    """
    Mandatory Validation 4:
    Assert unrelated images are not falsely linked to the provenance tree.
    """
    # Create an unrelated noise image
    noise = np.random.randint(0, 256, (600, 800), dtype=np.uint8)
    unrelated_path = tmp_path / "unrelated.png"
    cv2.imwrite(str(unrelated_path), noise)

    res = run_provenance_analysis([ORIG_IMG, unrelated_path], similarity_threshold=0.55)
    unrelated_id = "unrelated"
    tree_pairs = {(e["source"], e["target"]) for e in res["edges"]}
    assert (ORIG_ID, unrelated_id) not in tree_pairs
    assert (unrelated_id, ORIG_ID) not in tree_pairs
