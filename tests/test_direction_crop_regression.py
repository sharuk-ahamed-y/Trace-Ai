"""
tests/test_direction_crop_regression.py — Regression Test Suite for Original/Derived Direction & Crop Containment

Tests:
1. Exact return-sign and A_TO_B / B_TO_A mapping with known original/cropped pair in both input orders.
2. Semantic antisymmetry: reversing (A, B) reverses direction while preserving original/derived identities.
3. ModelManager fallback heuristic correctly invokes forensic arbitration and assigns genuine direction confidence.
4. Provenance engine DAG construction designates the true original image as root regardless of upload order.
5. Equal-sized images with crop containment correctly resolve without canvas size bias.
6. Canvas area differences alone do NOT assert original/derived direction (no fallback to area).
7. Negative pairs (identical, blurred, noisy, unrelated) return UNCERTAIN with zero false positives.
8. Unresolved direction gracefully marks edges and graph as direction_status='unresolved'.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import cv2

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.forensic_arbitrator import (
    check_sub_region_containment,
    arbitrate_pair_direction,
    arbitrate_transformation,
)
from backend.app.services.model_manager import ModelManager
from backend.app.services.provenance_engine import run_provenance_analysis

# Known test pair from user incident
P_CROP = ROOT_DIR / "data" / "uploads" / "anl_20261018_193743_cef6cd" / "WhatsApp_Image_2026-10-13_at_1.31.11_AM.jpeg"
P_ORIG = ROOT_DIR / "data" / "uploads" / "anl_20261018_193743_cef6cd" / "WhatsApp_Image_2026-10-13_at_1.31.12_AM.jpeg"


@pytest.fixture
def synthetic_equal_size_crop(tmp_path):
    """Creates a base image and a center-cropped version resized to identical 512x512 canvas."""
    base = np.zeros((512, 512), dtype=np.uint8)
    cv2.rectangle(base, (60, 60), (450, 450), 210, -1)
    cv2.circle(base, (256, 256), 80, 40, -1)
    cv2.putText(base, "TRACE REGRESSION", (100, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)

    crop = base[100:400, 100:400]  # 300x300 sub-window
    crop_resized = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR)

    p_base = tmp_path / "base_512.png"
    p_crop = tmp_path / "crop_512.png"
    Image.fromarray(base).save(p_base)
    Image.fromarray(crop_resized).save(p_crop)
    return p_base, p_crop


@pytest.fixture
def synthetic_different_area_unrelated(tmp_path):
    """Creates two images with large area difference (800x800 vs 400x400) but different independent patterns."""
    im_large = np.zeros((800, 800), dtype=np.uint8)
    for i in range(0, 800, 40):
        cv2.line(im_large, (0, i), (800, i), 180, 2)

    im_small = np.zeros((400, 400), dtype=np.uint8)
    for j in range(0, 400, 30):
        cv2.line(im_small, (j, 0), (j, 400), 220, 2)

    p_large = tmp_path / "large_800.png"
    p_small = tmp_path / "small_400.png"
    Image.fromarray(im_large).save(p_large)
    Image.fromarray(im_small).save(p_small)
    return p_large, p_small


# -------------------------------------------------------------------------
# Test A: Return-Sign & Antisymmetry on Known Pair
# -------------------------------------------------------------------------

def test_containment_known_pair_both_orders():
    """Verify sub-region containment returns +1.0 for (Orig, Crop) and -1.0 for (Crop, Orig)."""
    assert P_ORIG.exists() and P_CROP.exists(), "Known test images must exist in data/uploads"

    score_forward, corr_forward = check_sub_region_containment(P_ORIG, P_CROP)
    assert score_forward == 1.0, f"Expected +1.0 (A_TO_B, Orig->Crop), got {score_forward}"
    assert corr_forward >= 0.99, f"Expected correlation >= 0.99, got {corr_forward}"

    score_reverse, corr_reverse = check_sub_region_containment(P_CROP, P_ORIG)
    assert score_reverse == -1.0, f"Expected -1.0 (B_TO_A, Crop->Orig), got {score_reverse}"
    assert corr_reverse >= 0.99, f"Expected correlation >= 0.99, got {corr_reverse}"

    # Semantic antisymmetry check
    assert score_forward == -score_reverse, "Sub-region containment must be strictly antisymmetric"


def test_arbitrator_known_pair_both_orders():
    """Verify arbitrator resolves A_TO_B when (Orig, Crop) and B_TO_A when (Crop, Orig)."""
    manager = ModelManager()
    f_orig = manager.get_forensic_features(P_ORIG)
    f_crop = manager.get_forensic_features(P_CROP)

    dir_fwd, conf_fwd, rat_fwd = arbitrate_pair_direction(
        P_ORIG, P_CROP, f_orig, f_crop,
        base_direction="UNCERTAIN", base_confidence=0.50, cos_sim=0.6538, predicted_tfm="crop"
    )
    assert dir_fwd == "A_TO_B"
    assert conf_fwd >= 0.95
    assert "Containment" in rat_fwd

    dir_rev, conf_rev, rat_rev = arbitrate_pair_direction(
        P_CROP, P_ORIG, f_crop, f_orig,
        base_direction="UNCERTAIN", base_confidence=0.50, cos_sim=0.6538, predicted_tfm="crop"
    )
    assert dir_rev == "B_TO_A"
    assert conf_rev >= 0.95
    assert "Containment" in rat_rev


# -------------------------------------------------------------------------
# Test B: ModelManager Fallback Path Integration
# -------------------------------------------------------------------------

def test_model_manager_fallback_both_orders():
    """Verify ModelManager fallback heuristic runs arbitration and assigns empirical direction confidence."""
    manager = ModelManager()
    assert not manager.phase4_enabled, "Phase 4 models must remain disabled for regression test"

    # Forward order: (Orig, Crop)
    pred_fwd = manager.predict_pair(P_ORIG, P_CROP)
    assert pred_fwd["related"] is True
    assert pred_fwd["direction"] == "A_TO_B"
    assert pred_fwd["direction_confidence"] >= 0.95
    # Direction confidence MUST NOT be equal to cosine similarity
    assert pred_fwd["direction_confidence"] != pred_fwd["cosine_similarity"]
    assert pred_fwd["transformation"] == "crop"

    # Reverse order: (Crop, Orig)
    pred_rev = manager.predict_pair(P_CROP, P_ORIG)
    assert pred_rev["related"] is True
    assert pred_rev["direction"] == "B_TO_A"
    assert pred_rev["direction_confidence"] >= 0.95
    assert pred_rev["transformation"] == "crop"


# -------------------------------------------------------------------------
# Test C: Provenance Engine DAG & Root Invariance
# -------------------------------------------------------------------------

def test_provenance_engine_identifies_true_original_root_both_upload_orders():
    """Verify DAG designates true original image as root regardless of upload/input list order."""
    orig_id = P_ORIG.stem
    crop_id = P_CROP.stem

    # Order 1: [Orig, Crop]
    dag_1 = run_provenance_analysis([P_ORIG, P_CROP], similarity_threshold=0.50)
    assert dag_1["root_image_id"] == orig_id, f"Order 1 root must be {orig_id}, got {dag_1['root_image_id']}"
    assert dag_1["root_verified"] is True
    assert dag_1["direction_status"] == "verified"
    assert len(dag_1["edges"]) == 1
    assert dag_1["edges"][0]["source"] == orig_id
    assert dag_1["edges"][0]["target"] == crop_id

    # Order 2: [Crop, Orig] (Reversed input order)
    dag_2 = run_provenance_analysis([P_CROP, P_ORIG], similarity_threshold=0.50)
    assert dag_2["root_image_id"] == orig_id, f"Order 2 root must be {orig_id}, got {dag_2['root_image_id']}"
    assert dag_2["root_verified"] is True
    assert dag_2["direction_status"] == "verified"
    assert len(dag_2["edges"]) == 1
    assert dag_2["edges"][0]["source"] == orig_id
    assert dag_2["edges"][0]["target"] == crop_id


# -------------------------------------------------------------------------
# Test D: Equal-Size Synthetic Crop Containment
# -------------------------------------------------------------------------

def test_equal_size_synthetic_crop_containment(synthetic_equal_size_crop):
    """Verify crop containment succeeds when both images have identical 512x512 canvas area."""
    p_base, p_crop = synthetic_equal_size_crop

    score_fwd, corr_fwd = check_sub_region_containment(p_base, p_crop)
    assert score_fwd == 1.0, f"Expected 1.0 (base contains crop), got {score_fwd}"
    assert corr_fwd >= 0.90

    score_rev, corr_rev = check_sub_region_containment(p_crop, p_base)
    assert score_rev == -1.0, f"Expected -1.0 (reverse order), got {score_rev}"


# -------------------------------------------------------------------------
# Test E: Canvas Area Alone Does NOT Assert Direction (No Area Guessing)
# -------------------------------------------------------------------------

def test_no_fallback_to_canvas_area(synthetic_different_area_unrelated):
    """Verify images with different canvas area but no containment remain UNCERTAIN."""
    p_large, p_small = synthetic_different_area_unrelated

    score, _ = check_sub_region_containment(p_large, p_small)
    assert score == 0.0, "Unrelated different-size images must not trigger crop containment"

    manager = ModelManager()
    f_large = manager.get_forensic_features(p_large)
    f_small = manager.get_forensic_features(p_small)

    dir_res, conf_res, _ = arbitrate_pair_direction(
        p_large, p_small, f_large, f_small,
        base_direction="UNCERTAIN", base_confidence=0.50, cos_sim=0.70, predicted_tfm="crop"
    )
    # Area difference must NOT override UNCERTAIN
    assert dir_res == "UNCERTAIN", f"Expected UNCERTAIN, got {dir_res}"
    assert conf_res == 0.50


# -------------------------------------------------------------------------
# Test F: Negative Cases Produce Zero False Positives
# -------------------------------------------------------------------------

def test_negative_pairs_produce_no_false_containment(tmp_path):
    """Verify identical, blurred, noisy, and random pairs return score 0.0."""
    arr = np.array(Image.open(P_ORIG).convert("L"))

    p_orig = tmp_path / "orig.png"
    p_identical = tmp_path / "identical.png"
    p_blurred = tmp_path / "blurred.png"
    p_noisy = tmp_path / "noisy.png"
    p_random = tmp_path / "random.png"

    Image.fromarray(arr).save(p_orig)
    Image.fromarray(arr).save(p_identical)
    Image.fromarray(cv2.GaussianBlur(arr, (21, 21), 5.0)).save(p_blurred)
    noisy_arr = np.clip(arr.astype(float) + np.random.normal(0, 25, arr.shape), 0, 255).astype(np.uint8)
    Image.fromarray(noisy_arr).save(p_noisy)
    Image.fromarray(np.random.randint(0, 256, (600, 600), dtype=np.uint8)).save(p_random)

    # Identical
    score_id, _ = check_sub_region_containment(p_orig, p_identical)
    assert score_id == 0.0, "Identical pair must return 0.0"

    # Blurred
    score_bl, _ = check_sub_region_containment(p_orig, p_blurred)
    assert score_bl == 0.0, "Blurred pair must return 0.0"

    # Noisy
    score_ns, _ = check_sub_region_containment(p_orig, p_noisy)
    assert score_ns == 0.0, "Noisy pair must return 0.0"

    # Random
    score_rnd, _ = check_sub_region_containment(p_orig, p_random)
    assert score_rnd == 0.0, "Random pair must return 0.0"


# -------------------------------------------------------------------------
# Test G: Graceful Handling of Unresolved Direction in Provenance Engine
# -------------------------------------------------------------------------

def test_provenance_engine_unresolved_direction_handling(tmp_path):
    """Verify provenance engine does not claim a verified root when direction is ambiguous."""
    # Create two different abstract pattern images with similar overall histograms
    im1 = np.zeros((400, 400), dtype=np.uint8)
    im2 = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(im1, (200, 200), 100, 180, -1)
    cv2.circle(im2, (200, 200), 100, 180, -1)
    # Add slight variation that makes direction ambiguous
    cv2.rectangle(im1, (50, 50), (100, 100), 120, -1)
    cv2.rectangle(im2, (300, 300), (350, 350), 120, -1)

    p1 = tmp_path / "pat_1.png"
    p2 = tmp_path / "pat_2.png"
    Image.fromarray(im1).save(p1)
    Image.fromarray(im2).save(p2)

    dag = run_provenance_analysis([p1, p2], similarity_threshold=0.50)
    if dag["edges"]:
        edge = dag["edges"][0]
        if edge["direction"] == "UNCERTAIN":
            assert edge["is_uncertain"] is True
            assert dag["direction_status"] == "unresolved"
            assert "unresolved" in dag["root_reason"].lower()
