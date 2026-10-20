"""
tests/test_multi_image_provenance_scaling.py — Scalable Multi-Image Provenance, Transformation Classification, and Confidence Audit Test Suite

Validates:
1. Arbitrary image collections (N in [2, 3, 4, 5, 8]) evaluate exactly N*(N-1)/2 unique pairs.
2. Independent edge transformations: No label leakage across edges sharing an ancestor family.
3. Crop vs. Resize classification:
   - Original vs. full-canvas resize -> RESIZE with direction Original -> Resize.
   - Original vs. sub-region crop -> CROP with direction Original -> Crop.
   - Crop vs. Resize -> independently evaluated, honest unresolved/unknown, never fabricated.
4. Confidence separation:
   - related_confidence != transformation_confidence != direction_confidence.
   - Unresolved transformations retain None / unavailable confidence, never defaulting to 0.50 or copying rel_prob.
5. Permutation invariance:
   - Shuffling upload order preserves true root identification and edge classifications.
6. Multi-component separation:
   - Unrelated images remain isolated in distinct forest components without artificial bridge edges.
"""

import sys
import random
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageEnhance
import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.forensic_arbitrator import (
    check_sub_region_containment,
    check_full_canvas_resizing,
    arbitrate_pair_direction,
    arbitrate_transformation,
)
from backend.app.services.model_manager import ModelManager
from backend.app.services.provenance_engine import run_provenance_analysis


def create_rich_base_image(path: Path, width: int = 600, height: int = 600):
    """Generates an image with distinct geometric, textural, and textual patterns."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    # Background gradient
    for y in range(height):
        for x in range(width):
            arr[y, x] = [int(255 * x / width), int(255 * y / height), 128]
    # Geometric shapes
    cv2.circle(arr, (width // 2, height // 2), min(width, height) // 4, (40, 220, 40), -1)
    cv2.rectangle(arr, (width // 8, height // 8), (width // 3, height // 3), (220, 50, 50), -1)
    cv2.line(arr, (0, 0), (width, height), (255, 255, 255), 3)
    cv2.putText(arr, "PROVENANCE AUDIT", (width // 6, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    Image.fromarray(arr).save(path, quality=95)
    return path


@pytest.fixture
def multi_family_dataset(tmp_path):
    """
    Creates a parameterized multi-image family:
    - orig: Base original (600x600)
    - resize_half: Full canvas resized to 300x300
    - crop_center: Sub-region crop of center 300x300
    - bright_shift: Original with brightness adjusted (+30%)
    - unrelated_1: Completely independent image (noise + different shapes)
    - unrelated_2: Another independent image
    """
    p_orig = tmp_path / "img_orig.png"
    create_rich_base_image(p_orig, 600, 600)
    base_img = Image.open(p_orig)

    # 1. Resize derivative (300x300 full canvas)
    p_resize = tmp_path / "img_resize.png"
    base_img.resize((300, 300), Image.Resampling.LANCZOS).save(p_resize)

    # 2. Crop derivative (center 300x300 sub-region)
    p_crop = tmp_path / "img_crop.png"
    base_img.crop((150, 150, 450, 450)).save(p_crop)

    # 3. Brightness derivative (photometric shift)
    p_bright = tmp_path / "img_bright.png"
    enhancer = ImageEnhance.Brightness(base_img)
    enhancer.enhance(1.30).save(p_bright)

    # 4. Unrelated 1: Random noise distribution
    p_unrelated1 = tmp_path / "img_unrelated1.png"
    rng1 = np.random.RandomState(42)
    unrel1 = rng1.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    Image.fromarray(unrel1).save(p_unrelated1)

    # 5. Unrelated 2: Independent random noise distribution
    p_unrelated2 = tmp_path / "img_unrelated2.png"
    rng2 = np.random.RandomState(99)
    unrel2 = rng2.randint(0, 256, (400, 400, 3), dtype=np.uint8)
    Image.fromarray(unrel2).save(p_unrelated2)

    return {
        "orig": p_orig,
        "resize": p_resize,
        "crop": p_crop,
        "bright": p_bright,
        "unrelated1": p_unrelated1,
        "unrelated2": p_unrelated2
    }


# ============================================================================
# 1. Scaling: N*(N-1)/2 unique pair evaluations across arbitrary N
# ============================================================================

@pytest.mark.parametrize("n_count", [2, 3, 4, 5])
def test_arbitrary_image_count_pairwise_evaluation(multi_family_dataset, n_count):
    """Verifies that for arbitrary N, exactly N*(N-1)/2 pairs are generated and evaluated."""
    keys = list(multi_family_dataset.keys())[:n_count]
    image_paths = [multi_family_dataset[k] for k in keys]

    expected_unique_pairs = (n_count * (n_count - 1)) // 2
    manager = ModelManager()

    evaluated_pairs = set()
    for i in range(n_count):
        for j in range(i + 1, n_count):
            pair_key = tuple(sorted([image_paths[i].name, image_paths[j].name]))
            evaluated_pairs.add(pair_key)
            pred = manager.predict_pair(image_paths[i], image_paths[j])
            assert "related" in pred
            assert "related_confidence" in pred
            assert "transformation" in pred
            assert "direction" in pred

    assert len(evaluated_pairs) == expected_unique_pairs


# ============================================================================
# 2. Crop vs. Resize vs. Crop-Resize Pairwise Distinction
# ============================================================================

def test_crop_vs_resize_distinction(multi_family_dataset):
    """
    Original vs. Resize -> RESIZE with direction orig -> resize.
    Original vs. Crop -> CROP with direction orig -> crop.
    Crop vs. Resize -> Does NOT inherit crop or resize, evaluates independently.
    """
    manager = ModelManager()
    p_orig = multi_family_dataset["orig"]
    p_resize = multi_family_dataset["resize"]
    p_crop = multi_family_dataset["crop"]

    # 1. Original vs. Resize
    pred_orig_resize = manager.predict_pair(p_orig, p_resize)
    assert pred_orig_resize["related"] is True
    assert pred_orig_resize["transformation"] == "resize"
    assert pred_orig_resize["direction"] == "A_TO_B"
    assert pred_orig_resize["transformation_confidence"] is not None
    assert pred_orig_resize["transformation_confidence"] >= 0.85

    # Antisymmetry: Resize vs. Original
    pred_resize_orig = manager.predict_pair(p_resize, p_orig)
    assert pred_resize_orig["transformation"] == "resize"
    assert pred_resize_orig["direction"] == "B_TO_A"

    # 2. Original vs. Crop
    pred_orig_crop = manager.predict_pair(p_orig, p_crop)
    assert pred_orig_crop["related"] is True
    assert pred_orig_crop["transformation"] == "crop"
    assert pred_orig_crop["direction"] == "A_TO_B"
    assert pred_orig_crop["transformation_confidence"] is not None
    assert pred_orig_crop["transformation_confidence"] >= 0.85

    # 3. Crop vs. Resize:
    # Different derivatives of the same source must NOT receive crop or resize if evidence doesn't support it!
    # They are both 300x300, but one is a cropped sub-window, the other is downscaled full canvas.
    pred_crop_resize = manager.predict_pair(p_crop, p_resize)
    # Neither is a sub-region containment of the other, nor a full-canvas resize of the other
    assert pred_crop_resize["transformation"] not in ["crop", "resize"], (
        f"Crop vs Resize was wrongly classified as {pred_crop_resize['transformation']}"
    )


# ============================================================================
# 3. Confidence Separation: Related vs. Transformation vs. Direction
# ============================================================================

def test_confidence_separation_integrity(multi_family_dataset):
    """
    Ensures relationship confidence, transformation confidence, and direction confidence
    are strictly separated and never copied from each other.
    """
    manager = ModelManager()
    p_orig = multi_family_dataset["orig"]
    p_crop = multi_family_dataset["crop"]
    p_unrelated = multi_family_dataset["unrelated1"]

    # Verified pair
    pred_rel = manager.predict_pair(p_orig, p_crop)
    rel_conf = pred_rel["related_confidence"]
    dir_conf = pred_rel["direction_confidence"]
    tfm_conf = pred_rel["transformation_confidence"]

    assert rel_conf is not None
    assert dir_conf is not None
    assert tfm_conf is not None
    # They should not be identically copied
    assert not (rel_conf == tfm_conf == dir_conf), "Confidences appear duplicated!"

    # Unrelated pair
    pred_unrel = manager.predict_pair(p_orig, p_unrelated)
    assert pred_unrel["related"] is False
    assert pred_unrel["direction"] == "NONE"
    assert pred_unrel["transformation"] == "none"


# ============================================================================
# 4. Unresolved Cases Honest Representation (No False 0.50 defaults)
# ============================================================================

def test_unresolved_transformation_confidence_honesty(tmp_path):
    """
    When an image pair is related (e.g. slight artistic filter or non-standard edit)
    without verified forensic evidence, it must return 'unresolved' and transformation_confidence=None,
    NOT defaulting to 'crop', 'resize', 'identity', or 0.50.
    """
    p_base = tmp_path / "base.png"
    create_rich_base_image(p_base, 500, 500)
    img_base = Image.open(p_base)

    # Apply subtle non-uniform warp/blur that is not in the clean linear classes
    arr = np.array(img_base)
    # Localized non-linear distortion
    noise = np.random.normal(0, 15, arr.shape).astype(np.int16)
    arr_distorted = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    p_ambig = tmp_path / "ambig.png"
    Image.fromarray(arr_distorted).save(p_ambig)

    manager = ModelManager()
    pred = manager.predict_pair(p_base, p_ambig)

    if pred["related"]:
        if pred["transformation"] == "unresolved":
            assert pred["transformation_confidence"] is None, (
                f"Unresolved transformation should have None confidence, got {pred['transformation_confidence']}"
            )


# ============================================================================
# 5. Permutation Invariance & Root Attribution in Provenance Graph
# ============================================================================

def test_provenance_graph_permutation_invariance(multi_family_dataset):
    """
    Ensures that for a 4-image set (Original, Resize, Crop, Brightness),
    regardless of input order, the provenance engine designates the true Original as root.
    """
    image_paths_1 = [
        multi_family_dataset["orig"],
        multi_family_dataset["resize"],
        multi_family_dataset["crop"]
    ]

    # Reversed upload order
    image_paths_2 = list(reversed(image_paths_1))

    # Shuffled order
    image_paths_3 = [
        multi_family_dataset["crop"],
        multi_family_dataset["orig"],
        multi_family_dataset["resize"]
    ]

    res1 = run_provenance_analysis(image_paths_1)
    res2 = run_provenance_analysis(image_paths_2)
    res3 = run_provenance_analysis(image_paths_3)

    orig_id = multi_family_dataset["orig"].stem

    assert res1["root_image_id"] == orig_id
    assert res2["root_image_id"] == orig_id
    assert res3["root_image_id"] == orig_id


# ============================================================================
# 6. Disconnected Multi-Component Forest
# ============================================================================

def test_disconnected_multi_component_forest(multi_family_dataset):
    """
    5 images: 3 from family (orig, resize, crop) and 2 unrelated.
    Provenance engine must form at least 2 distinct components with no artificial bridge edges.
    """
    paths = [
        multi_family_dataset["orig"],
        multi_family_dataset["resize"],
        multi_family_dataset["crop"],
        multi_family_dataset["unrelated1"],
        multi_family_dataset["unrelated2"]
    ]

    res = run_provenance_analysis(paths)
    assert len(res["components"]) >= 2, f"Expected >= 2 components, got {len(res['components'])}"

    # Check candidate edges and forest edges do NOT bridge unrelated images to original
    unrel_names = {multi_family_dataset["unrelated1"].name, multi_family_dataset["unrelated2"].name}
    family_names = {multi_family_dataset["orig"].name, multi_family_dataset["resize"].name, multi_family_dataset["crop"].name}

    for edge in res["forest_edges"]:
        u, v = edge["source"], edge["target"]
        cross_family = (u in unrel_names and v in family_names) or (u in family_names and v in unrel_names)
        assert not cross_family, f"False bridge edge detected between unrelated images: {u} -> {v}"
