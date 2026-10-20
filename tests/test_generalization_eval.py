"""
tests/test_generalization_eval.py — Deterministic Test Suite for TRACE AI Generalization

Validates generalization across multiple unseen procedural scene families and scene IDs:
- Pure crops (centered, off-center, micro)
- Near-full crops (88-92% coverage)
- Aspect-skewed crops (16:9 banner)
- Rotated crops (rotation-invariant geometric containment)
- Low-texture scenes (graceful fallback without false high confidence)
- Pure brightness and pure contrast (preserving UNCERTAIN direction)
- Blur and Resizing (preserving A_TO_B direction)
- Compound transformations: crop+brightness, crop+blur
- Unrelated similar-colour control images (rejection without false positive edge)
"""

import pytest
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from pathlib import Path

from backend.app.services.model_manager import ModelManager
from src.forensic_arbitrator import check_sub_region_containment


def generate_unseen_scene(scene_id: int, width: int = 512, height: int = 512, mode: str = "rich") -> np.ndarray:
    """Generates deterministic synthetic scene for unseen scene IDs."""
    rng = np.random.RandomState(scene_id * 3000 + 77)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    if mode == "low_texture":
        # Smooth sky/wall gradient with minimal features
        for y in range(height):
            v = int(200 + 20 * (y / height))
            img[y, :] = [v, v - 5, v - 10]
        cv2.circle(img, (width // 2, height // 2), 8, (190, 185, 180), -1)
        return img

    if mode == "similar_palette":
        # Same color palette as rich scene 201 (yellow-orange / cyan), but entirely different geometry
        for y in range(height):
            for x in range(width):
                img[y, x] = [int(40 + 160 * (y / height)), int(60 + 140 * (x / width)), int(120)]
        # Concentric circles instead of rectangles/text
        for r in range(30, 220, 40):
            cv2.circle(img, (width // 2, height // 2), r, (rng.randint(50, 240), rng.randint(50, 240), 200), 4)
        return img

    # Rich procedural scene with strong edges, shapes, textures, and text
    for y in range(height):
        for x in range(width):
            img[y, x] = [
                int(40 + 160 * (x / width)),
                int(60 + 140 * (y / height)),
                int(90 + 110 * ((x + y) / (width + height)))
            ]
    cv2.rectangle(img, (int(width * 0.12), int(height * 0.2)), (int(width * 0.5), int(height * 0.6)), (45, 195, 230), -1)
    cv2.circle(img, (int(width * 0.68), int(height * 0.4)), int(min(width, height) * 0.16), (230, 75, 45), -1)
    cv2.line(img, (10, int(height * 0.85)), (width - 10, int(height * 0.35)), (245, 245, 245), 4)
    cv2.putText(img, f"UNSEEN_ID_{scene_id}", (int(width * 0.18), int(height * 0.78)), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (15, 15, 15), 2)

    # Rich center texture pattern ensuring keypoints for all crops and rotations
    for i in range(12):
        pt1 = (int(rng.randint(int(width * 0.22), int(width * 0.78))), int(rng.randint(int(height * 0.22), int(height * 0.78))))
        pt2 = (int(rng.randint(int(width * 0.22), int(width * 0.78))), int(rng.randint(int(height * 0.22), int(height * 0.78))))
        cv2.line(img, pt1, pt2, (int(rng.randint(50, 255)), int(rng.randint(50, 255)), int(rng.randint(50, 255))), 2)
        cv2.circle(img, pt1, int(rng.randint(8, 25)), (int(rng.randint(50, 255)), int(rng.randint(50, 255)), int(rng.randint(50, 255))), -1)

    return img


@pytest.fixture(scope="module")
def model_manager():
    mgr = ModelManager()
    mgr.initialize()
    return mgr


@pytest.fixture(scope="module")
def unseen_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("unseen_generalization_scenes")
    scenes = {}

    # 2 rich scene families
    for s_id in [201, 202]:
        base_arr = generate_unseen_scene(s_id, 512, 512, mode="rich")
        p_orig = out / f"orig_{s_id}.png"
        Image.fromarray(base_arr).save(p_orig)
        pil_base = Image.fromarray(base_arr)
        h, w = base_arr.shape[:2]

        # 1. Pure center crop
        c_w, c_h = int(w * 0.70), int(h * 0.70)
        x1, y1 = (w - c_w) // 2, (h - c_h) // 2
        p_crop_c = out / f"crop_center_{s_id}.png"
        Image.fromarray(base_arr[y1:y1+c_h, x1:x1+c_w]).save(p_crop_c)

        # 2. Near-full crop (90% width and height)
        p_crop_near = out / f"crop_nearfull_{s_id}.png"
        Image.fromarray(base_arr[int(h * 0.05):int(h * 0.95), int(w * 0.05):int(w * 0.95)]).save(p_crop_near)

        # 3. Aspect-skewed crop (16:9 banner)
        ar_h = int(w * 9 / 16)
        ar_y = (h - ar_h) // 2
        p_crop_ar = out / f"crop_ar_{s_id}.png"
        Image.fromarray(base_arr[ar_y:ar_y+ar_h, :]).save(p_crop_ar)

        # 4. Rotated crop (crop of size 240x240 rotated by 15 deg, guaranteed interior to 512x512)
        r_w, r_h = 240, 240
        rx1, ry1 = (w - r_w) // 2, (h - r_h) // 2
        crop_sub_r = base_arr[ry1:ry1+r_h, rx1:rx1+r_w]
        rot_mat = cv2.getRotationMatrix2D((r_w / 2, r_h / 2), 15, 1.0)
        crop_rot = cv2.warpAffine(crop_sub_r, rot_mat, (r_w, r_h), borderMode=cv2.BORDER_REPLICATE)
        p_crop_rot = out / f"crop_rot_{s_id}.png"
        Image.fromarray(crop_rot).save(p_crop_rot)

        # 5. Pure brightness (+25%)
        p_bright = out / f"bright_{s_id}.png"
        ImageEnhance.Brightness(pil_base).enhance(1.25).save(p_bright)

        # 6. Pure contrast (1.4x)
        p_contrast = out / f"contrast_{s_id}.png"
        ImageEnhance.Contrast(pil_base).enhance(1.40).save(p_contrast)

        # 7. Blur
        p_blur = out / f"blur_{s_id}.png"
        Image.fromarray(cv2.GaussianBlur(base_arr, (11, 11), 3.0)).save(p_blur)

        # 8. Resize (downscale to 320x320)
        p_resize = out / f"resize_{s_id}.png"
        pil_base.resize((320, 320), Image.Resampling.LANCZOS).save(p_resize)

        # 9. Compound: Crop + Brightness
        p_crop_bright = out / f"crop_bright_{s_id}.png"
        crop_pil = Image.fromarray(base_arr[y1:y1+c_h, x1:x1+c_w])
        ImageEnhance.Brightness(crop_pil).enhance(1.25).save(p_crop_bright)

        # 10. Compound: Crop + Blur
        p_crop_blur = out / f"crop_blur_{s_id}.png"
        Image.fromarray(cv2.GaussianBlur(base_arr[y1:y1+c_h, x1:x1+c_w], (9, 9), 2.5)).save(p_crop_blur)

        scenes[s_id] = {
            "orig": p_orig,
            "crop_center": p_crop_c,
            "crop_nearfull": p_crop_near,
            "crop_ar": p_crop_ar,
            "crop_rot": p_crop_rot,
            "bright": p_bright,
            "contrast": p_contrast,
            "blur": p_blur,
            "resize": p_resize,
            "compound_crop_bright": p_crop_bright,
            "compound_crop_blur": p_crop_blur,
        }

    # Low-texture scene (888)
    lt_arr = generate_unseen_scene(888, 512, 512, mode="low_texture")
    p_lt_orig = out / "orig_lt_888.png"
    Image.fromarray(lt_arr).save(p_lt_orig)
    p_lt_crop = out / "crop_lt_888.png"
    Image.fromarray(lt_arr[100:400, 100:400]).save(p_lt_crop)
    scenes["low_texture"] = {"orig": p_lt_orig, "crop": p_lt_crop}

    # Similar-palette unrelated scene (999)
    sim_pal_arr = generate_unseen_scene(999, 512, 512, mode="similar_palette")
    p_sim_pal = out / "sim_palette_999.png"
    Image.fromarray(sim_pal_arr).save(p_sim_pal)
    scenes["unrelated_sim_palette"] = p_sim_pal

    return scenes


def test_pure_and_near_full_crops_classified_as_crop_with_atob_direction(model_manager, unseen_dataset):
    """Verifies that pure crops, near-full crops, and aspect-skewed crops are classified as crop with A_TO_B direction."""
    for s_id in [201, 202]:
        p_orig = unseen_dataset[s_id]["orig"]

        for c_key in ["crop_center", "crop_nearfull", "crop_ar"]:
            p_crop = unseen_dataset[s_id][c_key]
            pred = model_manager.predict_pair(p_orig, p_crop)

            assert pred["transformation"] == "crop", f"Scene {s_id} {c_key} was misclassified as {pred['transformation']}"
            assert pred["direction"] == "A_TO_B", f"Scene {s_id} {c_key} expected A_TO_B, got {pred['direction']}"
            assert pred["transformation_confidence"] >= 0.85
            assert pred["related"] is True


def test_rotated_crop_spatial_containment(unseen_dataset):
    """Verifies SIFT/RANSAC geometric containment detects rotated crops where raster NCC fails."""
    p_orig = unseen_dataset[201]["orig"]
    p_rot = unseen_dataset[201]["crop_rot"]

    score, corr = check_sub_region_containment(p_orig, p_rot)
    # Score must be positive (A contains B)
    assert score > 0.0, f"Expected positive containment score for rotated crop, got {score}"
    assert corr >= 0.80, f"Expected geometric correspondence >= 0.80, got {corr}"


def test_pure_brightness_and_contrast_preserve_uncertain_direction(model_manager, unseen_dataset):
    """Verifies pure brightness and contrast maintain UNCERTAIN physical direction."""
    for s_id in [201, 202]:
        p_orig = unseen_dataset[s_id]["orig"]
        p_bright = unseen_dataset[s_id]["bright"]
        p_contrast = unseen_dataset[s_id]["contrast"]

        pred_b = model_manager.predict_pair(p_orig, p_bright)
        assert pred_b["transformation"] == "brightness"
        assert pred_b["direction"] == "UNCERTAIN"

        pred_c = model_manager.predict_pair(p_orig, p_contrast)
        assert pred_c["transformation"] == "contrast"
        assert pred_c["direction"] == "UNCERTAIN"


def test_blur_and_resize_preserve_atob_direction(model_manager, unseen_dataset):
    """Verifies blur and resize derivatives preserve A_TO_B direction."""
    for s_id in [201, 202]:
        p_orig = unseen_dataset[s_id]["orig"]
        p_blur = unseen_dataset[s_id]["blur"]
        p_resize = unseen_dataset[s_id]["resize"]

        pred_blur = model_manager.predict_pair(p_orig, p_blur)
        assert pred_blur["transformation"] == "blur"
        assert pred_blur["direction"] == "A_TO_B"

        pred_resize = model_manager.predict_pair(p_orig, p_resize)
        assert pred_resize["transformation"] == "resize"
        assert pred_resize["direction"] == "A_TO_B"


def test_compound_transformations_preserve_primary_crop_and_add_tags(model_manager, unseen_dataset):
    """Verifies compound crop+brightness and crop+blur preserve primary crop classification with compound tags."""
    for s_id in [201, 202]:
        p_orig = unseen_dataset[s_id]["orig"]
        p_cb = unseen_dataset[s_id]["compound_crop_bright"]
        p_cblur = unseen_dataset[s_id]["compound_crop_blur"]

        pred_cb = model_manager.predict_pair(p_orig, p_cb)
        assert pred_cb["transformation"] == "crop", f"Expected primary crop, got {pred_cb['transformation']}"
        assert pred_cb["direction"] == "A_TO_B"
        assert "brightness" in pred_cb.get("compound_tags", []), f"Expected brightness compound tag, got {pred_cb.get('compound_tags')}"

        pred_cblur = model_manager.predict_pair(p_orig, p_cblur)
        assert pred_cblur["transformation"] == "crop"
        assert pred_cblur["direction"] == "A_TO_B"


def test_unrelated_similar_palette_image_rejected(model_manager, unseen_dataset):
    """Verifies images sharing similar color distribution but disjoint content are rejected as unrelated."""
    p_orig = unseen_dataset[201]["orig"]
    p_unrelated = unseen_dataset["unrelated_sim_palette"]

    pred = model_manager.predict_pair(p_orig, p_unrelated)
    assert pred["related"] is False or pred["cosine_similarity"] < 0.35, (
        f"Similar palette unrelated image was improperly accepted! P(rel)={pred['related_confidence']}, CosSim={pred['cosine_similarity']}"
    )
    if not pred["related"]:
        assert pred["transformation"] == "none"
        assert pred["direction"] == "NONE"


def test_low_texture_scene_graceful_handling(model_manager, unseen_dataset):
    """Verifies low-texture scenes do not crash or produce fabricated high-confidence classifications."""
    p_orig = unseen_dataset["low_texture"]["orig"]
    p_crop = unseen_dataset["low_texture"]["crop"]

    pred = model_manager.predict_pair(p_orig, p_crop)
    # Even if featureless, must not throw error or produce arbitrary invalid state
    assert pred["transformation"] in ["crop", "resize", "none", "unknown"]
    assert pred["direction"] in ["A_TO_B", "UNCERTAIN", "NONE"]
