"""
tests/test_generalization_baseline.py — Deterministic Isolated Development Harness for Transformation & Crop Generalization

Generates controlled procedural scenes (held completely separate from any benchmark data)
and systematically evaluates baseline ModelManager & ForensicArbitrator predictions:
- Spatial crops: centered, off-center, micro, medium, near-full, upscaled, aspect-ratio skewed, rotated, nested
- Photometrics: pure brightness, pure contrast
- Degeneracies: blur, resize, jpeg, noise, sharpen
- Compounds: crop+brightness, crop+blur, resize+brightness
- Controls: low-texture scene, visually similar unrelated scene

Measures:
- Pairwise classification accuracy & confusion matrix
- Direction correctness & calibration
- Candidate filter recall (whether parent-child pair is retained or omitted)
- Runtime per pair
"""

import time
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
from PIL import Image, ImageEnhance

from backend.app.services.model_manager import ModelManager


def generate_scene(scene_id: int, width: int = 512, height: int = 512, low_texture: bool = False) -> np.ndarray:
    """Generates a deterministic synthetic scene."""
    rng = np.random.RandomState(scene_id * 1000 + 42)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    if low_texture:
        # Smooth subtle gradient with minimal features
        for y in range(height):
            val = int(180 + 30 * (y / height))
            img[y, :] = [val, val - 10, val - 5]
        # Tiny subtle mark
        cv2.circle(img, (width // 2, height // 2), 12, (150, 140, 145), -1)
        return img

    # Rich textural scene
    # 1. Base gradient
    for y in range(height):
        for x in range(width):
            img[y, x] = [
                int(50 + 150 * (x / width)),
                int(80 + 120 * (y / height)),
                int(100 + 100 * ((x + y) / (width + height)))
            ]
    # 2. Geometric structures
    cv2.rectangle(img, (int(width * 0.1), int(height * 0.15)), (int(width * 0.45), int(height * 0.55)), (30, 180, 220), -1)
    cv2.circle(img, (int(width * 0.7), int(height * 0.35)), int(min(width, height) * 0.18), (220, 60, 40), -1)
    cv2.line(img, (0, int(height * 0.8)), (width, int(height * 0.4)), (240, 240, 240), 3)

    # 3. High-frequency texture / text
    cv2.putText(img, f"SCENE_{scene_id}_TRACE", (int(width * 0.15), int(height * 0.75)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    # Add localized noise block
    noise_block = rng.randint(0, 50, (int(height * 0.2), int(width * 0.2), 3), dtype=np.uint8)
    img[int(height * 0.2):int(height * 0.4), int(width * 0.5):int(width * 0.7)] = np.clip(
        img[int(height * 0.2):int(height * 0.4), int(width * 0.5):int(width * 0.7)] + noise_block, 0, 255
    )

    return img


def create_derivatives(base_img: np.ndarray, out_dir: Path) -> Dict[str, Path]:
    """Creates a full suite of controlled derivatives from a base image."""
    h, w = base_img.shape[:2]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # Base image
    p_orig = out_dir / "orig.png"
    Image.fromarray(base_img).save(p_orig)
    paths["orig"] = p_orig
    pil_base = Image.fromarray(base_img)

    # 1. Centered crop (50% area)
    c_w, c_h = int(w * 0.707), int(h * 0.707)
    x1, y1 = (w - c_w) // 2, (h - c_h) // 2
    p_crop_center = out_dir / "crop_center.png"
    Image.fromarray(base_img[y1:y1+c_h, x1:x1+c_w]).save(p_crop_center)
    paths["crop_center"] = p_crop_center

    # 2. Off-center crop (top-left quadrant)
    p_crop_offcenter = out_dir / "crop_offcenter.png"
    Image.fromarray(base_img[int(h * 0.05):int(h * 0.55), int(w * 0.05):int(w * 0.55)]).save(p_crop_offcenter)
    paths["crop_offcenter"] = p_crop_offcenter

    # 3. Micro crop (20% dimensions, 4% area)
    p_crop_micro = out_dir / "crop_micro.png"
    Image.fromarray(base_img[int(h * 0.4):int(h * 0.6), int(w * 0.4):int(w * 0.6)]).save(p_crop_micro)
    paths["crop_micro"] = p_crop_micro

    # 4. Near-full crop (88% dimensions, ~77% area)
    p_crop_nearfull = out_dir / "crop_nearfull.png"
    Image.fromarray(base_img[int(h * 0.06):int(h * 0.94), int(w * 0.06):int(w * 0.94)]).save(p_crop_nearfull)
    paths["crop_nearfull"] = p_crop_nearfull

    # 5. Upscaled crop (cropped 300x300, resized back to 512x512)
    crop_sub = base_img[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    crop_upscaled = cv2.resize(crop_sub, (w, h), interpolation=cv2.INTER_LANCZOS4)
    p_crop_upscaled = out_dir / "crop_upscaled.png"
    Image.fromarray(crop_upscaled).save(p_crop_upscaled)
    paths["crop_upscaled"] = p_crop_upscaled

    # 6. Aspect-ratio changed crop (16:9 slice from 1:1)
    target_h = int(w * 9 / 16)
    ar_y1 = (h - target_h) // 2
    p_crop_aspect = out_dir / "crop_aspect.png"
    Image.fromarray(base_img[ar_y1:ar_y1+target_h, :]).save(p_crop_aspect)
    paths["crop_aspect"] = p_crop_aspect

    # 7. Nested crop (crop of crop_center)
    crop_c_arr = base_img[y1:y1+c_h, x1:x1+c_w]
    nc_h, nc_w = crop_c_arr.shape[:2]
    p_crop_nested = out_dir / "crop_nested.png"
    Image.fromarray(crop_c_arr[int(nc_h * 0.15):int(nc_h * 0.85), int(nc_w * 0.15):int(nc_w * 0.85)]).save(p_crop_nested)
    paths["crop_nested"] = p_crop_nested

    # 8. Pure brightness (+25%)
    p_bright = out_dir / "bright.png"
    ImageEnhance.Brightness(pil_base).enhance(1.25).save(p_bright)
    paths["brightness"] = p_bright

    # 9. Pure contrast (1.4x)
    p_contrast = out_dir / "contrast.png"
    ImageEnhance.Contrast(pil_base).enhance(1.40).save(p_contrast)
    paths["contrast"] = p_contrast

    # 10. Blur (Gaussian radius=5)
    p_blur = out_dir / "blur.png"
    blur_arr = cv2.GaussianBlur(base_img, (11, 11), 3.0)
    Image.fromarray(blur_arr).save(p_blur)
    paths["blur"] = p_blur

    # 11. Full-canvas resize (downscale to 320x320)
    p_resize = out_dir / "resize.png"
    pil_base.resize((320, 320), Image.Resampling.LANCZOS).save(p_resize)
    paths["resize"] = p_resize

    # 12. JPEG recompression (Q=35)
    p_jpeg = out_dir / "jpeg.jpg"
    pil_base.save(p_jpeg, quality=35)
    paths["jpeg"] = p_jpeg

    # 13. Noise
    p_noise = out_dir / "noise.png"
    rng_n = np.random.RandomState(1234)
    noise = rng_n.normal(0, 15, base_img.shape).astype(np.int16)
    noise_arr = np.clip(base_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(noise_arr).save(p_noise)
    paths["noise"] = p_noise

    # 14. Sharpen
    p_sharpen = out_dir / "sharpen.png"
    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp_arr = np.clip(cv2.filter2D(base_img, -1, kernel_sharp), 0, 255).astype(np.uint8)
    Image.fromarray(sharp_arr).save(p_sharpen)
    paths["sharpen"] = p_sharpen

    # 15. Compound: Crop + Brightness (+25% on crop_center)
    p_crop_bright = out_dir / "crop_bright.png"
    crop_c_pil = Image.fromarray(base_img[y1:y1+c_h, x1:x1+c_w])
    ImageEnhance.Brightness(crop_c_pil).enhance(1.25).save(p_crop_bright)
    paths["compound_crop_bright"] = p_crop_bright

    # 16. Compound: Crop + Blur (blur on crop_center)
    p_crop_blur = out_dir / "crop_blur.png"
    crop_blur_arr = cv2.GaussianBlur(base_img[y1:y1+c_h, x1:x1+c_w], (9, 9), 2.5)
    Image.fromarray(crop_blur_arr).save(p_crop_blur)
    paths["compound_crop_blur"] = p_crop_blur

    # 17. Compound: Resize + Brightness
    p_res_bright = out_dir / "resize_bright.png"
    res_pil = pil_base.resize((320, 320), Image.Resampling.LANCZOS)
    ImageEnhance.Brightness(res_pil).enhance(1.25).save(p_res_bright)
    paths["compound_resize_bright"] = p_res_bright

    return paths


def run_baseline_evaluation():
    manager = ModelManager()
    manager.initialize()

    base_dir = Path("data/dev_isolated_test")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Generate 3 distinct rich scenes and 1 low-texture scene
    scenes = {}
    for s_id in [101, 102, 103]:
        arr = generate_scene(s_id, 512, 512, low_texture=False)
        scenes[f"scene_{s_id}"] = create_derivatives(arr, base_dir / f"scene_{s_id}")

    # Low-texture scene
    lt_arr = generate_scene(999, 512, 512, low_texture=True)
    scenes["scene_low_tex"] = create_derivatives(lt_arr, base_dir / "scene_low_tex")

    # Generate an unrelated independent scene for negative testing
    unrel_arr = generate_scene(777, 512, 512, low_texture=False)
    p_unrelated = base_dir / "unrelated_control.png"
    Image.fromarray(unrel_arr).save(p_unrelated)

    results = []
    print("\n" + "="*80)
    print("BASELINE GENERALIZATION EVALUATION")
    print("="*80)

    for s_name, derivs in scenes.items():
        p_orig = derivs["orig"]
        for key, p_deriv in derivs.items():
            if key == "orig":
                continue

            t0 = time.time()
            pred = manager.predict_pair(p_orig, p_deriv)
            dt = time.time() - t0

            # Determine expected label
            if key.startswith("crop_"):
                expected_tfm = "crop"
                expected_dir = "A_TO_B"
            elif key == "brightness":
                expected_tfm = "brightness"
                expected_dir = "UNCERTAIN"
            elif key == "contrast":
                expected_tfm = "contrast"
                expected_dir = "UNCERTAIN"
            elif key == "blur":
                expected_tfm = "blur"
                expected_dir = "A_TO_B"
            elif key == "resize":
                expected_tfm = "resize"
                expected_dir = "A_TO_B"
            elif key == "jpeg":
                expected_tfm = "jpeg"
                expected_dir = "UNCERTAIN"
            elif key == "noise":
                expected_tfm = "noise"
                expected_dir = "UNCERTAIN"
            elif key == "sharpen":
                expected_tfm = "sharpen"
                expected_dir = "UNCERTAIN"
            elif key == "compound_crop_bright":
                expected_tfm = "crop"  # Primary spatial, compound has brightness
                expected_dir = "A_TO_B"
            elif key == "compound_crop_blur":
                expected_tfm = "crop"  # Primary spatial, compound has blur
                expected_dir = "A_TO_B"
            elif key == "compound_resize_bright":
                expected_tfm = "resize"
                expected_dir = "A_TO_B"
            else:
                expected_tfm = key
                expected_dir = "A_TO_B"

            res_entry = {
                "scene": s_name,
                "variant": key,
                "expected_tfm": expected_tfm,
                "predicted_tfm": pred["transformation"],
                "tfm_conf": pred["transformation_confidence"],
                "expected_dir": expected_dir,
                "predicted_dir": pred["direction"],
                "dir_conf": pred["direction_confidence"],
                "related": pred["related"],
                "rel_conf": pred["related_confidence"],
                "cos_sim": pred["cosine_similarity"],
                "compound_tags": pred.get("compound_tags", []),
                "dt_ms": round(dt * 1000, 1)
            }
            results.append(res_entry)

            status = "MATCH" if (pred["transformation"] == expected_tfm) else "MISMATCH"
            print(f"[{status:8s}] {s_name:14s} | {key:22s} -> Tfm: {pred['transformation']:12s} (exp: {expected_tfm:10s}), Dir: {pred['direction']:10s} (exp: {expected_dir:10s}), P(rel): {pred['related_confidence']:.2f}, CosSim: {pred['cosine_similarity']:.2f}, Time: {res_entry['dt_ms']}ms")

    # Evaluate Unrelated Negative Pairs
    for s_name, derivs in scenes.items():
        p_orig = derivs["orig"]
        t0 = time.time()
        pred_neg = manager.predict_pair(p_orig, p_unrelated)
        dt = time.time() - t0
        status_neg = "MATCH" if (not pred_neg["related"]) else "FALSE_POSITIVE"
        print(f"[{status_neg:8s}] {s_name:14s} | UNRELATED_PAIR        -> Related: {str(pred_neg['related']):5s} (exp: False), Tfm: {pred_neg['transformation']:10s}, CosSim: {pred_neg['cosine_similarity']:.2f}, Time: {round(dt*1000,1)}ms")
        results.append({
            "scene": s_name,
            "variant": "unrelated",
            "expected_tfm": "none",
            "predicted_tfm": pred_neg["transformation"],
            "tfm_conf": pred_neg["transformation_confidence"],
            "expected_dir": "NONE",
            "predicted_dir": pred_neg["direction"],
            "dir_conf": pred_neg["direction_confidence"],
            "related": pred_neg["related"],
            "rel_conf": pred_neg["related_confidence"],
            "cos_sim": pred_neg["cosine_similarity"],
            "compound_tags": pred_neg.get("compound_tags", []),
            "dt_ms": round(dt * 1000, 1)
        })

    # Summary Statistics
    total_pairs = len(results)
    tfm_matches = sum(1 for r in results if r["predicted_tfm"] == r["expected_tfm"])
    crop_total = sum(1 for r in results if r["expected_tfm"] == "crop")
    crop_to_bright = sum(1 for r in results if r["expected_tfm"] == "crop" and r["predicted_tfm"] == "brightness")
    crop_correct = sum(1 for r in results if r["expected_tfm"] == "crop" and r["predicted_tfm"] == "crop")
    bright_total = sum(1 for r in results if r["expected_tfm"] == "brightness")
    bright_correct = sum(1 for r in results if r["expected_tfm"] == "brightness" and r["predicted_tfm"] == "brightness")

    print("\n" + "="*80)
    print("BASELINE SUMMARY METRICS")
    print("="*80)
    print(f"Total Evaluated Pairs: {total_pairs}")
    print(f"Overall Transformation Accuracy: {tfm_matches}/{total_pairs} ({tfm_matches/total_pairs*100:.1f}%)")
    print(f"Crop Accuracy: {crop_correct}/{crop_total} ({crop_correct/max(1, crop_total)*100:.1f}%)")
    print(f"Crop-to-Brightness Misclassifications: {crop_to_bright}/{crop_total} ({crop_to_bright/max(1, crop_total)*100:.1f}%)")
    print(f"Brightness Accuracy: {bright_correct}/{bright_total} ({bright_correct/max(1, bright_total)*100:.1f}%)")
    print(f"Candidate Filter Recall on Parent-Child: {sum(1 for r in results if r['variant'] != 'unrelated' and r['related'])}/{sum(1 for r in results if r['variant'] != 'unrelated')}")


if __name__ == "__main__":
    run_baseline_evaluation()
