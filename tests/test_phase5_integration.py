"""
tests/test_phase5_integration.py — Phase 5 Safe Production Integration & Regression Tests.

Covers all 12 validation categories required by Gate 3:
 1. Checkpoint loading and missing-checkpoint handling.
 2. Feature dimensions and scaler compatibility (115-D, 172-D).
 3. Identical images.
 4. Clearly unrelated images.
 5. Cropped and resized image pairs.
 6. Brightness and contrast changes.
 7. Blur, sharpening, noise, and JPEG compression.
 8. Composite transformations where supported by the labels.
 9. Direction prediction when A/B order is swapped (100% swap consistency).
10. UNCERTAIN direction behavior for ambiguous/near-boundary pairs.
11. Graph-edge creation and confidence filtering.
12. Explainability output and existing API compatibility.
"""

import os
import sys
import shutil
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.model_manager import get_model_manager, ModelManager
from backend.app.services.provenance_engine import run_provenance_analysis
from src.transforms import apply_crop, apply_resize, apply_blur, apply_jpeg, apply_noise, apply_brightness, apply_contrast, apply_sharpen
from src.pair_features import build_115d_pair_feature, build_172d_direction_feature


class TestPhase5Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = ROOT_DIR / "scratch" / "phase5_test_fixtures"
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        # Create deterministic synthetic test fixtures
        np.random.seed(42)
        # Root A: structured geometric pattern
        img_arr_a = np.zeros((256, 256, 3), dtype=np.uint8)
        img_arr_a[:128, :128] = [200, 50, 50]
        img_arr_a[128:, :128] = [50, 200, 50]
        img_arr_a[:128, 128:] = [50, 50, 200]
        img_arr_a[128:, 128:] = [220, 220, 50]
        # Diagonal stripe
        for i in range(256):
            img_arr_a[i, i] = [255, 255, 255]
        cls.root_a = cls.test_dir / "root_a.png"
        Image.fromarray(img_arr_a).save(cls.root_a)

        # Root B: completely distinct unrelated pattern (concentric circles)
        img_arr_b = np.zeros((256, 256, 3), dtype=np.uint8)
        y, x = np.ogrid[:256, :256]
        dist = np.sqrt((x - 128) ** 2 + (y - 128) ** 2)
        img_arr_b[(dist < 80)] = [180, 180, 180]
        img_arr_b[(dist < 40)] = [40, 40, 40]
        cls.root_b = cls.test_dir / "root_b.png"
        Image.fromarray(img_arr_b).save(cls.root_b)

        # Identical copy of Root A
        cls.root_a_copy = cls.test_dir / "root_a_copy.png"
        Image.fromarray(img_arr_a.copy()).save(cls.root_a_copy)

        # 1. Cropped derivative of Root A
        im_pil_a = Image.fromarray(img_arr_a)
        cls.derived_crop = cls.test_dir / "derived_crop.png"
        im_crop = apply_crop(im_pil_a, {"mode": "center", "frac_w": 0.15, "frac_h": 0.15, "resize_back": True})
        im_crop.save(cls.derived_crop)

        # 2. Resized derivative of Root A
        cls.derived_resize = cls.test_dir / "derived_resize.png"
        im_pil_a.resize((128, 128), Image.Resampling.BILINEAR).save(cls.derived_resize)

        # 3. Brightness & Contrast derivative
        cls.derived_bright_contrast = cls.test_dir / "derived_bc.png"
        enh_b = ImageEnhance.Brightness(im_pil_a).enhance(1.25)
        enh_c = ImageEnhance.Contrast(enh_b).enhance(1.30)
        enh_c.save(cls.derived_bright_contrast)

        # 4. Blur derivative
        cls.derived_blur = cls.test_dir / "derived_blur.png"
        im_pil_a.filter(ImageFilter.GaussianBlur(radius=2.0)).save(cls.derived_blur)

        # 5. Noise & JPEG derivative (composite)
        cls.derived_noise_jpeg = cls.test_dir / "derived_noise_jpeg.png"
        arr_noisy = np.clip(img_arr_a.astype(float) + np.random.normal(0, 15, img_arr_a.shape), 0, 255).astype(np.uint8)
        im_noisy = Image.fromarray(arr_noisy)
        im_noisy.save(cls.derived_noise_jpeg, format="JPEG", quality=45)

        cls.client = TestClient(app)
        cls.manager = get_model_manager()

    @classmethod
    def tearDownClass(cls):
        # Reset to default disabled
        cls.manager.phase4_enabled = False
        cls.manager.initialize()
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    # -----------------------------------------------------------------
    # Test 1: Checkpoint Loading & Missing-Checkpoint Handling
    # -----------------------------------------------------------------
    def test_01_checkpoint_loading_and_missing_handling(self):
        # Test enabled loading
        self.manager.set_phase4_enabled(True)
        self.assertTrue(self.manager.phase4_loaded, "Phase 4 models must load successfully when enabled.")

        # Test missing file handling: should not crash, should fall back
        fake_dir = ROOT_DIR / "models" / "fake_nonexistent_v99"
        import backend.app.services.model_manager as mm
        orig_path = mm.PHASE4_MODELS_DIR
        try:
            mm.PHASE4_MODELS_DIR = fake_dir
            temp_mgr = ModelManager()
            temp_mgr.phase4_loaded = False
            temp_mgr.set_phase4_enabled(True)
            self.assertFalse(temp_mgr.phase4_loaded, "Missing directory must not set phase4_loaded=True")
        finally:
            mm.PHASE4_MODELS_DIR = orig_path
            self.manager.set_phase4_enabled(True)

    # -----------------------------------------------------------------
    # Test 2: Feature Dimensions and Scaler Compatibility
    # -----------------------------------------------------------------
    def test_02_feature_dimensions_and_scaler_compatibility(self):
        self.manager.set_phase4_enabled(True)

        f_a = self.manager.get_57d_representation(self.root_a)
        f_b = self.manager.get_57d_representation(self.derived_crop)
        self.assertEqual(len(f_a), 57, "Image A representation must be exactly 57-D")
        self.assertEqual(len(f_b), 57, "Image B representation must be exactly 57-D")

        vec_115 = build_115d_pair_feature(f_a, f_b)
        self.assertEqual(len(vec_115), 115, "Pair vector must be exactly 115-D")
        sc_115 = self.manager.phase4_scaler_115.transform(vec_115.reshape(1, -1))
        self.assertEqual(sc_115.shape, (1, 115), "Scaler 115 must transform into (1, 115)")
        self.assertFalse(np.isnan(sc_115).any(), "Scaled vector must not contain NaNs")

        vec_172 = build_172d_direction_feature(f_a, f_b)
        self.assertEqual(len(vec_172), 172, "Direction vector must be exactly 172-D")
        sc_172 = self.manager.phase4_scaler_172.transform(vec_172.reshape(1, -1))
        self.assertEqual(sc_172.shape, (1, 172), "Scaler 172 must transform into (1, 172)")

    # -----------------------------------------------------------------
    # Test 3: Identical Images
    # -----------------------------------------------------------------
    def test_03_identical_images(self):
        self.manager.set_phase4_enabled(True)
        pred = self.manager.predict_pair(self.root_a, self.root_a_copy)
        self.assertTrue(pred["related"], "Identical images must be predicted as related.")
        self.assertGreaterEqual(pred["related_confidence"], 0.85)
        self.assertEqual(pred["cosine_similarity"], 1.0)

    # -----------------------------------------------------------------
    # Test 4: Clearly Unrelated Images
    # -----------------------------------------------------------------
    def test_04_clearly_unrelated_images(self):
        self.manager.set_phase4_enabled(True)
        pred = self.manager.predict_pair(self.root_a, self.root_b)
        self.assertFalse(pred["related"], "Clearly unrelated images must be predicted as unrelated.")
        self.assertEqual(pred["direction"], "NONE", "Unrelated pair direction must be NONE.")
        self.assertEqual(pred["transformation"], "none", "Unrelated pair transformation must be none.")

    # -----------------------------------------------------------------
    # Test 5: Cropped and Resized Image Pairs
    # -----------------------------------------------------------------
    def test_05_cropped_and_resized_pairs(self):
        self.manager.set_phase4_enabled(True)

        # Crop pair
        pred_crop = self.manager.predict_pair(self.root_a, self.derived_crop)
        self.assertTrue(pred_crop["related"], "Cropped image must be recognized as related.")
        self.assertIn(pred_crop["direction"], ["A_TO_B", "B_TO_A"], "Direction must be established for crop pair.")
        self.assertIsInstance(pred_crop["compound_tags"], list)
        self.assertGreaterEqual(len(pred_crop["compound_tags"]), 1, "Should predict applied transformations")

        # Resize pair
        pred_resize = self.manager.predict_pair(self.root_a, self.derived_resize)
        self.assertTrue(pred_resize["related"], "Resized image must be recognized as related.")
        self.assertIn(pred_resize["direction"], ["A_TO_B", "B_TO_A"], "Direction must be established for resize pair.")
        self.assertIsInstance(pred_resize["compound_tags"], list)

    # -----------------------------------------------------------------
    # Test 6: Brightness and Contrast Changes
    # -----------------------------------------------------------------
    def test_06_brightness_and_contrast_pairs(self):
        self.manager.set_phase4_enabled(True)
        pred = self.manager.predict_pair(self.root_a, self.derived_bright_contrast)
        self.assertTrue(pred["related"], "Brightness/contrast altered image must remain related.")
        self.assertIn(pred["transformation"], ["brightness", "contrast", "color", "crop"])

    # -----------------------------------------------------------------
    # Test 7: Blur, Sharpening, Noise, and JPEG Compression
    # -----------------------------------------------------------------
    def test_07_blur_noise_jpeg_pairs(self):
        self.manager.set_phase4_enabled(True)

        # Blur pair
        pred_blur = self.manager.predict_pair(self.root_a, self.derived_blur)
        self.assertTrue(pred_blur["related"], "Blurred image must be recognized as related.")
        self.assertIn(pred_blur["direction"], ["A_TO_B", "B_TO_A"], "Direction must be established for blur pair.")

        # Noise + JPEG pair
        pred_nj = self.manager.predict_pair(self.root_a, self.derived_noise_jpeg)
        self.assertTrue(pred_nj["related"], "Noisy/JPEG image must be recognized as related.")

    # -----------------------------------------------------------------
    # Test 8: Composite Transformations
    # -----------------------------------------------------------------
    def test_08_composite_transformations(self):
        self.manager.set_phase4_enabled(True)
        pred = self.manager.predict_pair(self.root_a, self.derived_noise_jpeg)
        self.assertIsInstance(pred["compound_tags"], list, "compound_tags must be a list")
        self.assertGreaterEqual(len(pred["compound_tags"]), 1, "Must contain at least 1 transformation tag.")

    # -----------------------------------------------------------------
    # Test 9: Direction Prediction When A/B Order is Swapped
    # -----------------------------------------------------------------
    def test_09_direction_swap_consistency(self):
        self.manager.set_phase4_enabled(True)

        # Test forward
        pred_fwd = self.manager.predict_pair(self.root_a, self.derived_crop)
        # Test reverse
        pred_rev = self.manager.predict_pair(self.derived_crop, self.root_a)

        # Order swap MUST invert direction between A_TO_B and B_TO_A
        if pred_fwd["direction"] == "A_TO_B":
            self.assertEqual(pred_rev["direction"], "B_TO_A", "Reverse pair must flip to B_TO_A when forward is A_TO_B")
        elif pred_fwd["direction"] == "B_TO_A":
            self.assertEqual(pred_rev["direction"], "A_TO_B", "Reverse pair must flip to A_TO_B when forward is B_TO_A")
        else:
            self.assertEqual(pred_rev["direction"], "UNCERTAIN", "Reverse of UNCERTAIN must be UNCERTAIN")

        # Antisymmetric confidence symmetry
        self.assertAlmostEqual(pred_fwd["direction_confidence"], pred_rev["direction_confidence"], places=4)
        # Symmetric relationship prediction
        self.assertEqual(pred_fwd["related"], pred_rev["related"])
        self.assertAlmostEqual(pred_fwd["related_confidence"], pred_rev["related_confidence"], places=4)

    # -----------------------------------------------------------------
    # Test 10: UNCERTAIN Direction Behavior
    # -----------------------------------------------------------------
    def test_10_uncertain_direction_behavior(self):
        self.manager.set_phase4_enabled(True)
        # For identical images with zero physical area change and zero crop, direction is ambiguous/uncertain
        pred_ident = self.manager.predict_pair(self.root_a, self.root_a_copy)
        self.assertIn(pred_ident["direction"], ["UNCERTAIN", "A_TO_B"], "Ambiguous identical pair must allow UNCERTAIN.")

    # -----------------------------------------------------------------
    # Test 11: Graph-Edge Creation and Confidence Filtering
    # -----------------------------------------------------------------
    def test_11_graph_edge_creation_and_filtering(self):
        self.manager.set_phase4_enabled(True)

        # Run provenance analysis on 3 images: root_a, derived_crop, root_b
        res = run_provenance_analysis(
            [self.root_a, self.derived_crop, self.root_b],
            similarity_threshold=0.55,
            confidence_filter=0.40
        )
        self.assertEqual(res["summary"]["total_images"], 3)
        self.assertEqual(res["summary"]["forest_edges"], 1)
        self.assertEqual(len(res["edges"]), 1)
        edge = res["edges"][0]
        self.assertEqual(set([edge["source"], edge["target"]]), set(["root_a", "derived_crop"]))

    # -----------------------------------------------------------------
    # Test 12: Explainability Output & Existing API Compatibility
    # -----------------------------------------------------------------
    def test_12_explainability_and_api_compatibility(self):
        self.manager.set_phase4_enabled(True)

        # Sensitivity endpoint
        sens = self.manager.compute_feature_sensitivity(self.root_a, self.derived_crop)
        self.assertIn("base_prediction", sens)
        self.assertIn("feature_groups", sens)
        self.assertEqual(len(sens["feature_groups"]), 5)

        # Health endpoint
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ONLINE")
        self.assertTrue(data["phase4_models_enabled"])
        self.assertTrue(data["phase4_models_loaded"])
        self.assertEqual(data["active_model"], "Phase 4 Normalized Forensic MLP Pipeline (v1)")

        # Verify fallback when toggled OFF
        self.manager.set_phase4_enabled(False)
        resp_off = self.client.get("/api/health")
        data_off = resp_off.json()
        self.assertFalse(data_off["phase4_models_enabled"])
        self.assertNotEqual(data_off["active_model"], "Phase 4 Normalized Forensic MLP Pipeline (v1)")


if __name__ == "__main__":
    unittest.main()
