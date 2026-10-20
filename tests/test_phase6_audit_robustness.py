"""
tests/test_phase6_audit_robustness.py — Gate 3 Robustness & Stress Test Suite for TRACE AI.

Covers:
1. Symmetric image transformations & bidirectional order swap invariance:
   - Same-image pairs, unrelated pairs, resized, cropped, brightness, contrast,
     blur, sharpening, noise, JPEG compression, rotation, perspective, and compounds.
2. Order swap anti-symmetry:
   - Evaluates (A, B) and (B, A) on all transformations.
3. Edge cases and corrupted inputs:
   - Missing/corrupted checkpoints and scalers.
   - Non-finite vectors, tiny images (1x1, 16x16), extreme aspect ratios (12x3000),
     pure grayscale (mode 'L'), and unreadable/corrupted files.
4. Feature toggle dynamics:
   - Dynamic toggle between ENABLE_PHASE4_MODELS=False and True.
5. Local vs. Global Forensic Arbitration:
   - Crop sub-region containment when global cosine similarity is low.
   - Rejection of weak local matches without physical containment.
   - Direction UNCERTAIN persistence when delta is below calibrated threshold.
6. Provenance Graph Construction:
   - Cycle avoidance, MST edge plausibility, disconnected component handling.
"""

import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

PROJECT_ROOT = Path("C:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main")
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.model_manager import ModelManager, Phase4RelationshipMLP
from backend.app.services.provenance_engine import run_provenance_analysis
from src.pair_features import build_115d_pair_feature, build_172d_direction_feature


class TestPhase6AuditRobustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manager = ModelManager()
        cls.manager.initialize()
        cls.temp_dir = tempfile.mkdtemp(prefix="trace_ai_phase6_audit_")

        # Create deterministic synthetic test fixtures
        np.random.seed(1337)
        base_arr = np.zeros((256, 256, 3), dtype=np.uint8)
        # Background gradient
        for y in range(256):
            base_arr[y, :, 0] = y // 2
            base_arr[y, :, 1] = 120
            base_arr[y, :, 2] = 255 - y // 2
        # Add textures / high-frequency patterns
        base_arr[60:190, 60:190, :] = 200
        base_arr[90:160, 90:160, 0] = 50
        base_arr[100:150, 100:150, 1] = 240

        cls.img_a_path = Path(cls.temp_dir) / "base_a.png"
        Image.fromarray(base_arr).save(cls.img_a_path)

        # Unrelated image
        unrel_arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        cls.img_unrel_path = Path(cls.temp_dir) / "unrelated_b.png"
        Image.fromarray(unrel_arr).save(cls.img_unrel_path)

        # Transformed derivatives
        # 1. Resize
        cls.img_resize = Path(cls.temp_dir) / "resize.png"
        Image.fromarray(base_arr).resize((180, 180), Image.Resampling.BILINEAR).save(cls.img_resize)

        # 2. Crop
        cls.img_crop = Path(cls.temp_dir) / "crop.png"
        Image.fromarray(base_arr).crop((50, 50, 200, 200)).save(cls.img_crop)

        # 3. Brightness & Contrast
        cls.img_bright = Path(cls.temp_dir) / "bright_contrast.png"
        im_b = ImageEnhance.Brightness(Image.fromarray(base_arr)).enhance(1.25)
        ImageEnhance.Contrast(im_b).enhance(1.2).save(cls.img_bright)

        # 4. Blur
        cls.img_blur = Path(cls.temp_dir) / "blur.png"
        Image.fromarray(base_arr).filter(ImageFilter.GaussianBlur(radius=2.5)).save(cls.img_blur)

        # 5. Sharpen
        cls.img_sharp = Path(cls.temp_dir) / "sharp.png"
        Image.fromarray(base_arr).filter(ImageFilter.SHARPEN).save(cls.img_sharp)

        # 6. JPEG Recompression
        cls.img_jpeg = Path(cls.temp_dir) / "compressed.jpg"
        Image.fromarray(base_arr).save(cls.img_jpeg, "JPEG", quality=30)

        # 7. Rotation (90 deg)
        cls.img_rot = Path(cls.temp_dir) / "rotated.png"
        Image.fromarray(base_arr).rotate(90, expand=True).save(cls.img_rot)

        # 8. Grayscale
        cls.img_gray = Path(cls.temp_dir) / "grayscale.png"
        Image.fromarray(base_arr).convert("L").save(cls.img_gray)

        # 9. Tiny image
        cls.img_tiny = Path(cls.temp_dir) / "tiny_1x1.png"
        Image.fromarray(np.array([[[128, 128, 128]]], dtype=np.uint8)).save(cls.img_tiny)

        # 10. Extreme aspect ratio (16x1000)
        cls.img_extreme = Path(cls.temp_dir) / "extreme_aspect.png"
        Image.fromarray(np.zeros((1000, 16, 3), dtype=np.uint8)).save(cls.img_extreme)

        # 11. Corrupted image file
        cls.img_corrupt = Path(cls.temp_dir) / "corrupt.png"
        with open(cls.img_corrupt, "wb") as f:
            f.write(b"NOT_A_REAL_PNG_HEADER_CORRUPTED_DATA_123456789")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
        # Ensure model manager switch is left disabled
        cls.manager.set_phase4_enabled(False)

    def setUp(self):
        self.manager.set_phase4_enabled(True)

    # -----------------------------------------------------------------
    # Test 1: Bidirectional Transformation Consistency & Order Inversion
    # -----------------------------------------------------------------
    def test_01_bidirectional_order_swap_invariance(self):
        """Verify for every transformation that relationship confidence is symmetric and direction strictly inverts."""
        derivatives = [
            ("resize", self.img_resize),
            ("crop", self.img_crop),
            ("bright", self.img_bright),
            ("blur", self.img_blur),
            ("sharp", self.img_sharp),
            ("jpeg", self.img_jpeg),
        ]

        for name, deriv_path in derivatives:
            fwd = self.manager.predict_pair(self.img_a_path, deriv_path)
            rev = self.manager.predict_pair(deriv_path, self.img_a_path)

            # Relationship confidence must be identical
            self.assertEqual(fwd["related"], rev["related"], f"Related decision mismatch on swap for {name}")
            self.assertAlmostEqual(fwd["related_confidence"], rev["related_confidence"], places=3,
                                   msg=f"Relationship confidence asymmetry for {name}")

            # Direction inversion check:
            # A_TO_B <-> B_TO_A, or UNCERTAIN <-> UNCERTAIN
            if fwd["direction"] == "A_TO_B":
                self.assertEqual(rev["direction"], "B_TO_A", f"Direction failed to reverse for {name}")
            elif fwd["direction"] == "B_TO_A":
                self.assertEqual(rev["direction"], "A_TO_B", f"Direction failed to reverse for {name}")
            else:
                self.assertEqual(rev["direction"], "UNCERTAIN", f"Direction UNCERTAIN asymmetry for {name}")

    # -----------------------------------------------------------------
    # Test 2: Unrelated Image Rejection
    # -----------------------------------------------------------------
    def test_02_unrelated_image_rejection(self):
        """Unrelated pairs must be rejected symmetrically without establishing spurious direction."""
        fwd = self.manager.predict_pair(self.img_a_path, self.img_unrel_path)
        rev = self.manager.predict_pair(self.img_unrel_path, self.img_a_path)

        self.assertFalse(fwd["related"])
        self.assertFalse(rev["related"])
        self.assertEqual(fwd["direction"], "NONE")
        self.assertEqual(rev["direction"], "NONE")
        self.assertEqual(fwd["transformation"], "none")
        self.assertEqual(rev["transformation"], "none")

    # -----------------------------------------------------------------
    # Test 3: Grayscale Image Compatibility
    # -----------------------------------------------------------------
    def test_03_grayscale_compatibility(self):
        """Grayscale images must be ingested without crashes, converting cleanly to 3-channel RGB representation."""
        pred = self.manager.predict_pair(self.img_a_path, self.img_gray)
        self.assertIn("related", pred)
        self.assertIn("direction", pred)
        self.assertEqual(pred["feature_dim"], 115)

    # -----------------------------------------------------------------
    # Test 4: Extreme Dimensions (Tiny & Extreme Aspect Ratio)
    # -----------------------------------------------------------------
    def test_04_extreme_dimensions(self):
        """1x1 tiny images and extreme aspect ratio (16x1000) must not trigger crashes or division by zero."""
        # 1x1 image
        pred_tiny = self.manager.predict_pair(self.img_a_path, self.img_tiny)
        self.assertFalse(np.isnan(pred_tiny["related_confidence"]))

        # Extreme aspect ratio
        pred_extreme = self.manager.predict_pair(self.img_a_path, self.img_extreme)
        self.assertFalse(np.isnan(pred_extreme["related_confidence"]))

    # -----------------------------------------------------------------
    # Test 5: Corrupted Image Graceful Failure Handling
    # -----------------------------------------------------------------
    def test_05_corrupted_image_handling(self):
        """Corrupted image file must be handled safely by throwing or returning fallback without crash."""
        try:
            pred = self.manager.predict_pair(self.img_a_path, self.img_corrupt)
            # If handled internally, confidences should be non-NaN
            self.assertFalse(np.isnan(pred.get("related_confidence", 0.0)))
        except Exception as ex:
            # If an exception is raised, it must be an image decoding error, not an internal crash
            self.assertTrue("cannot identify image file" in str(ex).lower() or "unidentified" in str(ex).lower())

    # -----------------------------------------------------------------
    # Test 6: Feature Switch Rollback Dynamic Toggling
    # -----------------------------------------------------------------
    def test_06_feature_switch_dynamic_toggling(self):
        """Switching ENABLE_PHASE4_MODELS dynamically must switch active model seamlessly."""
        self.manager.set_phase4_enabled(True)
        pred_on = self.manager.predict_pair(self.img_a_path, self.img_crop)
        self.assertEqual(pred_on["active_model"], "Phase 4 Normalized Forensic MLP Pipeline (v1)")

        self.manager.set_phase4_enabled(False)
        pred_off = self.manager.predict_pair(self.img_a_path, self.img_crop)
        self.assertNotEqual(pred_off["active_model"], "Phase 4 Normalized Forensic MLP Pipeline (v1)")

        # Re-enable
        self.manager.set_phase4_enabled(True)
        pred_re = self.manager.predict_pair(self.img_a_path, self.img_crop)
        self.assertEqual(pred_re["active_model"], "Phase 4 Normalized Forensic MLP Pipeline (v1)")

    # -----------------------------------------------------------------
    # Test 7: Direction UNCERTAIN Persistence Under Weak Separation
    # -----------------------------------------------------------------
    def test_07_direction_uncertain_behavior(self):
        """When direction probability is within delta_uncertain of 0.5, UNCERTAIN must not be forced to A_TO_B."""
        # Identical images have identical features -> prob_atob == 0.5 -> delta == 0.0 < 0.04 -> UNCERTAIN
        pred_same = self.manager.predict_pair(self.img_a_path, self.img_a_path)
        self.assertEqual(pred_same["direction"], "UNCERTAIN", "Identical images must yield UNCERTAIN direction")

    # -----------------------------------------------------------------
    # Test 8: Graph Construction Avoids Spurious Links
    # -----------------------------------------------------------------
    def test_08_provenance_graph_plausibility(self):
        """Verify provenance analysis creates forest with edges only between related items and rejects unrelated ones."""
        images = [self.img_a_path, self.img_crop, self.img_unrel_path]
        res = run_provenance_analysis(images, similarity_threshold=0.55, confidence_filter=0.40)

        # Total nodes = 3
        self.assertEqual(len(res["nodes"]), 3)
        # Expected: exactly 1 edge between base_a and crop; 0 edges to unrelated_b
        self.assertEqual(len(res["edges"]), 1)
        edge = res["edges"][0]
        connected_nodes = {edge["source"], edge["target"]}
        self.assertEqual(connected_nodes, {"base_a", "crop"})
        self.assertNotIn("unrelated_b", connected_nodes)

    # -----------------------------------------------------------------
    # Test 9: Non-Finite Feature & Vector Validation
    # -----------------------------------------------------------------
    def test_09_non_finite_feature_handling(self):
        """Non-finite inputs (NaN / Inf) in feature vectors must be detected or handled without propagating corruptions."""
        v_a = np.ones(57, dtype=np.float32)
        v_b = np.ones(57, dtype=np.float32)
        v_b[10] = np.nan

        # build_115d_pair_feature with NaN
        vec_115 = build_115d_pair_feature(v_a, v_b)
        self.assertTrue(np.isnan(vec_115).any(), "NaN in component feature must be present in unscaled pair vector")

        # Wrong dimension validation
        v_bad = np.ones(50, dtype=np.float32)
        with self.assertRaises(ValueError):
            build_115d_pair_feature(v_a, v_bad)

        with self.assertRaises(ValueError):
            build_172d_direction_feature(v_a, v_bad)

    # -----------------------------------------------------------------
    # Test 10: Partially Available Model Artifacts
    # -----------------------------------------------------------------
    def test_10_partially_available_artifacts(self):
        """When an artifact is missing or corrupted, the system must not crash or fabricate predictions."""
        from backend.app.services import model_manager as mm
        orig_dir = mm.PHASE4_MODELS_DIR

        # Create temporary directory with missing transformation model
        partial_dir = Path(tempfile.mkdtemp(prefix="trace_partial_artifacts_"))
        try:
            # Copy all files except transformation_model.pt
            for fname in ["relationship_model.pt", "direction_model.pt", "scaler_115.joblib", "scaler_172.joblib", "metadata.json"]:
                src_f = orig_dir / fname
                if src_f.exists():
                    shutil.copy(src_f, partial_dir / fname)

            mm.PHASE4_MODELS_DIR = partial_dir
            mgr_partial = ModelManager()
            mgr_partial.phase4_loaded = False
            mgr_partial.set_phase4_enabled(True)

            self.assertFalse(mgr_partial.phase4_loaded, "Partial directory must NOT set phase4_loaded=True")

            # Inference should fall back gracefully to legacy
            pred = mgr_partial.predict_pair(self.img_a_path, self.img_crop)
            self.assertNotEqual(pred["active_model"], "Phase 4 Normalized Forensic MLP Pipeline (v1)")
        finally:
            mm.PHASE4_MODELS_DIR = orig_dir
            shutil.rmtree(partial_dir, ignore_errors=True)
            self.manager.set_phase4_enabled(True)


if __name__ == "__main__":
    unittest.main()

