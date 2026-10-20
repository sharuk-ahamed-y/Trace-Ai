"""
tests/test_phase7_metrics_and_inference.py — Phase 7 Metrics Integrity & Inference Transparency Test Suite

Verifies:
1. Phase 4 disabled: legacy inference remains functional and is identified accurately.
2. Phase 4 enabled with valid checkpoints: the correct models and scalers are used.
3. Missing or corrupt metrics metadata: metrics are marked unavailable without crashing.
4. Missing or corrupt checkpoints: fallback behavior is explicit and safe.
5. Partial checkpoints: no unintentional mixture of model versions.
6. API schema compatibility for /api/health, /api/model-info, and /api/model-metrics.
7. Model identity agreement across health, model-info, and prediction results.
8. Repeated requests do not retain stale model identity or stale metrics after a supported configuration change.
"""

import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

PROJECT_ROOT = Path("C:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main")
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app
from backend.app.services.model_manager import ModelManager
import backend.app.api.endpoints as endpoints_mod


class TestPhase7MetricsAndInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.manager = ModelManager()
        cls.manager.initialize()
        cls.temp_dir = tempfile.mkdtemp(prefix="trace_ai_phase7_")

        # Create 2 simple test images
        arr_a = np.full((128, 128, 3), 100, dtype=np.uint8)
        arr_b = np.full((128, 128, 3), 150, dtype=np.uint8)
        cls.path_a = Path(cls.temp_dir) / "img_a.png"
        cls.path_b = Path(cls.temp_dir) / "img_b.png"
        Image.fromarray(arr_a).save(cls.path_a)
        Image.fromarray(arr_b).save(cls.path_b)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
        cls.manager.set_phase4_enabled(False)

    def tearDown(self):
        # Reset any temporary monkey-patching
        self.manager.set_phase4_enabled(False)

    # -----------------------------------------------------------------
    # Test 1: Phase 4 Disabled — Legacy Inference Functional & Accurately Identified
    # -----------------------------------------------------------------
    def test_01_phase4_disabled_legacy_identity(self):
        self.manager.set_phase4_enabled(False)
        pred = self.manager.predict_pair(self.path_a, self.path_b)

        self.assertFalse(self.manager.phase4_enabled)
        self.assertNotEqual(pred["active_model"], "Phase 4 Normalized Forensic MLP Pipeline (v1)")
        self.assertIn("inference_mode", pred)
        self.assertIn("confidence_calibration", pred)
        self.assertIn("fallback_reason", pred)

        # Health endpoint check
        r_health = self.client.get("/api/health").json()
        self.assertFalse(r_health["phase4_models_enabled"])
        self.assertEqual(r_health["active_model"], pred["active_model"])

        # Model metrics check
        r_metrics = self.client.get("/api/model-metrics").json()
        self.assertFalse(r_metrics["phase4_enabled"])
        self.assertEqual(r_metrics["selected_relationship_model"]["test_split_metrics"]["status"], "LEGACY_BASELINE_METRICS")
        self.assertEqual(r_metrics["selected_relationship_model"]["real_world_metrics"]["status"], "NOT_VALIDATED")

    # -----------------------------------------------------------------
    # Test 2: Phase 4 Enabled — Correct Models, Scalers, and Calibrated Metadata
    # -----------------------------------------------------------------
    def test_02_phase4_enabled_model_and_metrics(self):
        self.manager.set_phase4_enabled(True)
        self.assertTrue(self.manager.phase4_enabled)
        self.assertTrue(self.manager.phase4_loaded)

        pred = self.manager.predict_pair(self.path_a, self.path_b)
        self.assertEqual(pred["active_model"], "Phase 4 Normalized Forensic MLP Pipeline (v1)")
        self.assertEqual(pred["inference_mode"], "trained_model")
        self.assertEqual(pred["confidence_calibration"], "uncalibrated_sigmoid")
        self.assertIsNone(pred["fallback_reason"])

        # Health check
        r_health = self.client.get("/api/health").json()
        self.assertTrue(r_health["phase4_models_enabled"])
        self.assertTrue(r_health["phase4_models_loaded"])
        self.assertEqual(r_health["active_model"], pred["active_model"])

        # Metrics check
        r_metrics = self.client.get("/api/model-metrics").json()
        self.assertTrue(r_metrics["phase4_enabled"])
        rel_test = r_metrics["selected_relationship_model"]["test_split_metrics"]
        self.assertEqual(rel_test["dataset_type"], "SYNTHETIC_HELD_OUT_TEST_SIPGD_V2")
        self.assertEqual(rel_test["evaluation_samples"], 324)
        self.assertAlmostEqual(rel_test["precision"], 0.9877, places=3)
        self.assertEqual(rel_test["status"], "VERIFIED_SAVED_EVALUATION")
        self.assertEqual(r_metrics["selected_relationship_model"]["real_world_metrics"]["status"], "NOT_VALIDATED")

    # -----------------------------------------------------------------
    # Test 3: Missing Metadata Handling — Marked Unavailable Without Crashing
    # -----------------------------------------------------------------
    def test_03_missing_metadata_marked_unavailable(self):
        self.manager.set_phase4_enabled(True)
        # Temporarily clear metadata and point to dummy path
        orig_meta = self.manager.phase4_metadata
        orig_dir = endpoints_mod.PHASE4_MODELS_DIR
        try:
            self.manager.phase4_metadata = None
            endpoints_mod.PHASE4_MODELS_DIR = Path("fake_nonexistent_metadata_path")

            r_metrics = self.client.get("/api/model-metrics").json()
            self.assertEqual(r_metrics["selected_relationship_model"]["status"], "METRICS_UNAVAILABLE")
            self.assertIsNone(r_metrics["selected_relationship_model"]["test_split_metrics"])
            self.assertEqual(r_metrics["selected_relationship_model"]["real_world_metrics"]["status"], "NOT_VALIDATED")
        finally:
            self.manager.phase4_metadata = orig_meta
            endpoints_mod.PHASE4_MODELS_DIR = orig_dir

    # -----------------------------------------------------------------
    # Test 4: Missing or Corrupt Checkpoints — Explicit & Safe Fallback
    # -----------------------------------------------------------------
    def test_04_missing_checkpoint_safe_fallback(self):
        fake_path = Path("fake_checkpoint_dir_for_test")
        orig_dir = self.manager.PHASE4_MODELS_DIR if hasattr(self.manager, "PHASE4_MODELS_DIR") else None
        from backend.app.services import model_manager as mm_mod
        orig_mod_dir = mm_mod.PHASE4_MODELS_DIR

        try:
            mm_mod.PHASE4_MODELS_DIR = fake_path
            self.manager.phase4_loaded = False
            self.manager.set_phase4_enabled(True)

            self.assertFalse(self.manager.phase4_loaded)
            self.assertIsNotNone(self.manager.phase4_fallback_reason)

            # Prediction falls back cleanly
            pred = self.manager.predict_pair(self.path_a, self.path_b)
            self.assertNotEqual(pred["active_model"], "Phase 4 Normalized Forensic MLP Pipeline (v1)")
            self.assertIn("fallback_reason", pred)
            self.assertIsNotNone(pred["fallback_reason"])
        finally:
            mm_mod.PHASE4_MODELS_DIR = orig_mod_dir
            self.manager.set_phase4_enabled(False)

    # -----------------------------------------------------------------
    # Test 5: Partial Checkpoints — No Unintentional Version Mixture
    # -----------------------------------------------------------------
    def test_05_partial_checkpoints_prevention_of_version_mixture(self):
        from backend.app.services import model_manager as mm_mod
        orig_dir = mm_mod.PHASE4_MODELS_DIR

        partial_dir = Path(tempfile.mkdtemp(prefix="trace_partial_chk_"))
        try:
            # Copy only relationship_model.pt and omit transformation_model.pt
            shutil.copy(orig_dir / "relationship_model.pt", partial_dir / "relationship_model.pt")
            shutil.copy(orig_dir / "scaler_115.joblib", partial_dir / "scaler_115.joblib")

            mm_mod.PHASE4_MODELS_DIR = partial_dir
            self.manager.phase4_loaded = False
            self.manager.set_phase4_enabled(True)

            # Must NOT be loaded if ANY required component is missing
            self.assertFalse(self.manager.phase4_loaded)
            self.assertIn("Missing required Phase 4 artifact", self.manager.phase4_fallback_reason)

            pred = self.manager.predict_pair(self.path_a, self.path_b)
            # Ensure output is purely legacy, not mixed
            self.assertNotEqual(pred["active_model"], "Phase 4 Normalized Forensic MLP Pipeline (v1)")
        finally:
            mm_mod.PHASE4_MODELS_DIR = orig_dir
            shutil.rmtree(partial_dir, ignore_errors=True)
            self.manager.set_phase4_enabled(False)

    # -----------------------------------------------------------------
    # Test 6: API Schema Compatibility for /api/health, /api/model-info, /api/model-metrics
    # -----------------------------------------------------------------
    def test_06_api_schema_compatibility(self):
        for mode in [False, True]:
            self.manager.set_phase4_enabled(mode)

            # 1. /api/health
            r_h = self.client.get("/api/health")
            self.assertEqual(r_h.status_code, 200)
            data_h = r_h.json()
            for key in ["status", "timestamp", "gpu_available", "device", "phase4_models_enabled", "phase4_models_loaded", "active_model"]:
                self.assertIn(key, data_h)

            # 2. /api/model-info
            r_i = self.client.get("/api/model-info")
            self.assertEqual(r_i.status_code, 200)
            data_i = r_i.json()
            for key in ["embedder", "active_relationship_model", "scikit_learn_pipeline", "pytorch_classifiers", "phase4_models", "supported_transformations"]:
                self.assertIn(key, data_i)

            # 3. /api/model-metrics
            r_m = self.client.get("/api/model-metrics")
            self.assertEqual(r_m.status_code, 200)
            data_m = r_m.json()
            for key in ["active_inference_path", "phase4_enabled", "selected_relationship_model", "transformations_evaluated"]:
                self.assertIn(key, data_m)

    # -----------------------------------------------------------------
    # Test 7: Model Identity Agreement Across Health, Info, and Predictions
    # -----------------------------------------------------------------
    def test_07_model_identity_agreement_across_endpoints(self):
        for mode in [True, False]:
            self.manager.set_phase4_enabled(mode)
            pred = self.manager.predict_pair(self.path_a, self.path_b)
            r_h = self.client.get("/api/health").json()
            r_i = self.client.get("/api/model-info").json()

            # active_model in prediction must agree with health and model-info
            self.assertEqual(pred["active_model"], r_h["active_model"])
            self.assertEqual(r_h["active_model"], r_i["active_relationship_model"])

    # -----------------------------------------------------------------
    # Test 8: Repeated Requests Do Not Retain Stale Model Identity
    # -----------------------------------------------------------------
    def test_08_no_stale_identity_on_repeated_toggles(self):
        # Repeat toggles 5 times
        for _ in range(5):
            self.manager.set_phase4_enabled(True)
            self.assertTrue(self.client.get("/api/health").json()["phase4_models_enabled"])
            self.assertEqual(self.manager.predict_pair(self.path_a, self.path_b)["active_model"],
                             "Phase 4 Normalized Forensic MLP Pipeline (v1)")

            self.manager.set_phase4_enabled(False)
            self.assertFalse(self.client.get("/api/health").json()["phase4_models_enabled"])
            self.assertNotEqual(self.manager.predict_pair(self.path_a, self.path_b)["active_model"],
                                "Phase 4 Normalized Forensic MLP Pipeline (v1)")


if __name__ == "__main__":
    unittest.main()
