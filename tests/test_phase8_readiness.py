"""
tests/test_phase8_readiness.py — Phase 8 Controlled Real-World Evaluation Readiness Review Suite

Verifies:
1. Probability/calibration labels are technically justified (uncalibrated sigmoid + validation-tuned thresholds).
2. Direction accuracy, confident-prediction accuracy, and abstention coverage are clearly distinguished.
3. Metadata-checkpoint compatibility safeguards (stale/incompatible metadata rejected).
4. Real-world evaluation status is strictly PREPARED BUT NOT EXECUTED.
5. Real-world evaluation runner does not fabricate numbers in the absence of authorized data.
6. Default configuration keeps ENABLE_PHASE4_MODELS=false.
"""

import unittest
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.model_manager import ModelManager, get_model_manager
from backend.app.config import ENABLE_PHASE4_MODELS, PHASE4_MODELS_DIR


class TestPhase8Readiness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.manager = get_model_manager()
        cls.test_dir = PROJECT_ROOT / "tests" / "test_scratch_phase8"
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        # Create two simple dummy images for pair inference testing
        from PIL import Image
        cls.path_a = str(cls.test_dir / "img_a.png")
        cls.path_b = str(cls.test_dir / "img_b.png")
        Image.new("RGB", (64, 64), color="blue").save(cls.path_a)
        Image.new("RGB", (64, 64), color="red").save(cls.path_b)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)
        # Ensure Phase 4 is left disabled
        cls.manager.set_phase4_enabled(False)

    def setUp(self):
        self.manager.set_phase4_enabled(False)

    def tearDown(self):
        self.manager.set_phase4_enabled(False)

    # -----------------------------------------------------------------
    # Test 1: Default Configuration Safety
    # -----------------------------------------------------------------
    def test_01_default_configuration_safety(self):
        """Verifies that Phase 4 is disabled by default in config and runtime."""
        self.assertFalse(ENABLE_PHASE4_MODELS, "ENABLE_PHASE4_MODELS must be False by default.")
        r = self.client.get("/api/health").json()
        self.assertFalse(r["phase4_models_enabled"], "Health endpoint must report Phase 4 disabled by default.")

    # -----------------------------------------------------------------
    # Test 2: Calibration Technical Justification
    # -----------------------------------------------------------------
    def test_02_calibration_technical_justification(self):
        """Verifies that model outputs are NOT labeled as calibrated probabilities without fitting."""
        self.manager.set_phase4_enabled(True)
        self.assertTrue(self.manager.phase4_loaded)

        pred = self.manager.predict_pair(self.path_a, self.path_b)
        self.assertEqual(
            pred["confidence_calibration"],
            "uncalibrated_sigmoid",
            "Model confidence must be explicitly labeled as uncalibrated sigmoid."
        )
        self.assertEqual(
            pred.get("decision_threshold_status"),
            "validation_tuned",
            "Decision threshold must be clearly marked as validation tuned."
        )

        # Check /api/model-metrics calibration disclosure
        r_metrics = self.client.get("/api/model-metrics").json()
        prob_cal = r_metrics["selected_relationship_model"].get("probability_calibration", {})
        self.assertEqual(prob_cal.get("status"), "UNCALIBRATED_SIGMOID")
        self.assertIsNone(prob_cal.get("procedure_fitted"))

    # -----------------------------------------------------------------
    # Test 3: Direction Metrics Demarcation
    # -----------------------------------------------------------------
    def test_03_direction_metrics_demarcation(self):
        """Verifies raw accuracy, confident accuracy, and abstention coverage are distinct."""
        self.manager.set_phase4_enabled(True)
        r_metrics = self.client.get("/api/model-metrics").json()
        dir_m = r_metrics.get("direction_classifier_metrics", {})

        self.assertIn("test_raw_accuracy", dir_m)
        self.assertIn("abstention_coverage", dir_m)
        self.assertIn("test_confident_accuracy", dir_m)

        raw_acc = dir_m["test_raw_accuracy"]
        conf_acc = dir_m["test_confident_accuracy"]
        coverage = dir_m["abstention_coverage"]

        self.assertGreaterEqual(conf_acc, raw_acc, "Confident accuracy should be >= raw accuracy.")
        self.assertGreater(coverage, 0.0)
        self.assertLessEqual(coverage, 1.0)

    # -----------------------------------------------------------------
    # Test 4: Checkpoint-Metadata Compatibility Safeguard
    # -----------------------------------------------------------------
    def test_04_checkpoint_metadata_incompatibility_rejection(self):
        """Verifies that mismatched or stale metadata causes Phase 4 loading to fail safely."""
        meta_path = PHASE4_MODELS_DIR / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            original_meta = json.load(f)

        corrupted_meta = dict(original_meta)
        corrupted_meta["version"] = "incompatible_v99"

        # Temporarily swap metadata with corrupted version
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(corrupted_meta, f)

        try:
            self.manager.set_phase4_enabled(True, reload=True)
            self.assertFalse(self.manager.phase4_loaded, "Should refuse to load with incompatible version.")
            self.assertIn("Incompatible metadata version", str(self.manager.phase4_fallback_reason))

            # API should report METRICS_UNAVAILABLE
            r_metrics = self.client.get("/api/model-metrics").json()
            self.assertEqual(r_metrics["selected_relationship_model"]["status"], "METRICS_UNAVAILABLE")
        finally:
            # Restore original valid metadata
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(original_meta, f, indent=2)
            self.manager.set_phase4_enabled(True, reload=True)
            self.manager.set_phase4_enabled(False)

    # -----------------------------------------------------------------
    # Test 5: Real-World Evaluation Status Safeguard
    # -----------------------------------------------------------------
    def test_05_real_world_metrics_unexecuted_safeguard(self):
        """Verifies real-world metrics are strictly reported as NOT_VALIDATED."""
        # Phase 4 Disabled
        self.manager.set_phase4_enabled(False)
        r1 = self.client.get("/api/model-metrics").json()
        self.assertEqual(r1["selected_relationship_model"]["real_world_metrics"]["status"], "NOT_VALIDATED")
        self.assertEqual(r1["selected_relationship_model"]["real_world_metrics"]["evaluation_samples"], 0)

        # Phase 4 Enabled
        self.manager.set_phase4_enabled(True)
        r2 = self.client.get("/api/model-metrics").json()
        self.assertEqual(r2["selected_relationship_model"]["real_world_metrics"]["status"], "NOT_VALIDATED")
        self.assertEqual(r2["selected_relationship_model"]["real_world_metrics"]["evaluation_samples"], 0)

    # -----------------------------------------------------------------
    # Test 6: Real-World Evaluation Script Safety
    # -----------------------------------------------------------------
    def test_06_real_world_script_safe_dry_run(self):
        """Verifies evaluate_real_world.py exits safely when no dataset is present."""
        cmd = [sys.executable, str(PROJECT_ROOT / "src" / "evaluate_real_world.py")]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("PREPARED BUT NOT EXECUTED", res.stdout)
        self.assertIn("Zero results fabricated or assumed", res.stdout)


if __name__ == "__main__":
    unittest.main()
