"""
tests/test_phase11_safe_benchmark_evaluation.py — Phase 11 Safe Benchmark Evaluation Audit Tests

Verifies:
1. All 35 benchmark pairs exist on disk, have valid PIL headers, valid dimensions, and valid metadata.
2. Evaluator cleanly categorizes pairs into three distinct subgroups (1 contributor-confirmed, 10 controlled-transform, 24 cross-scene negative).
3. Evaluator dry-run outputs the subgroup breakdown and exits 0 without running inference.
4. Evaluator without --execute safely reports PREPARED BUT NOT EXECUTED with 0 results fabricated.
5. Evaluator refuses silent heuristic fallback if Phase 4 models cannot be loaded.
6. ENABLE_PHASE4_MODELS remains false by default in production configuration.
"""

import unittest
import sys
import subprocess
import os
import pandas as pd
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_real_world import validate_manifest, classify_pair_category
from backend.app.config import ENABLE_PHASE4_MODELS


class TestPhase11SafeBenchmarkEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ct_dir = PROJECT_ROOT / "data" / "real_world_benchmark" / "controlled_transforms"
        cls.manifest_path = cls.ct_dir / "manifest.csv"
        cls.images_dir = cls.ct_dir / "images"

    def test_01_all_35_pairs_verified_and_readable(self):
        """All 35 pairs in the benchmark manifest must be readable and have valid PIL headers."""
        self.assertTrue(self.manifest_path.exists(), "Manifest must exist.")
        df = pd.read_csv(self.manifest_path)
        self.assertEqual(len(df), 35, "Manifest must contain exactly 35 pairs.")

        for idx, row in df.iterrows():
            pid = row["pair_id"]
            img_a = self.images_dir / row["image_a"]
            img_b = self.images_dir / row["image_b"]

            self.assertTrue(img_a.exists(), f"{pid}: image_a {row['image_a']} must exist on disk.")
            self.assertTrue(img_b.exists(), f"{pid}: image_b {row['image_b']} must exist on disk.")

            with Image.open(img_a) as im_a:
                self.assertGreater(im_a.width, 0)
                self.assertGreater(im_a.height, 0)

            with Image.open(img_b) as im_b:
                self.assertGreater(im_b.width, 0)
                self.assertGreater(im_b.height, 0)

    def test_02_evaluator_subgroup_classification(self):
        """Pairs must be cleanly partitioned into 3 isolated categories without conflation."""
        df = pd.read_csv(self.manifest_path)
        categories = [classify_pair_category(row) for _, row in df.iterrows()]

        from collections import Counter
        counts = Counter(categories)

        self.assertEqual(counts["contributor_confirmed"], 1, "Exactly 1 contributor-confirmed pair.")
        self.assertEqual(counts["controlled_transform"], 10, "Exactly 10 controlled-transform pairs.")
        self.assertEqual(counts["cross_scene_negative"], 24, "Exactly 24 cross-scene negative pairs.")
        self.assertEqual(counts.get("unclassified", 0), 0, "No unclassified pairs allowed.")

    def test_03_evaluator_dry_run_outputs_subgroup_breakdown(self):
        """Dry-run must output the subgroup breakdown and exit with code 0."""
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "evaluate_real_world.py"),
            "--manifest", str(self.manifest_path),
            "--images-dir", str(self.images_dir),
            "--dry-run"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Contributor-Confirmed Real Pairs: 1", res.stdout)
        self.assertIn("Controlled-Transform Pairs:       10", res.stdout)
        self.assertIn("Cross-Scene Negative Pairs:       24", res.stdout)
        self.assertIn("Dry-run complete", res.stdout)

    def test_04_evaluator_without_execute_exits_safely(self):
        """Without --execute, evaluator reports PREPARED BUT NOT EXECUTED and exits 0."""
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "evaluate_real_world.py"),
            "--manifest", str(self.manifest_path),
            "--images-dir", str(self.images_dir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("PREPARED BUT NOT EXECUTED", res.stdout)
        self.assertIn("Zero results fabricated or assumed", res.stdout)
        self.assertIn("Subgroup Reporting: Enabled", res.stdout)

    def test_05_evaluator_refuses_heuristic_fallback(self):
        """Evaluator must reject running when Phase 4 checkpoints cannot be loaded."""
        from unittest.mock import patch
        from backend.app.services.model_manager import ModelManager
        from src.evaluate_real_world import main

        def fake_set_phase4(mgr_self, enabled, reload=False):
            mgr_self.phase4_loaded = False
            mgr_self.phase4_fallback_reason = "Simulated missing checkpoint for test"

        with patch.object(ModelManager, "set_phase4_enabled", fake_set_phase4):
            with patch("sys.argv", [
                "evaluate_real_world.py",
                "--manifest", str(self.manifest_path),
                "--images-dir", str(self.images_dir),
                "--execute"
            ]):
                ret = main()
                self.assertEqual(ret, 1, "Evaluator must exit with returncode 1 when Phase 4 models fail to load.")

    def test_06_default_flag_remains_false(self):
        """ENABLE_PHASE4_MODELS must remain false by default."""
        self.assertFalse(ENABLE_PHASE4_MODELS, "ENABLE_PHASE4_MODELS must remain false by default.")

    def test_07_missing_or_corrupt_checkpoint_keeps_phase4_unavailable(self):
        """Corrupt or missing checkpoints must keep Phase 4 unavailable with a diagnostic reason."""
        import torch
        from unittest.mock import patch, MagicMock
        from backend.app.services.model_manager import ModelManager

        manager = ModelManager()
        manager.initialized = True
        manager.device = torch.device("cpu")
        manager.embedder = MagicMock()
        manager.embedder_dim = 768

        with patch("torch.load", side_effect=RuntimeError("Corrupt checkpoint file")):
            manager.set_phase4_enabled(True, reload=True)
            self.assertFalse(manager.phase4_loaded, "Phase 4 must not be marked loaded on checkpoint corruption.")
            self.assertIn("Corrupt checkpoint", manager.phase4_fallback_reason)

    def test_08_invalid_metadata_or_scaler_dimensions_rejected(self):
        """Incompatible scaler dimensions or invalid threshold values must be rejected."""
        import torch
        import joblib
        from unittest.mock import patch, MagicMock
        from backend.app.services.model_manager import ModelManager
        import numpy as np

        manager = ModelManager()
        manager.initialized = True
        manager.device = torch.device("cpu")
        manager.embedder = MagicMock()
        manager.embedder_dim = 768

        # Mock an invalid scaler with 10 features instead of 115
        bad_scaler = MagicMock()
        bad_scaler.n_features_in_ = 10
        bad_scaler.mean_ = np.zeros(10)
        bad_scaler.scale_ = np.ones(10)

        with patch("joblib.load", return_value=bad_scaler):
            manager.set_phase4_enabled(True, reload=True)
            self.assertFalse(manager.phase4_loaded, "Phase 4 must not load with incompatible scaler dimensions.")
            self.assertIn("scaler", manager.phase4_fallback_reason.lower())

    def test_09_phase4_inference_exception_followed_by_legacy_fallback_refused_in_strict_mode(self):
        """In strict_phase4 mode, Phase 4 exceptions must re-raise rather than silently falling back."""
        import torch
        from unittest.mock import patch, MagicMock
        from backend.app.services.model_manager import ModelManager

        manager = ModelManager()
        manager.initialized = True
        manager.embedder = MagicMock()
        manager.embedder_dim = 768
        manager.get_embedding = MagicMock(return_value=torch.zeros(768))
        manager.phase4_enabled = True
        manager.phase4_loaded = True

        with patch.object(manager, "get_57d_representation", side_effect=RuntimeError("CUDA out of memory simulation")):
            dummy_img = self.images_dir / "ct_WhatsApp Image 2026-10-13 at 1.31.12 AM_crop.jpeg"
            with self.assertRaises(RuntimeError) as ctx:
                manager.predict_pair(dummy_img, dummy_img, strict_phase4=True)
            self.assertIn("Phase 4 inference failed under strict mode", str(ctx.exception))

    def test_10_incorrect_or_missing_inference_provenance_rejected(self):
        """Evaluator must reject results with missing or fallback inference provenance."""
        from unittest.mock import patch, MagicMock
        from backend.app.services.model_manager import ModelManager
        from src.evaluate_real_world import main

        fake_res = {
            "related_confidence": 0.99,
            "direction": "A_TO_B",
            "direction_confidence": 0.95,
            "transformation": "crop",
            "compound_tags": ["crop"],
            "inference_mode": "trained_model",
            "active_model": "Phase 4 Normalized Forensic MLP Pipeline (v1)",
            # Missing or forged inference_path indicating fallback
            "inference_path": {"pipeline": "legacy_fallback", "is_fallback": True}
        }

        mock_mgr = MagicMock()
        mock_mgr.phase4_loaded = True
        mock_mgr.phase4_rel_threshold = 0.5
        mock_mgr.phase4_dir_delta_uncertain = 0.1
        mock_mgr.predict_pair.return_value = fake_res

        with patch("backend.app.services.model_manager.ModelManager", return_value=mock_mgr):
            with patch("sys.argv", [
                "evaluate_real_world.py",
                "--manifest", str(self.manifest_path),
                "--images-dir", str(self.images_dir),
                "--execute"
            ]):
                ret = main()
                self.assertEqual(ret, 1, "Evaluator must exit with returncode 1 when inference provenance is invalid.")

    def test_11_manifest_validation_rejects_missing_or_unreadable_image(self):
        """Manifest validation must fail if an image is missing or corrupted on disk."""
        import tempfile
        df = pd.read_csv(self.manifest_path).copy()
        df.loc[0, "image_a"] = "non_existent_image_12345.jpg"

        is_valid, errors = validate_manifest(df, self.images_dir, check_images=True)
        self.assertFalse(is_valid, "Validation must fail for missing image.")
        self.assertTrue(any("missing" in e for e in errors), f"Expected 'missing' in errors: {errors}")

        # Test corrupt (0-byte) image
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            corrupt_file = tmp_path / "corrupt.jpg"
            corrupt_file.write_bytes(b"")  # 0 bytes

            df_corrupt = df.copy()
            df_corrupt.loc[0, "image_a"] = "corrupt.jpg"
            df_corrupt.loc[0, "image_b"] = "corrupt.jpg"

            is_valid_c, errors_c = validate_manifest(df_corrupt, tmp_path, check_images=True)
            self.assertFalse(is_valid_c, "Validation must fail for corrupt image.")
            self.assertTrue(any("unreadable" in e for e in errors_c), f"Expected 'unreadable' in errors: {errors_c}")

    def test_12_manifest_validation_rejects_unclassified_row(self):
        """Manifest validation must reject any unclassified pair row."""
        df = pd.read_csv(self.manifest_path).copy()
        df.loc[0, "pair_id"] = "unknown_pair_9999"
        df.loc[0, "notes"] = "some arbitrary notes"
        df.loc[0, "related"] = 1  # positive without recognized ID or notes pattern

        is_valid, errors = validate_manifest(df, self.images_dir, check_images=False)
        self.assertFalse(is_valid, "Validation must fail for unclassified row.")
        self.assertTrue(any("unclassified" in e for e in errors), f"Expected 'unclassified' in errors: {errors}")

    def test_13_manifest_validation_rejects_path_traversal(self):
        """Manifest validation must reject paths containing traversal sequences."""
        df = pd.read_csv(self.manifest_path).copy()
        df.loc[0, "image_a"] = "../secret.jpg"

        is_valid, errors = validate_manifest(df, self.images_dir, check_images=False)
        self.assertFalse(is_valid, "Validation must fail for path traversal.")
        self.assertTrue(any("path traversal" in e or "invalid relative path" in e for e in errors))

    def test_14_partial_evaluation_fails_to_produce_complete_report(self):
        """Evaluator must halt with code 1 if any pair fails to evaluate, preventing partial reports."""
        from unittest.mock import patch, MagicMock
        from backend.app.services.model_manager import ModelManager
        from src.evaluate_real_world import main

        mock_mgr = MagicMock()
        mock_mgr.phase4_loaded = True
        mock_mgr.phase4_rel_threshold = 0.5
        mock_mgr.phase4_dir_delta_uncertain = 0.1
        # Predict raises on pair 5
        call_count = [0]
        def fake_predict(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 5:
                raise RuntimeError("Simulated transient failure on pair 5")
            return {
                "related_confidence": 0.99,
                "direction": "A_TO_B",
                "direction_confidence": 0.95,
                "transformation": "crop",
                "compound_tags": ["crop"],
                "inference_mode": "trained_model",
                "active_model": "Phase 4 Normalized Forensic MLP Pipeline (v1)",
                "inference_path": {"pipeline": "phase4_v1", "is_fallback": False}
            }
        mock_mgr.predict_pair.side_effect = fake_predict

        with patch("backend.app.services.model_manager.ModelManager", return_value=mock_mgr):
            with patch("sys.argv", [
                "evaluate_real_world.py",
                "--manifest", str(self.manifest_path),
                "--images-dir", str(self.images_dir),
                "--execute"
            ]):
                ret = main()
                self.assertEqual(ret, 1, "Evaluator must return 1 and halt when evaluation is partial.")


if __name__ == "__main__":
    unittest.main()
