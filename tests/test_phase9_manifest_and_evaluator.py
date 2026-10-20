"""
tests/test_phase9_manifest_and_evaluator.py — Phase 9 Manifest Validation & Evaluator Safeguards

Verifies:
1. Missing manifest safely reports PREPARED BUT NOT EXECUTED (exit code 0).
2. Manifest template passes all schema and semantic consistency checks.
3. Missing required columns are caught and reported with actionable errors.
4. Source leakage across splits is detected and rejected.
5. Semantic inconsistencies (related=1 with distinct sources, or related=0 with identical sources) are rejected.
6. Duplicate pair IDs or duplicate image pairs are detected and rejected.
7. Path traversal and unauthorized file extensions are blocked.
"""

import unittest
import sys
import subprocess
import shutil
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_real_world import validate_manifest


class TestPhase9ManifestAndEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = PROJECT_ROOT / "tests" / "test_scratch_phase9"
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.dummy_img_dir = cls.test_dir / "images"
        cls.dummy_img_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_missing_manifest_exits_safely(self):
        """When manifest does not exist, script reports PREPARED BUT NOT EXECUTED and exits 0."""
        cmd = [sys.executable, str(PROJECT_ROOT / "src" / "evaluate_real_world.py"), "--manifest", "nonexistent_manifest.csv"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("PREPARED BUT NOT EXECUTED", res.stdout)
        self.assertIn("Zero results fabricated or assumed", res.stdout)

    def test_02_manifest_template_validates_cleanly(self):
        """The manifest template should pass all schema and semantic consistency checks."""
        template_path = PROJECT_ROOT / "data" / "real_world_benchmark" / "real_pairs_manifest.template.csv"
        self.assertTrue(template_path.exists(), "Template file must exist.")
        df = pd.read_csv(template_path)
        is_valid, errors = validate_manifest(df, self.dummy_img_dir)
        self.assertTrue(is_valid, f"Template validation failed with errors: {errors}")
        self.assertEqual(len(errors), 0)

    def test_03_missing_columns_rejected(self):
        """Missing required columns must be rejected with actionable message."""
        bad_df = pd.DataFrame([{
            "pair_id": "p1",
            "image_a": "img_a.jpg",
            "image_b": "img_b.jpg",
            "source_a": "s1",
            "source_b": "s1",
            "related": 1
            # Missing ground_truth_direction
        }])
        is_valid, errors = validate_manifest(bad_df, self.dummy_img_dir)
        self.assertFalse(is_valid)
        self.assertTrue(any("ground_truth_direction" in e for e in errors))

    def test_04_source_leakage_across_splits_rejected(self):
        """Same source appearing in multiple splits must trigger a source-leakage error."""
        leaky_df = pd.DataFrame([
            {"pair_id": "p1", "image_a": "a1.jpg", "image_b": "a2.jpg", "source_a": "src_shared", "source_b": "src_shared", "related": 1, "ground_truth_direction": "A_TO_B", "split": "train"},
            {"pair_id": "p2", "image_a": "a3.jpg", "image_b": "a4.jpg", "source_a": "src_shared", "source_b": "src_shared", "related": 1, "ground_truth_direction": "A_TO_B", "split": "test"}
        ])
        is_valid, errors = validate_manifest(leaky_df, self.dummy_img_dir)
        self.assertFalse(is_valid)
        self.assertTrue(any("Source leakage across evaluation splits" in e for e in errors))

    def test_05_semantic_inconsistency_rejected(self):
        """Contradictions between source IDs, relationship labels, and directions must be rejected."""
        # Case A: related=1 but sources differ
        inconsistent_df_1 = pd.DataFrame([
            {"pair_id": "p1", "image_a": "a.jpg", "image_b": "b.jpg", "source_a": "src_1", "source_b": "src_2", "related": 1, "ground_truth_direction": "A_TO_B"}
        ])
        is_valid, errors = validate_manifest(inconsistent_df_1, self.dummy_img_dir)
        self.assertFalse(is_valid)
        self.assertTrue(any("'related' is 1 but source_a" in e for e in errors))

        # Case B: related=0 but sources are identical
        inconsistent_df_2 = pd.DataFrame([
            {"pair_id": "p2", "image_a": "a.jpg", "image_b": "b.jpg", "source_a": "src_1", "source_b": "src_1", "related": 0, "ground_truth_direction": "NONE"}
        ])
        is_valid, errors = validate_manifest(inconsistent_df_2, self.dummy_img_dir)
        self.assertFalse(is_valid)
        self.assertTrue(any("'related' is 0 but source_a == source_b" in e for e in errors))

        # Case C: related=0 but direction is A_TO_B
        inconsistent_df_3 = pd.DataFrame([
            {"pair_id": "p3", "image_a": "a.jpg", "image_b": "b.jpg", "source_a": "src_1", "source_b": "src_2", "related": 0, "ground_truth_direction": "A_TO_B"}
        ])
        is_valid, errors = validate_manifest(inconsistent_df_3, self.dummy_img_dir)
        self.assertFalse(is_valid)
        self.assertTrue(any("must be 'NONE'" in e for e in errors))

    def test_06_duplicate_entries_rejected(self):
        """Duplicate pair IDs or duplicate image pairs must be detected and rejected."""
        dup_df = pd.DataFrame([
            {"pair_id": "p1", "image_a": "img_a.jpg", "image_b": "img_b.jpg", "source_a": "src_1", "source_b": "src_1", "related": 1, "ground_truth_direction": "A_TO_B"},
            {"pair_id": "p1", "image_a": "img_c.jpg", "image_b": "img_d.jpg", "source_a": "src_2", "source_b": "src_2", "related": 1, "ground_truth_direction": "A_TO_B"}
        ])
        is_valid, errors = validate_manifest(dup_df, self.dummy_img_dir)
        self.assertFalse(is_valid)
        self.assertTrue(any("Duplicate pair_id entries found" in e for e in errors))

    def test_07_invalid_path_and_extension_rejected(self):
        """Path traversal and unsupported extensions must be rejected."""
        bad_path_df = pd.DataFrame([
            {"pair_id": "p1", "image_a": "../secret.png", "image_b": "b.exe", "source_a": "src_1", "source_b": "src_1", "related": 1, "ground_truth_direction": "A_TO_B"}
        ])
        is_valid, errors = validate_manifest(bad_path_df, self.dummy_img_dir)
        self.assertFalse(is_valid)
        self.assertTrue(any("invalid relative path" in e for e in errors))
        self.assertTrue(any("unsupported file extension" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
