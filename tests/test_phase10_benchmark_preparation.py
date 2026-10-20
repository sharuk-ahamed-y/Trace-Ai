"""
tests/test_phase10_benchmark_preparation.py — Phase 10 Real-World Benchmark Dataset Preparation Tests

Verifies:
1. Preservation of contributor-confirmed pair (pair_0001) with ground truth direction (B_TO_A).
2. Proper output isolation in data/real_world_benchmark/controlled_transforms/.
3. Labeling compliance: generated pairs marked as 'controlled_transform', never independent captures.
4. Negative pair validity: all negative pairs have related=0, direction=NONE, and distinct source IDs.
5. Zero source leakage across splits and full manifest validation pass.
6. File existence and PIL readability of all referenced benchmark images.
7. Successful dry-run with src/evaluate_real_world.py on the prepared manifest.
8. Transform parameters recorded in JSON log.
9. Single-source safeguard: 0 negative pairs generated when only 1 distinct physical source exists.
10. Representation of all 24 confirmed independent physical scenes.
"""

import unittest
import sys
import subprocess
import shutil
import json
import tempfile
import pandas as pd
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_real_world import validate_manifest
from src.prepare_real_world_benchmark import prepare_benchmark_dataset


class TestPhase10BenchmarkPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ct_dir = PROJECT_ROOT / "data" / "real_world_benchmark" / "controlled_transforms"
        cls.manifest_path = cls.ct_dir / "manifest.csv"
        cls.images_dir = cls.ct_dir / "images"
        cls.params_log_path = cls.ct_dir / "transform_params.json"

    def test_01_controlled_transforms_artifacts_exist(self):
        """Manifest, parameter log, and image directory must exist."""
        self.assertTrue(self.ct_dir.exists(), f"Output dir {self.ct_dir} must exist.")
        self.assertTrue(self.manifest_path.exists(), f"Manifest {self.manifest_path} must exist.")
        self.assertTrue(self.images_dir.exists(), f"Images dir {self.images_dir} must exist.")
        self.assertTrue(self.params_log_path.exists(), f"Params log {self.params_log_path} must exist.")

    def test_02_preservation_of_contributor_pair(self):
        """Preserves contributor-confirmed pair_0001 exactly as documented."""
        df = pd.read_csv(self.manifest_path)
        row = df[df["pair_id"] == "pair_0001"]
        self.assertEqual(len(row), 1, "pair_0001 must be present in manifest.")
        p = row.iloc[0]
        self.assertEqual(p["image_a"], "WhatsApp Image 2026-10-13 at 1.31.11 AM.jpeg")
        self.assertEqual(p["image_b"], "WhatsApp Image 2026-10-13 at 1.31.12 AM.jpeg")
        self.assertEqual(p["source_a"], "real_src_0001")
        self.assertEqual(p["source_b"], "real_src_0001")
        self.assertEqual(int(p["related"]), 1)
        self.assertEqual(p["ground_truth_direction"], "B_TO_A")
        self.assertEqual(p["applied_transformations"], "crop")

    def test_03_controlled_transform_labels_and_notes(self):
        """Generated pairs must be labeled strictly as controlled_transform in notes."""
        df = pd.read_csv(self.manifest_path)
        ct_pairs = df[df["pair_id"].str.startswith("ct_pair_")]
        self.assertGreaterEqual(len(ct_pairs), 10, "Should have at least 10 controlled transform pairs.")
        for _, row in ct_pairs.iterrows():
            self.assertTrue(
                str(row["notes"]).startswith("controlled_transform:"),
                f"Pair {row['pair_id']} notes must start with 'controlled_transform:', got: {row['notes']}"
            )
            self.assertNotIn("independent_capture", str(row["notes"]).lower())

    def test_04_cross_scene_negative_pairs_validation(self):
        """Cross-scene negative pairs must have related=0, direction=NONE, distinct sources, and valid notes."""
        df = pd.read_csv(self.manifest_path)
        neg_pairs = df[df["related"] == 0]
        self.assertGreaterEqual(len(neg_pairs), 20, "Should have generated cross-scene negative pairs.")
        for _, row in neg_pairs.iterrows():
            self.assertEqual(int(row["related"]), 0)
            self.assertEqual(str(row["ground_truth_direction"]).strip(), "NONE")
            self.assertNotEqual(
                str(row["source_a"]).strip(),
                str(row["source_b"]).strip(),
                f"Negative pair {row['pair_id']} must have distinct sources."
            )
            self.assertEqual(str(row["applied_transformations"]).strip(), "none")
            self.assertTrue(
                str(row["notes"]).startswith("cross_source_negative:"),
                f"Negative pair notes should indicate cross-source negative, got: {row['notes']}"
            )

    def test_05_manifest_validation_clean(self):
        """Prepared manifest must pass evaluate_real_world.validate_manifest cleanly."""
        df = pd.read_csv(self.manifest_path)
        is_valid, errors = validate_manifest(df, self.images_dir)
        self.assertTrue(is_valid, f"Manifest validation failed with errors: {errors}")
        self.assertEqual(len(errors), 0)

    def test_06_image_files_exist_and_readable(self):
        """All images referenced in manifest must exist and be readable by PIL."""
        df = pd.read_csv(self.manifest_path)
        for _, row in df.iterrows():
            img_a_path = self.images_dir / str(row["image_a"])
            img_b_path = self.images_dir / str(row["image_b"])
            self.assertTrue(img_a_path.exists(), f"Image A missing: {img_a_path}")
            self.assertTrue(img_b_path.exists(), f"Image B missing: {img_b_path}")
            with Image.open(img_a_path) as im_a, Image.open(img_b_path) as im_b:
                im_a.verify()
                im_b.verify()

    def test_07_evaluator_dry_run_succeeds(self):
        """Evaluator dry-run must pass with code 0 on the prepared dataset."""
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "evaluate_real_world.py"),
            "--manifest", str(self.manifest_path),
            "--images-dir", str(self.images_dir),
            "--dry-run"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Dry-run failed: {res.stderr}")
        self.assertIn("Dry-run complete", res.stdout)
        self.assertIn("PASSED", res.stdout)

    def test_08_transform_params_logged(self):
        """Transform parameters must be recorded in JSON log."""
        with open(self.params_log_path, "r", encoding="utf-8") as f:
            params_log = json.load(f)
        self.assertIsInstance(params_log, dict)
        self.assertIn("ct_real_src_0001_crop.jpg", params_log)
        crop_entry = params_log["ct_real_src_0001_crop.jpg"]
        self.assertEqual(crop_entry["transformation"], "crop")
        self.assertEqual(crop_entry["source_id"], "real_src_0001")

    def test_09_single_source_safeguard_zero_negatives(self):
        """When only 1 source scene is present, 0 negative pairs must be fabricated."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_out = Path(tmp_dir) / "ct_test"
            n_pos, n_neg, _ = prepare_benchmark_dataset(
                input_images_dir=PROJECT_ROOT / "data" / "real_world_benchmark" / "images",
                existing_manifest_path=PROJECT_ROOT / "data" / "real_world_benchmark" / "real_pairs_manifest.csv",
                output_dir=tmp_out,
                seed=42,
                transforms=["crop"],
                source_mapping={
                    "WhatsApp Image 2026-10-13 at 1.31.11 AM.jpeg": "real_src_0001",
                    "WhatsApp Image 2026-10-13 at 1.31.12 AM.jpeg": "real_src_0001"
                },
                negative_strategy="ring"
            )
            self.assertEqual(n_neg, 0, "When only 1 source exists, 0 negative pairs must be created.")

    def test_10_independent_physical_sources_count(self):
        """Manifest must represent all 24 confirmed independent physical scenes."""
        df = pd.read_csv(self.manifest_path)
        all_sources = set(df["source_a"]).union(set(df["source_b"]))
        self.assertEqual(len(all_sources), 24, f"Expected 24 independent sources, found {len(all_sources)}: {all_sources}")


if __name__ == "__main__":
    unittest.main()
