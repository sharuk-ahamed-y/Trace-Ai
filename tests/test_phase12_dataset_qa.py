"""
tests/test_phase12_dataset_qa.py — Phase 12 Dataset Quality Assurance & Leakage Prevention Tests

Verifies:
1. Active benchmark manifest passes all QA audit checks cleanly.
2. Missing image files are caught and reported as critical errors.
3. Corrupted image files (unreadable headers) are caught and reported.
4. Path traversal sequences (../, absolute paths) are rejected.
5. Inconsistent labels (related=1 with source_a != source_b, related=0 with source_a == source_b) are flagged.
6. Invalid direction values (e.g. related=0 with direction != NONE) are flagged.
7. Duplicate image SHA-256 content hashes are detected and clustered.
8. Source-group leakage across evaluation splits is strictly caught.
9. Quarantined unknown labels ('unknown', -1) are treated as candidate unverified rows.
10. Unclassified positive rows without explicit provenance categorization are rejected.
11. Markdown report generation is deterministic and reproducible.
12. Phase 11 safety safeguards and default ENABLE_PHASE4_MODELS=false remain intact.
"""

import unittest
import sys
import tempfile
import shutil
import hashlib
from pathlib import Path
from PIL import Image
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qa_benchmark_dataset import audit_dataset, generate_markdown_report
from backend.app.config import ENABLE_PHASE4_MODELS


class TestPhase12DatasetQA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ct_dir = PROJECT_ROOT / "data" / "real_world_benchmark" / "controlled_transforms"
        cls.manifest_path = cls.ct_dir / "manifest.csv"
        cls.images_dir = cls.ct_dir / "images"

    def test_01_active_benchmark_passes_qa(self):
        """Active benchmark manifest must pass all QA checks with zero critical errors."""
        res = audit_dataset(self.manifest_path, self.images_dir)
        self.assertTrue(res["is_valid"], f"Active manifest failed QA with errors: {res['errors']}")
        self.assertEqual(len(res["errors"]), 0)
        self.assertEqual(res["summary"]["total_pairs"], 35)
        self.assertEqual(res["summary"]["unique_source_count"], 24)

    def test_02_missing_image_caught(self):
        """Missing image file referenced in manifest must be caught as a critical error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            df = pd.read_csv(self.manifest_path).copy()
            df.loc[0, "image_a"] = "non_existent_img_9999.jpg"
            bad_manifest = tmp_path / "manifest.csv"
            df.to_csv(bad_manifest, index=False)

            res = audit_dataset(bad_manifest, self.images_dir)
            self.assertFalse(res["is_valid"])
            self.assertTrue(any("missing" in e.lower() for e in res["errors"]))

    def test_03_corrupt_image_caught(self):
        """0-byte or corrupted image must fail PIL verification during QA."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_tmp = tmp_path / "images"
            images_tmp.mkdir()

            # Copy one valid image
            src_img = next(self.images_dir.glob("*.jpeg"))
            shutil.copy(src_img, images_tmp / "valid.jpg")

            # Create corrupt image (0 bytes)
            (images_tmp / "corrupt.jpg").write_bytes(b"")

            df = pd.DataFrame([{
                "pair_id": "test_p01",
                "image_a": "valid.jpg",
                "image_b": "corrupt.jpg",
                "source_a": "s1",
                "source_b": "s1",
                "related": 1,
                "ground_truth_direction": "A_TO_B",
                "notes": "controlled_transform test"
            }])
            m_path = tmp_path / "manifest.csv"
            df.to_csv(m_path, index=False)

            res = audit_dataset(m_path, images_tmp)
            self.assertFalse(res["is_valid"])
            self.assertTrue(any("corrupted or unreadable" in e for e in res["errors"]))

    def test_04_path_traversal_rejected(self):
        """Image paths escaping the designated directory must trigger traversal errors."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            df = pd.read_csv(self.manifest_path).copy()
            df.loc[0, "image_a"] = "../secret_file.jpg"
            m_path = tmp_path / "manifest.csv"
            df.to_csv(m_path, index=False)

            res = audit_dataset(m_path, self.images_dir)
            self.assertFalse(res["is_valid"])
            self.assertTrue(any("path traversal" in e for e in res["errors"]))

    def test_05_inconsistent_labels_flagged(self):
        """Related=1 with different sources and Related=0 with same sources must fail."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            df = pd.read_csv(self.manifest_path).copy()
            # Inconsistency 1: related=1 but sources differ
            df.loc[0, "source_b"] = "diff_source_9999"
            # Inconsistency 2: related=0 but sources match
            df.loc[12, "source_b"] = df.loc[12, "source_a"]
            m_path = tmp_path / "manifest.csv"
            df.to_csv(m_path, index=False)

            res = audit_dataset(m_path, self.images_dir)
            self.assertFalse(res["is_valid"])
            self.assertTrue(any("'related' is 1 but source_a" in e for e in res["errors"]))
            self.assertTrue(any("'related' is 0 but source_a == source_b" in e for e in res["errors"]))

    def test_06_invalid_direction_rules_flagged(self):
        """Unrelated pair with non-NONE direction must trigger semantic error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            df = pd.read_csv(self.manifest_path).copy()
            df.loc[12, "ground_truth_direction"] = "A_TO_B"  # related is 0!
            m_path = tmp_path / "manifest.csv"
            df.to_csv(m_path, index=False)

            res = audit_dataset(m_path, self.images_dir)
            self.assertFalse(res["is_valid"])
            self.assertTrue(any("is 0 but direction is 'A_TO_B'" in e for e in res["errors"]))

    def test_07_duplicate_image_hashes_detected(self):
        """Identical image contents across different filenames must be detected as duplicate clusters."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_tmp = tmp_path / "images"
            images_tmp.mkdir()

            src_img = next(self.images_dir.glob("*.jpeg"))
            shutil.copy(src_img, images_tmp / "img_original.jpg")
            shutil.copy(src_img, images_tmp / "img_copy_exact.jpg")

            df = pd.DataFrame([{
                "pair_id": "test_dup_01",
                "image_a": "img_original.jpg",
                "image_b": "img_copy_exact.jpg",
                "source_a": "s1",
                "source_b": "s1",
                "related": 1,
                "ground_truth_direction": "A_TO_B",
                "notes": "controlled_transform test"
            }])
            m_path = tmp_path / "manifest.csv"
            df.to_csv(m_path, index=False)

            res = audit_dataset(m_path, images_tmp)
            self.assertTrue(res["is_valid"])
            self.assertEqual(res["summary"]["duplicate_hash_clusters_count"], 1)
            self.assertTrue(any("Exact duplicate image content detected" in w for w in res["warnings"]))

    def test_08_source_group_leakage_across_splits_caught(self):
        """Derivative of a source placed into a different split must be caught as leakage."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            df = pd.read_csv(self.manifest_path).copy()
            # Split pair 1 into train and pair 2 (sharing real_src_0001) into test
            df.loc[0, "split"] = "train"
            df.loc[1, "split"] = "test"
            m_path = tmp_path / "manifest.csv"
            df.to_csv(m_path, index=False)

            res = audit_dataset(m_path, self.images_dir)
            self.assertFalse(res["is_valid"])
            self.assertTrue(any("Source-group leakage" in e for e in res["errors"]))
            self.assertTrue(len(res["summary"]["split_leakage_errors"]) > 0)

    def test_09_quarantined_unknown_labels_tracked(self):
        """Unknown or unverified labels must be classified as candidate_unverified, not error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            df = pd.read_csv(self.manifest_path).copy()
            df["related"] = df["related"].astype(object)
            df.loc[0, "related"] = "unknown"
            df.loc[0, "notes"] = "placeholder candidate"
            m_path = tmp_path / "manifest.csv"
            df.to_csv(m_path, index=False)

            res = audit_dataset(m_path, self.images_dir)
            self.assertTrue(res["is_valid"])
            self.assertEqual(res["summary"]["category_counts"]["candidate_unverified"], 1)

    def test_10_unclassified_positive_pairs_rejected(self):
        """Positive pair without contributor confirmation or controlled tag must be rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            df = pd.read_csv(self.manifest_path).copy()
            df.loc[0, "pair_id"] = "unknown_pair_123"
            df.loc[0, "notes"] = "some arbitrary notes without lineage"
            m_path = tmp_path / "manifest.csv"
            df.to_csv(m_path, index=False)

            res = audit_dataset(m_path, self.images_dir)
            self.assertFalse(res["is_valid"])
            self.assertTrue(any("Positive pair lacks clear categorization" in e for e in res["errors"]))

    def test_11_markdown_report_generation(self):
        """Markdown report must generate reproducibly and contain key audit sections."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            report_out = tmp_path / "audit_report.md"

            res = audit_dataset(self.manifest_path, self.images_dir)
            generate_markdown_report(res, report_out)

            self.assertTrue(report_out.exists())
            content = report_out.read_text(encoding="utf-8")
            self.assertIn("REAL-WORLD DATASET QA & INTEGRITY REPORT", content)
            self.assertIn("**Total Pairs Audited:** 35", content)
            self.assertIn("Disaggregated Category Breakdown", content)
            self.assertIn("Source Group & Leakage Audit", content)

    def test_12_production_safeguard_defaults_preserved(self):
        """Phase 4 models must remain disabled by default in production config."""
        self.assertFalse(ENABLE_PHASE4_MODELS)


if __name__ == "__main__":
    unittest.main()
