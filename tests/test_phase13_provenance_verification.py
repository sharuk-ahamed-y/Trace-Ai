"""
tests/test_phase13_provenance_verification.py — Phase 13 Provenance Verification & Annotation Tests

Verifies:
1. Active benchmark manifest maintains backward compatibility (passes QA audit).
2. Strict provenance validation (--check-provenance) catches missing required provenance columns.
3. 'contributor_confirmed' pairs require explicit affirmative consent confirmation.
4. 'contributor_confirmed' pairs require a documented, non-empty provenance chain of custody.
5. Hard-negative candidates cannot share identical source IDs (source_a != source_b enforced).
6. Hard-negative candidates cannot be assigned positive related=1 labels.
7. Hard-negative candidates cannot be assigned directional transform labels (must be NONE or UNCERTAIN).
8. Hard-negative candidates cannot be marked 'reviewer_approved' without documented review notes.
9. Pairs flagged with uncertainty_flag=1 cannot be labeled related=1 and must be quarantined/pending.
10. Invalid provenance or verification statuses are detected and rejected.
11. Candidate workspace template contains all required provenance and isolation columns.
12. Phase 4 models remain disabled by default (ENABLE_PHASE4_MODELS=false).
"""

import unittest
import sys
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qa_benchmark_dataset import audit_dataset, REQUIRED_PROVENANCE_COLUMNS
from backend.app.config import ENABLE_PHASE4_MODELS


class TestPhase13ProvenanceVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ct_dir = PROJECT_ROOT / "data" / "real_world_benchmark" / "controlled_transforms"
        cls.manifest_path = cls.ct_dir / "manifest.csv"
        cls.images_dir = cls.ct_dir / "images"
        cls.workspace_dir = PROJECT_ROOT / "data" / "real_world_benchmark" / "expansion_workspace"

    def test_01_active_benchmark_backward_compatibility(self):
        """Active benchmark manifest without provenance columns passes standard QA audit."""
        res = audit_dataset(self.manifest_path, self.images_dir, check_provenance=False)
        self.assertTrue(res["is_valid"], f"Active manifest failed QA: {res['errors']}")
        self.assertEqual(len(res["errors"]), 0)
        self.assertEqual(res["summary"]["total_pairs"], 35)

    def test_02_provenance_check_requires_provenance_columns(self):
        """When check_provenance=True, manifests lacking provenance columns trigger critical errors."""
        res = audit_dataset(self.manifest_path, self.images_dir, check_provenance=True)
        self.assertFalse(res["is_valid"])
        self.assertTrue(any("missing required provenance columns" in e for e in res["errors"]))

    def test_03_contributor_confirmed_requires_valid_consent(self):
        """Contributor confirmed pair without valid affirmative consent must fail audit."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_tmp = tmp_path / "images"
            images_tmp.mkdir()

            src_img = next(self.images_dir.glob("*.jpeg"))
            shutil.copy(src_img, images_tmp / "img_orig.jpg")
            shutil.copy(src_img, images_tmp / "img_der.jpg")

            df = pd.DataFrame([{
                "pair_id": "cand_01",
                "image_a": "img_orig.jpg",
                "image_b": "img_der.jpg",
                "source_a": "s100",
                "source_b": "s100",
                "related": 1,
                "ground_truth_direction": "A_TO_B",
                "provenance_status": "contributor_confirmed",
                "verification_status": "pending_review",
                "consent_confirmed": "unconfirmed",  # Missing valid consent!
                "provenance_chain": "camera capture -> crop",
                "uncertainty_flag": 0,
                "notes": "test"
            }])
            m_path = tmp_path / "manifest.csv"
            df.to_csv(m_path, index=False)

            res = audit_dataset(m_path, images_tmp, check_provenance=True)
            self.assertFalse(res["is_valid"])
            self.assertTrue(any("requires valid consent_confirmed" in e for e in res["errors"]))

    def test_04_contributor_confirmed_requires_non_empty_provenance_chain(self):
        """Contributor confirmed pair with empty provenance chain must fail audit."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_tmp = tmp_path / "images"
            images_tmp.mkdir()

            src_img = next(self.images_dir.glob("*.jpeg"))
            shutil.copy(src_img, images_tmp / "img_orig.jpg")
            shutil.copy(src_img, images_tmp / "img_der.jpg")

            df = pd.DataFrame([{
                "pair_id": "cand_02",
                "image_a": "img_orig.jpg",
                "image_b": "img_der.jpg",
                "source_a": "s100",
                "source_b": "s100",
                "related": 1,
                "ground_truth_direction": "A_TO_B",
                "provenance_status": "contributor_confirmed",
                "verification_status": "pending_review",
                "consent_confirmed": "confirmed",
                "provenance_chain": "",  # Empty provenance chain!
                "uncertainty_flag": 0,
                "notes": "test"
            }])
            m_path = tmp_path / "manifest.csv"
            df.to_csv(m_path, index=False)

            res = audit_dataset(m_path, images_tmp, check_provenance=True)
            self.assertFalse(res["is_valid"])
            self.assertTrue(any("requires non-empty provenance_chain" in e for e in res["errors"]))

    def test_05_hard_negative_candidate_source_group_consistency(self):
        """Hard-negative candidate cannot share the same source_id."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_tmp = tmp_path / "images"
            images_tmp.mkdir()

            src_img = next(self.images_dir.glob("*.jpeg"))
            shutil.copy(src_img, images_tmp / "img_a.jpg")
            shutil.copy(src_img, images_tmp / "img_b.jpg")

            df = pd.DataFrame([{
                "pair_id": "cand_03",
                "image_a": "img_a.jpg",
                "image_b": "img_b.jpg",
                "source_a": "s200",
                "source_b": "s200",  # Same source ID!
                "related": 0,
                "ground_truth_direction": "NONE",
                "provenance_status": "hard_negative_visual_match",
                "verification_status": "pending_review",
                "consent_confirmed": "confirmed",
                "provenance_chain": "two shots of same scene",
                "uncertainty_flag": 0,
                "notes": "test hard negative"
            }])
            m_path = tmp_path / "manifest.csv"
            df.to_csv(m_path, index=False)

            res = audit_dataset(m_path, images_tmp, check_provenance=True)
            self.assertFalse(res["is_valid"])
            self.assertTrue(any("Hard negative pair cannot share identical source IDs" in e for e in res["errors"]))

    def test_06_hard_negative_candidate_cannot_be_related_positive(self):
        """Hard-negative candidate cannot be labeled related=1."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_tmp = tmp_path / "images"
            images_tmp.mkdir()

            src_img = next(self.images_dir.glob("*.jpeg"))
            shutil.copy(src_img, images_tmp / "img_a.jpg")
            shutil.copy(src_img, images_tmp / "img_b.jpg")

            df = pd.DataFrame([{
                "pair_id": "cand_04",
                "image_a": "img_a.jpg",
                "image_b": "img_b.jpg",
                "source_a": "s200",
                "source_b": "s201",
                "related": 1,  # Invalid: cannot be related=1!
                "ground_truth_direction": "NONE",
                "provenance_status": "hard_negative_visual_match",
                "verification_status": "pending_review",
                "consent_confirmed": "confirmed",
                "provenance_chain": "two shots of same scene",
                "uncertainty_flag": 0,
                "notes": "test"
            }])
            m_path = tmp_path / "manifest.csv"
            df.to_csv(m_path, index=False)

            res = audit_dataset(m_path, images_tmp, check_provenance=True)
            self.assertFalse(res["is_valid"])
            self.assertTrue(any("Hard negative pair cannot be labeled related=1" in e for e in res["errors"]))

    def test_07_hard_negative_candidate_cannot_have_transform_direction(self):
        """Hard-negative candidate cannot have transform direction A_TO_B or B_TO_A."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_tmp = tmp_path / "images"
            images_tmp.mkdir()

            src_img = next(self.images_dir.glob("*.jpeg"))
            shutil.copy(src_img, images_tmp / "img_a.jpg")
            shutil.copy(src_img, images_tmp / "img_b.jpg")

            df = pd.DataFrame([{
                "pair_id": "cand_05",
                "image_a": "img_a.jpg",
                "image_b": "img_b.jpg",
                "source_a": "s200",
                "source_b": "s201",
                "related": 0,
                "ground_truth_direction": "A_TO_B",  # Invalid for hard negative!
                "provenance_status": "hard_negative_visual_match",
                "verification_status": "pending_review",
                "consent_confirmed": "confirmed",
                "provenance_chain": "two shots of same scene",
                "uncertainty_flag": 0,
                "notes": "test"
            }])
            m_path = tmp_path / "manifest.csv"
            df.to_csv(m_path, index=False)

            res = audit_dataset(m_path, images_tmp, check_provenance=True)
            self.assertFalse(res["is_valid"])
            self.assertTrue(any("Hard negative pair direction must be 'NONE' or 'UNCERTAIN'" in e for e in res["errors"]))

    def test_08_hard_negative_candidate_requires_justification_for_approval(self):
        """Hard negative candidate marked reviewer_approved without review notes must fail."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_tmp = tmp_path / "images"
            images_tmp.mkdir()

            src_img = next(self.images_dir.glob("*.jpeg"))
            shutil.copy(src_img, images_tmp / "img_a.jpg")
            shutil.copy(src_img, images_tmp / "img_b.jpg")

            df = pd.DataFrame([{
                "pair_id": "cand_06",
                "image_a": "img_a.jpg",
                "image_b": "img_b.jpg",
                "source_a": "s200",
                "source_b": "s201",
                "related": 0,
                "ground_truth_direction": "NONE",
                "provenance_status": "hard_negative_visual_match",
                "verification_status": "reviewer_approved",
                "consent_confirmed": "confirmed",
                "provenance_chain": "two independent captures",
                "uncertainty_flag": 0,
                "notes": ""  # Missing reviewer justification notes!
            }])
            m_path = tmp_path / "manifest.csv"
            df.to_csv(m_path, index=False)

            res = audit_dataset(m_path, images_tmp, check_provenance=True)
            self.assertFalse(res["is_valid"])
            self.assertTrue(any("approved without review notes" in e for e in res["errors"]))

    def test_09_uncertainty_flag_quarantine_enforcement(self):
        """Uncertain pair (uncertainty_flag=1) cannot be marked related=1."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_tmp = tmp_path / "images"
            images_tmp.mkdir()

            src_img = next(self.images_dir.glob("*.jpeg"))
            shutil.copy(src_img, images_tmp / "img_a.jpg")
            shutil.copy(src_img, images_tmp / "img_b.jpg")

            df = pd.DataFrame([{
                "pair_id": "cand_07",
                "image_a": "img_a.jpg",
                "image_b": "img_b.jpg",
                "source_a": "s300",
                "source_b": "s300",
                "related": 1,  # Flagged uncertain cannot be labeled related=1!
                "ground_truth_direction": "UNCERTAIN",
                "provenance_status": "candidate_unverified",
                "verification_status": "quarantined",
                "consent_confirmed": "confirmed",
                "provenance_chain": "unknown source",
                "uncertainty_flag": 1,
                "notes": "quarantined ambiguous lineage"
            }])
            m_path = tmp_path / "manifest.csv"
            df.to_csv(m_path, index=False)

            res = audit_dataset(m_path, images_tmp, check_provenance=True)
            self.assertFalse(res["is_valid"])
            self.assertTrue(any("Pair flagged with uncertainty_flag cannot be labeled related=1" in e for e in res["errors"]))

    def test_10_invalid_provenance_and_verification_statuses_caught(self):
        """Unrecognized provenance or verification status values must trigger errors."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_tmp = tmp_path / "images"
            images_tmp.mkdir()

            src_img = next(self.images_dir.glob("*.jpeg"))
            shutil.copy(src_img, images_tmp / "img_a.jpg")
            shutil.copy(src_img, images_tmp / "img_b.jpg")

            df = pd.DataFrame([{
                "pair_id": "cand_08",
                "image_a": "img_a.jpg",
                "image_b": "img_b.jpg",
                "source_a": "s400",
                "source_b": "s400",
                "related": 1,
                "ground_truth_direction": "A_TO_B",
                "provenance_status": "bogus_provenance",
                "verification_status": "bogus_verification",
                "consent_confirmed": "confirmed",
                "provenance_chain": "chain",
                "uncertainty_flag": 0,
                "notes": "contributor-confirmed test"
            }])
            m_path = tmp_path / "manifest.csv"
            df.to_csv(m_path, index=False)

            res = audit_dataset(m_path, images_tmp, check_provenance=True)
            self.assertFalse(res["is_valid"])
            self.assertTrue(any("Invalid provenance_status 'bogus_provenance'" in e for e in res["errors"]))
            self.assertTrue(any("Invalid verification_status 'bogus_verification'" in e for e in res["errors"]))

    def test_11_candidate_workspace_template_schema_conformance(self):
        """Candidate workspace template must include all required provenance columns."""
        template_file = self.workspace_dir / "candidates_manifest.template.csv"
        self.assertTrue(template_file.exists(), f"Template not found: {template_file}")

        df = pd.read_csv(template_file)
        for col in REQUIRED_PROVENANCE_COLUMNS:
            self.assertIn(col, df.columns, f"Candidate template missing provenance column: {col}")
        self.assertIn("uncertainty_flag", df.columns)
        self.assertIn("consent_confirmed", df.columns)
        self.assertIn("provenance_chain", df.columns)
        self.assertEqual(len(df), 3, "Template should contain 3 representative illustrative candidate rows.")

    def test_12_production_safeguards_preserved(self):
        """Production configuration must keep ENABLE_PHASE4_MODELS disabled."""
        self.assertFalse(ENABLE_PHASE4_MODELS)


if __name__ == "__main__":
    unittest.main()
