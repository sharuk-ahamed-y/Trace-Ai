"""
tests/test_confidence_filter_regression.py
Regression tests verifying the two-stage candidate edge filtering in provenance_engine:
  Stage 1: P(related) >= similarity_threshold
  Stage 2: P(related) * P(direction) >= confidence_filter
"""

import unittest
from pathlib import Path
import networkx as nx
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.provenance_engine import run_provenance_analysis
from backend.app.services.model_manager import get_model_manager

ROOT = Path(__file__).resolve().parent.parent
TEST_IMG_DIR = ROOT / "dataset" / "images" / "test"


class TestConfidenceFilterRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.manager = get_model_manager()
        cls.manager.initialize()

        # Find available test images from graph_0001
        cls.test_images = list(TEST_IMG_DIR.glob("src_0001_*.jpg"))
        if len(cls.test_images) < 4:
            cls.test_images = list(TEST_IMG_DIR.glob("*.jpg"))[:5]
        assert len(cls.test_images) >= 2, "Need at least 2 test images for regression testing"

    def test_01_edges_below_similarity_threshold_are_rejected(self):
        """Pairs with P(related) < similarity_threshold must be rejected even if confidence_filter is low."""
        # Extremely high similarity threshold: 0.999
        res = run_provenance_analysis(
            self.test_images,
            similarity_threshold=0.999,
            confidence_filter=0.10
        )
        self.assertEqual(res["summary"]["candidate_edges"], 0)
        self.assertEqual(res["summary"]["forest_edges"], 0)
        n = len(self.test_images)
        total_pairs = n * (n - 1) // 2
        self.assertEqual(res["summary"]["rejected_pairs"], total_pairs)

    def test_02_edges_below_confidence_filter_are_rejected(self):
        """Pairs meeting similarity_threshold but failing confidence_filter must be rejected."""
        # Standard similarity_threshold=0.50, but unattainable confidence_filter=0.999
        res = run_provenance_analysis(
            self.test_images,
            similarity_threshold=0.50,
            confidence_filter=0.999
        )
        self.assertEqual(res["summary"]["candidate_edges"], 0)
        self.assertEqual(res["summary"]["forest_edges"], 0)
        n = len(self.test_images)
        total_pairs = n * (n - 1) // 2
        self.assertEqual(res["summary"]["rejected_pairs"], total_pairs)

    def test_03_edges_meeting_both_conditions_are_admitted(self):
        """Pairs meeting BOTH similarity_threshold and confidence_filter must be admitted into candidate_edges."""
        # Baseline with permissive thresholds: both 0.50
        res_permissive = run_provenance_analysis(
            self.test_images,
            similarity_threshold=0.50,
            confidence_filter=0.50
        )
        candidates_admitted = res_permissive["summary"]["candidate_edges"]
        self.assertGreater(candidates_admitted, 0)

        for edge in res_permissive["candidate_edges"]:
            self.assertGreaterEqual(edge["related_prob"], 0.50)
            self.assertGreaterEqual(edge["confidence"], 0.50)

        # Stricter confidence filter (0.80) should admit fewer or equal candidates than 0.50
        res_stricter = run_provenance_analysis(
            self.test_images,
            similarity_threshold=0.50,
            confidence_filter=0.80
        )
        self.assertLessEqual(
            res_stricter["summary"]["candidate_edges"],
            candidates_admitted
        )
        for edge in res_stricter["candidate_edges"]:
            self.assertGreaterEqual(edge["related_prob"], 0.50)
            self.assertGreaterEqual(edge["confidence"], 0.80)

    def test_04_api_parameters_and_defaults_work(self):
        """Verify POST /api/analyze accepts and processes confidence_filter parameter and defaults."""
        # Upload 2 test images
        img_a = self.test_images[0]
        img_b = self.test_images[1]

        with open(img_a, "rb") as fa, open(img_b, "rb") as fb:
            upload_res = self.client.post(
                "/api/upload",
                files=[
                    ("files", (img_a.name, fa, "image/jpeg")),
                    ("files", (img_b.name, fb, "image/jpeg"))
                ]
            )
        self.assertEqual(upload_res.status_code, 200)
        analysis_id = upload_res.json()["analysis_id"]

        # 1. Explicit confidence_filter in request
        resp_explicit = self.client.post(
            "/api/analyze",
            json={
                "analysis_id": analysis_id,
                "similarity_threshold": 0.50,
                "confidence_filter": 0.40
            }
        )
        self.assertEqual(resp_explicit.status_code, 200)
        data = resp_explicit.json()
        self.assertEqual(data["status"], "COMPLETED")
        self.assertEqual(data["image_count"], 2)

        # 2. Omitting confidence_filter uses default safely
        resp_default = self.client.post(
            "/api/analyze",
            json={
                "analysis_id": analysis_id
            }
        )
        self.assertEqual(resp_default.status_code, 200)
        self.assertEqual(resp_default.json()["status"], "COMPLETED")

    def test_05_no_self_loops_or_directed_cycles(self):
        """Verify reconstructed graphs never contain self-loops or directed cycles."""
        for c_val in [0.30, 0.50, 0.70]:
            res = run_provenance_analysis(
                self.test_images,
                similarity_threshold=0.50,
                confidence_filter=c_val
            )
            edges = res["edges"]
            # No self-loops
            for e in edges:
                self.assertNotEqual(e["source"], e["target"])

            # No directed cycles
            DG = nx.DiGraph()
            for n in res["nodes"]:
                DG.add_node(n["id"])
            for e in edges:
                DG.add_edge(e["source"], e["target"])
            self.assertTrue(nx.is_directed_acyclic_graph(DG))


if __name__ == "__main__":
    unittest.main()
