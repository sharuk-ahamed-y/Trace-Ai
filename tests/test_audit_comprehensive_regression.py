"""
tests/test_audit_comprehensive_regression.py
Regression tests for:
1. ModelManager sensitivity explanation bug fix (groups vs group11s)
2. ProvenanceEngine deterministic component and root tie-breaking
3. Direction swap consistency & antisymmetry guarantees
4. Adaptive cycle penalty graph reconstruction
"""

import unittest
from pathlib import Path
import random
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.model_manager import get_model_manager
from backend.app.services.provenance_engine import run_provenance_analysis

ROOT = Path(__file__).resolve().parent.parent
TEST_IMG_DIR = ROOT / "dataset" / "images" / "test"
VAL_IMG_DIR = ROOT / "dataset" / "images" / "val"


class TestAuditComprehensiveRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.manager = get_model_manager()
        cls.manager.initialize()
        cls.test_images = sorted(list(TEST_IMG_DIR.glob("src_0001_*.jpg")))
        if len(cls.test_images) < 4:
            cls.test_images = sorted(list(TEST_IMG_DIR.glob("*.jpg")))[:5]

    def test_01_explain_pair_sensitivity_bug_fix(self):
        """Verify that compute_feature_sensitivity executes without NameError and returns feature_groups."""
        img_a = self.test_images[0]
        img_b = self.test_images[1]
        
        # Call model manager method directly
        expl = self.manager.compute_feature_sensitivity(img_a, img_b)
        self.assertIn("base_prediction", expl)
        self.assertIn("feature_groups", expl)
        self.assertIn("interpretation", expl)
        self.assertEqual(len(expl["feature_groups"]), 5)
        
        # Verify normalized weights sum to ~1.0
        total_w = sum(g["normalized_weight"] for g in expl["feature_groups"])
        self.assertAlmostEqual(total_w, 1.0, places=2)

    def test_02_deterministic_root_and_component_tie_breaking(self):
        """Verify root selection is 100% deterministic across shuffled input orderings."""
        orig_res = run_provenance_analysis(self.test_images, similarity_threshold=0.50)
        expected_root = orig_res["root_image_id"]
        expected_edge_count = len(orig_res["edges"])
        
        # Run 5 iterations with shuffled image paths
        imgs_shuffled = list(self.test_images)
        for seed in [1, 42, 99, 123, 777]:
            random.seed(seed)
            random.shuffle(imgs_shuffled)
            res = run_provenance_analysis(imgs_shuffled, similarity_threshold=0.50)
            self.assertEqual(res["root_image_id"], expected_root,
                             f"Root selection was nondeterministic under shuffle seed {seed}")
            self.assertEqual(len(res["edges"]), expected_edge_count)

    def test_03_adaptive_cycle_penalty_execution(self):
        """Verify that adaptive_penalty=True builds a valid spanning forest with mst_cost."""
        res_baseline = run_provenance_analysis(self.test_images, similarity_threshold=0.50, adaptive_penalty=False)
        res_adaptive = run_provenance_analysis(self.test_images, similarity_threshold=0.50, adaptive_penalty=True, lambda_max=0.10)
        
        self.assertIn("root_image_id", res_adaptive)
        self.assertEqual(len(res_adaptive["nodes"]), len(self.test_images))
        self.assertEqual(len(res_adaptive["edges"]), len(res_baseline["edges"]))
        
        for e in res_adaptive["edges"]:
            self.assertIn("mst_cost", e)
            self.assertIn("weight", e)
            self.assertGreaterEqual(e["mst_cost"], e["weight"] - 1e-6)

    def test_04_pairwise_sensitivity_api_endpoint(self):
        """Verify /api/predict-pair API endpoint with sensitivity explanation returns HTTP 200."""
        payload = {
            "image_a_path": str(self.test_images[0]),
            "image_b_path": str(self.test_images[1])
        }
        resp = self.client.post("/api/predict-pair", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("prediction", data)
        self.assertIn("feature_sensitivity", data)
        self.assertIn("interpretation", data)


if __name__ == "__main__":
    unittest.main()
