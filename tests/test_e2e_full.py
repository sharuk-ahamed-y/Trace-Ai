"""
tests/test_e2e_full.py — Exhaustive End-to-End Test Suite for Reconstructed TRACEAI Platform
Validates:
1. Model loading & hardware telemetry (/api/health, /api/model-info)
2. Image directory resolution (/api/images)
3. Real 115-D Pairwise prediction & transformation (/api/predict-pair)
4. Full Graph Reconstruction Pipeline (/api/demo)
5. Component-aware Minimum Spanning Forest (/api/analysis/{id}/forest)
6. Feature Sensitivity & Ablation (/api/analysis/{id}/sensitivity)
7. Explainable AI Heatmap & Decision Evidence (/api/analysis/{id}/explanation)
8. Cytoscape DAG Canvas Serialization (/api/analysis/{id}/graph)
9. HTML Investigation Dossier Export (/api/download/{id}/html)
10. SQLite Database Persistence (/api/history)
"""

import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.main import app


class TestTRACEAIFullSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_and_telemetry(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ONLINE")
        self.assertTrue(data["model_loaded"])
        self.assertIn("cpu_usage_percent", data)
        self.assertIn("memory_usage_percent", data)
        self.assertIn("active_model", data)

    def test_02_model_info_representation(self):
        resp = self.client.get("/api/model-info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("ViT-B/16", data["embedder"])
        self.assertIn("scikit_learn_pipeline", data)
        self.assertIn("115-dimensional", data["scikit_learn_pipeline"]["feature_representation"])
        self.assertIn("crop", data["supported_transformations"])
        self.assertIn("methodology_note", data)

    def test_03_available_images(self):
        resp = self.client.get("/api/images")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("images", data)
        self.assertGreater(len(data["images"]), 1)

    def test_04_pairwise_inference(self):
        imgs_resp = self.client.get("/api/images")
        images = imgs_resp.json()["images"]
        self.assertGreaterEqual(len(images), 2)

        img_a = images[0]["id"]
        img_b = images[1]["id"]

        resp = self.client.post("/api/predict-pair", json={
            "image_a_path": img_a,
            "image_b_path": img_b
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("relationship", data)
        self.assertIn("direction", data)
        self.assertIn("transformation", data)
        self.assertIn("features_summary", data)
        self.assertGreaterEqual(data["relationship"]["probability"], 0.0)
        self.assertLessEqual(data["relationship"]["probability"], 1.0)

    def test_05_reconstruction_pipeline_and_forest(self):
        # Run demo on benchmark graph_0001
        resp = self.client.post("/api/demo?graph_id=graph_0001")
        self.assertEqual(resp.status_code, 200)
        analysis = resp.json()
        self.assertEqual(analysis["status"], "COMPLETED")
        self.assertIsNotNone(analysis["root_image_id"])
        self.assertGreater(analysis["image_count"], 1)
        self.assertGreater(analysis["edge_count"], 0)

        analysis_id = analysis["analysis_id"]

        # Verify forest endpoint
        forest_resp = self.client.get(f"/api/analysis/{analysis_id}/forest")
        self.assertEqual(forest_resp.status_code, 200)
        forest_data = forest_resp.json()
        self.assertIn("forest_edges", forest_data)
        self.assertIn("weight_semantics", forest_data)
        self.assertIn("d = 1 - P(related)", forest_data["weight_semantics"])
        self.assertIn("provisional_tree_notice", forest_data)

        # Verify sensitivity endpoint
        sens_resp = self.client.get(f"/api/analysis/{analysis_id}/sensitivity")
        self.assertEqual(sens_resp.status_code, 200)

        # Verify XAI explanations & heatmaps
        xai_resp = self.client.get(f"/api/analysis/{analysis_id}/explanation")
        self.assertEqual(xai_resp.status_code, 200)
        xai_data = xai_resp.json()
        self.assertIn("xai_results", xai_data)
        if len(xai_data["xai_results"]) > 0:
            first_xai = xai_data["xai_results"][0]
            self.assertIn("spatial_metrics", first_xai)
            self.assertIn("evidence", first_xai)

        # Verify Cytoscape graph canvas elements
        graph_resp = self.client.get(f"/api/analysis/{analysis_id}/graph")
        self.assertEqual(graph_resp.status_code, 200)
        g_data = graph_resp.json()
        self.assertIn("cytoscape_elements", g_data)
        self.assertGreater(len(g_data["cytoscape_elements"]), 0)

        # Verify HTML dossier generation
        html_resp = self.client.get(f"/api/download/{analysis_id}/html")
        self.assertEqual(html_resp.status_code, 200)
        self.assertIn("text/html", html_resp.headers.get("content-type", ""))
        self.assertIn("TRACEAI FORENSIC REPORT", html_resp.text)

    def test_06_history_persistence(self):
        resp = self.client.get("/api/history")
        self.assertEqual(resp.status_code, 200)
        history = resp.json()
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)


if __name__ == "__main__":
    unittest.main()
