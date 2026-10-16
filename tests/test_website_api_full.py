"""
tests/test_website_api_full.py - Complete API & Web Server Verification Test Suite using unittest
"""

import os
import io
import sys
import json
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from backend.app.main import app

class TestWebsiteApiFull(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.root_dir = Path(__file__).resolve().parent.parent
        cls.test_img_dir = cls.root_dir / "dataset/images/test"
        cls.sample_root = cls.test_img_dir / "src_0001_checker_16_navymint_root.jpg"
        cls.sample_child = cls.test_img_dir / "src_0001_checker_16_navymint_gen1_n02.jpg"

    def test_01_health_endpoint(self):
        """Verify system health, DB connection, and model loading status."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200, f"Health check failed: {response.text}")
        data = response.json()
        self.assertEqual(data["status"], "ONLINE")
        self.assertEqual(data["database"], "CONNECTED")
        self.assertTrue(data["sklearn_pipeline_loaded"])
        self.assertIn("selected_relationship_classifier.pkl", data["active_model"])

    def test_02_model_info_endpoint(self):
        """Verify model specifications returned to frontend."""
        response = self.client.get("/api/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("ViT-B/16", data["embedder"])
        self.assertTrue(data["scikit_learn_pipeline"]["detected"])
        self.assertEqual(data["scikit_learn_pipeline"]["pipeline_type"], "StandardScaler + MLPClassifier")
        self.assertGreaterEqual(len(data["supported_transformations"]), 10)

    def test_03_model_metrics_endpoint(self):
        """Verify empirical metrics endpoint without fabrications."""
        response = self.client.get("/api/model-metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        metrics = data["selected_relationship_model"]["test_split_metrics"]
        self.assertEqual(metrics["evaluation_samples"], 96)
        self.assertEqual(metrics["accuracy"], 0.8542)
        self.assertEqual(metrics["precision"], 1.0000)

    def test_04_upload_disallowed_extension(self):
        """Verify rejection of invalid file types."""
        dummy_txt = io.BytesIO(b"Hello forensic text")
        response = self.client.post(
            "/api/upload",
            files={"files": ("malicious.txt", dummy_txt, "text/plain")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("No valid image files", response.json()["detail"])

    def test_05_pairwise_prediction_api(self):
        """Verify live pairwise prediction endpoint with real image paths."""
        response = self.client.post(
            "/api/predict-pair",
            json={
                "image_a_path": str(self.sample_root),
                "image_b_path": str(self.sample_child)
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction", data)
        pred = data["prediction"]
        self.assertTrue(pred["related"])
        self.assertEqual(pred["direction"], "A_TO_B")
        self.assertEqual(pred["transformation"], "perspective")
        self.assertIn("feature_sensitivity", data)

    def test_06_full_analysis_and_graph_generation(self):
        """Verify full end-to-end analysis including graph, XAI, DB and HTML report."""
        # 1. Upload real images
        with open(self.sample_root, "rb") as f1, open(self.sample_child, "rb") as f2:
            upload_resp = self.client.post(
                "/api/upload",
                files=[
                    ("files", ("sample_root.jpg", f1, "image/jpeg")),
                    ("files", ("sample_child.jpg", f2, "image/jpeg"))
                ]
            )
        self.assertEqual(upload_resp.status_code, 200)
        analysis_id = upload_resp.json()["analysis_id"]

        # 2. Trigger analysis
        analyze_resp = self.client.post(
            "/api/analyze",
            json={
                "analysis_id": analysis_id,
                "similarity_threshold": 0.50,
                "confidence_filter": 0.50
            }
        )
        self.assertEqual(analyze_resp.status_code, 200)
        data = analyze_resp.json()
        self.assertEqual(data["status"], "COMPLETED")
        self.assertEqual(data["image_count"], 2)
        self.assertGreaterEqual(data["edge_count"], 1)
        self.assertEqual(data["root_image_id"], "sample_root")
        self.assertGreaterEqual(len(data["xai_results"]), 1)

        # 3. Verify Database Retrieval
        hist_resp = self.client.get("/api/history")
        self.assertEqual(hist_resp.status_code, 200)
        records = hist_resp.json()
        self.assertIsInstance(records, list)
        found = any(r["id"] == analysis_id for r in records)
        self.assertTrue(found, f"Analysis {analysis_id} not found in DB history")

        # 4. Verify Detailed Analysis Retrieval
        detail_resp = self.client.get(f"/api/analysis/{analysis_id}")
        self.assertEqual(detail_resp.status_code, 200)
        detail = detail_resp.json()
        self.assertEqual(detail["analysis"]["id"], analysis_id)
        self.assertEqual(detail["analysis"]["root_image_id"], "sample_root")

        # 5. Verify HTML Report Exists & Downloadable
        report_resp = self.client.get(f"/api/download/{analysis_id}/html")
        self.assertEqual(report_resp.status_code, 200)
        self.assertIn("TRACEAI FORENSIC REPORT", report_resp.text.upper())

    def test_07_error_handling(self):
        """Verify appropriate HTTP status codes for edge and error cases."""
        # Non-existent session
        resp = self.client.post("/api/analyze", json={"analysis_id": "non_existent_12345"})
        self.assertEqual(resp.status_code, 404)

        # Non-existent image in predict-pair
        resp2 = self.client.post(
            "/api/predict-pair",
            json={"image_a_path": "non_existent_a.jpg", "image_b_path": "non_existent_b.jpg"}
        )
        self.assertEqual(resp2.status_code, 404)


if __name__ == "__main__":
    unittest.main()
