"""
tests/test_pipeline.py — End-to-End Integration Test for TRACEAI Provenance Engine & REST API
"""

import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.main import app
from backend.app.config import DATASET_DIR, UPLOAD_DIR

class TestProvenancePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_check(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ONLINE")
        self.assertTrue(data["model_loaded"])

    def test_02_model_info(self):
        response = self.client.get("/api/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("ViT-B/16", data["embedder"])
        self.assertIn("crop", data["supported_transformations"])

    def test_03_dataset_endpoint(self):
        response = self.client.get("/api/dataset")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_original_images"], 100)

    def test_04_demo_analysis_execution(self):
        response = self.client.post("/api/demo?graph_id=graph_0001")
        if response.status_code != 200:
            print("ERROR DETAIL:", response.text)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["status"], "COMPLETED")
        self.assertIsNotNone(data["root_image_id"])
        self.assertGreater(data["image_count"], 1)
        self.assertGreater(data["edge_count"], 0)
        self.assertGreater(len(data["xai_results"]), 0)

        analysis_id = data["analysis_id"]

        # Verify graph endpoint
        graph_resp = self.client.get(f"/api/analysis/{analysis_id}/graph")
        self.assertEqual(graph_resp.status_code, 200)
        cy_data = graph_resp.json()
        self.assertIn("cytoscape_elements", cy_data)

        # Verify explanation endpoint
        xai_resp = self.client.get(f"/api/analysis/{analysis_id}/explanation")
        self.assertEqual(xai_resp.status_code, 200)

        # Verify HTML report download
        report_resp = self.client.get(f"/api/download/{analysis_id}/html")
        self.assertEqual(report_resp.status_code, 200)
        self.assertIn("TRACEAI FORENSIC REPORT", report_resp.text)

if __name__ == "__main__":
    unittest.main()
