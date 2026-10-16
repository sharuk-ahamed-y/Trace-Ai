"""
tests/test_sklearn_models.py — Dedicated Scikit-Learn Model Integration & Inference Test Suite

Verifies:
1. Presence of original Colab .pkl files in models/
2. Direct loading of original .pkl files (Pipeline / Dict wrappers)
3. Dimensional compatibility (115-D relationship, 172-D direction, 20-D transformation)
4. Valid output classes and non-fabricated probabilities
5. ModelManager prioritization of Scikit-Learn pipeline over PyTorch baseline
6. Backend REST API endpoints (/api/health, /api/model-info, /api/predict-pair)
   actively using the Scikit-Learn pipeline for live inference.
"""

import sys
import unittest
import pickle
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

MODELS_DIR = ROOT_DIR / "models"
REL_PKL_CANDIDATES = [
    MODELS_DIR / "selected_relationship_classifier.pkl",
    MODELS_DIR / "final_relationship_classifier.pkl",
    MODELS_DIR / "relationship_classifier.pkl"
]
DIR_PKL_CANDIDATES = [
    MODELS_DIR / "direction_classifier.pkl",
    MODELS_DIR / "selected_direction_classifier.pkl"
]
TFM_PKL_CANDIDATES = [
    MODELS_DIR / "transformation_classifier_top20.pkl",
    MODELS_DIR / "transformation_classifier.pkl"
]


def _load_model_file(model_path):
    try:
        import joblib
        return joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            return pickle.load(f)


class TestOriginalSklearnIntegration(unittest.TestCase):
    def test_01_check_original_pkl_files_presence(self):
        """Checks whether original Google Colab .pkl files exist locally."""
        found_rel = [p for p in REL_PKL_CANDIDATES if p.exists()]
        found_dir = [p for p in DIR_PKL_CANDIDATES if p.exists()]
        found_tfm = [p for p in TFM_PKL_CANDIDATES if p.exists()]

        print("\n" + "=" * 60)
        print("  ORIGINAL SKLEARN MODEL FILES PRESENCE AUDIT")
        print("=" * 60)
        print(f"Relationship Classifier Candidates: {[p.name for p in REL_PKL_CANDIDATES]}")
        print(f"  -> Found: {[p.name for p in found_rel] if found_rel else 'None (Awaiting Colab Transfer)'}")
        print(f"Direction Classifier Candidates: {[p.name for p in DIR_PKL_CANDIDATES]}")
        print(f"  -> Found: {[p.name for p in found_dir] if found_dir else 'None (Awaiting Colab Transfer)'}")
        print(f"Transformation Classifier Candidates: {[p.name for p in TFM_PKL_CANDIDATES]}")
        print(f"  -> Found: {[p.name for p in found_tfm] if found_tfm else 'None (Awaiting Colab Transfer)'}")

        self.assertTrue(len(found_rel) > 0, "No relationship classifier found.")
        self.assertTrue(len(found_dir) > 0, "No direction classifier found.")
        self.assertTrue(len(found_tfm) > 0, "No transformation classifier found.")

    def test_02_load_relationship_classifier(self):
        """Directly loads the original relationship classifier and verifies 115-D inference."""
        found_rel = [p for p in REL_PKL_CANDIDATES if p.exists()]
        self.assertTrue(len(found_rel) > 0, "No relationship classifier .pkl present.")

        model_path = found_rel[0]
        obj = _load_model_file(model_path)

        model = obj.get("model", obj) if isinstance(obj, dict) else obj
        self.assertTrue(hasattr(model, "predict") or hasattr(model, "predict_proba"))

        # Verify dimension
        in_dim = getattr(model, "n_features_in_", None)
        if in_dim is None and hasattr(model, "steps"):
            for _, step in model.steps:
                if hasattr(step, "n_features_in_"):
                    in_dim = step.n_features_in_
                    break
        self.assertEqual(in_dim, 115, f"Expected 115-D relationship input, got {in_dim}")

        # Run inference with compatible 115-D vector
        vec_115 = np.zeros((1, 115), dtype=np.float32)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vec_115)[0]
            self.assertEqual(len(probs), 2)
            self.assertAlmostEqual(sum(probs), 1.0, places=4)
            print(f"[TEST PASS] {model_path.name} predict_proba(115-D) -> {probs}")
        else:
            pred = model.predict(vec_115)[0]
            print(f"[TEST PASS] {model_path.name} predict(115-D) -> {pred}")

    def test_03_load_direction_classifier(self):
        """Directly loads the original direction classifier and verifies 172-D inference."""
        found_dir = [p for p in DIR_PKL_CANDIDATES if p.exists()]
        self.assertTrue(len(found_dir) > 0, "No direction classifier .pkl present.")

        model_path = found_dir[0]
        obj = _load_model_file(model_path)

        model = obj.get("model", obj) if isinstance(obj, dict) else obj
        self.assertTrue(hasattr(model, "predict"))

        in_dim = getattr(model, "n_features_in_", None)
        if in_dim is None and hasattr(model, "steps"):
            for _, step in model.steps:
                if hasattr(step, "n_features_in_"):
                    in_dim = step.n_features_in_
                    break
        self.assertEqual(in_dim, 172, f"Expected 172-D direction input, got {in_dim}")

        vec_172 = np.zeros((1, 172), dtype=np.float32)
        pred = model.predict(vec_172)[0]
        self.assertIn(pred, ["A_TO_B", "B_TO_A", 0, 1])
        print(f"[TEST PASS] {model_path.name} predict(172-D) -> {pred}")

    def test_04_load_transformation_classifier(self):
        """Directly loads transformation classifier and verifies top 20 feature selection."""
        found_tfm = [p for p in TFM_PKL_CANDIDATES if p.exists()]
        self.assertTrue(len(found_tfm) > 0, "No transformation classifier .pkl present.")

        model_path = found_tfm[0]
        obj = _load_model_file(model_path)

        if isinstance(obj, dict):
            model = obj.get("model", obj.get("classifier", obj))
            selected = None
            for k in ["selected_feature_indices", "selected_features", "top_features", "feature_indices"]:
                if k in obj and obj[k] is not None:
                    selected = obj[k]
                    break
            req_dim = len(selected) if selected is not None else getattr(model, "n_features_in_", 20)
        else:
            model = obj
            req_dim = getattr(model, "n_features_in_", 20)

        self.assertEqual(req_dim, 20, f"Expected 20-D transformation input, got {req_dim}")
        vec = np.zeros((1, 20), dtype=np.float32)
        pred = model.predict(vec)[0]
        print(f"[TEST PASS] {model_path.name} predict(20-D) -> {pred}")

    def test_05_backend_model_manager_active_model(self):
        """Tests that backend ModelManager prioritizes the scikit-learn pipeline."""
        from backend.app.services.model_manager import get_model_manager
        # Force fresh initialization
        manager = get_model_manager()
        manager.initialized = False
        manager.initialize()
        status = manager.get_status()

        self.assertTrue(status["sklearn_pipeline_detected"])
        self.assertTrue(status["active_relationship_model"].startswith("Scikit-learn"))
        print(f"[TEST PASS] Active Model in Backend: {status['active_relationship_model']}")

    def test_06_fastapi_endpoints_use_sklearn_when_present(self):
        """Tests that /api/health and /api/predict-pair live endpoints report and use the sklearn pipeline."""
        from fastapi.testclient import TestClient
        from backend.app.main import app

        client = TestClient(app)
        health_resp = client.get("/api/health")
        self.assertEqual(health_resp.status_code, 200)
        health_data = health_resp.json()

        self.assertTrue(health_data["sklearn_pipeline_loaded"])
        self.assertIn("Scikit-learn", health_data["active_model"])
        print(f"[TEST PASS] Health Endpoint Active Model: {health_data['active_model']}")

        # Test live pairwise prediction on sample images
        imgs_resp = client.get("/api/images")
        self.assertEqual(imgs_resp.status_code, 200)
        images = imgs_resp.json().get("images", [])
        if len(images) >= 2:
            pair_resp = client.post("/api/predict-pair", json={
                "image_a_path": images[0]["id"],
                "image_b_path": images[1]["id"]
            })
            self.assertEqual(pair_resp.status_code, 200)
            pair_data = pair_resp.json()
            active_m = pair_data["features_summary"]["active_model"]
            self.assertIn("Scikit-learn", active_m)
            self.assertEqual(pair_data["features_summary"]["feature_dim"], 115)
            self.assertIn(pair_data["direction"]["hypothesis"], ["A_TO_B", "B_TO_A"])
            print(f"[TEST PASS] Live Predict Pair Active Model: {active_m}")
            print(f"[TEST PASS] Predicted Relationship Probability: {pair_data['relationship']['probability']}")
            print(f"[TEST PASS] Predicted Direction: {pair_data['direction']['hypothesis']}")
            print(f"[TEST PASS] Predicted Transformation: {pair_data['transformation']['class']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
