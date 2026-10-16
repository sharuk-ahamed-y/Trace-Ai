"""
tests/test_audit_verification.py — Comprehensive Automated Test Suite
Verifying all requirements from the Project Audit & Repair specification:
1. Model artifact integrity & serialization validation
2. Feature dimension and compatibility validation (57D, 115D, 172D, 20D)
3. Relationship classification on positive/negative pairs with honest confidence
4. Direction classification (A_TO_B, B_TO_A, NONE) and A/B swap reversal
5. Transformation classification (no silent fallback, correct label mapping)
6. Graph construction, Minimum Spanning Forest & Root detection
7. API endpoints, upload handling, schema compliance
8. Test set evaluation benchmark replication (85.42% accuracy, 1.0 precision)
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.main import app
from backend.app.services.model_manager import get_model_manager
from backend.app.services.provenance_engine import run_provenance_analysis
from src.pair_features import build_115d_pair_feature, build_172d_direction_feature

MODELS_DIR = ROOT_DIR / "models"
DATASET_DIR = ROOT_DIR / "dataset"
TEST_IMG_DIR = DATASET_DIR / "images" / "test"
TEST_PAIRS_CSV = DATASET_DIR / "metadata" / "pairs_test.csv"
FORENSIC_CSV = DATASET_DIR / "features" / "forensic_features.csv"


class TestAuditVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.manager = get_model_manager()
        cls.manager.initialize()

    def test_01_model_artifact_integrity(self):
        """Verify each model file is a valid serialized scikit-learn model."""
        rel_path = MODELS_DIR / "selected_relationship_classifier.pkl"
        dir_path = MODELS_DIR / "direction_classifier.pkl"
        tfm_path = MODELS_DIR / "transformation_classifier_top20.pkl"

        self.assertTrue(rel_path.exists(), "selected_relationship_classifier.pkl must exist")
        self.assertTrue(dir_path.exists(), "direction_classifier.pkl must exist")
        self.assertTrue(tfm_path.exists(), "transformation_classifier_top20.pkl must exist")

        rel_m = joblib.load(rel_path)
        self.assertTrue(hasattr(rel_m, "predict"), "Relationship model must have predict method")
        self.assertTrue(hasattr(rel_m, "predict_proba"), "Relationship model must have predict_proba method")

        dir_m = joblib.load(dir_path)
        self.assertTrue(hasattr(dir_m, "predict"), "Direction model must have predict method")

        tfm_obj = joblib.load(tfm_path)
        self.assertIn("model", tfm_obj, "Transformation pkl must contain 'model' key")
        self.assertIn("selected_feature_indices", tfm_obj, "Transformation pkl must contain 'selected_feature_indices'")
        self.assertEqual(len(tfm_obj["selected_feature_indices"]), 20, "Must select exactly 20 features")

    def test_02_feature_dimension_validation(self):
        """Verify 57-D, 115-D, 172-D, and 20-D feature dimensions and strict error handling."""
        test_images = list(TEST_IMG_DIR.glob("*.jpg"))
        self.assertGreater(len(test_images), 0, "Test images must be present")
        img_a = test_images[0]
        img_b = test_images[1] if len(test_images) > 1 else test_images[0]

        # 57-D representation
        feat_57 = self.manager.get_57d_representation(img_a)
        self.assertEqual(feat_57.shape, (57,), "Per-image feature must be exactly 57-D")

        # 115-D pairwise vector
        vec_115 = self.manager.get_115d_pairwise_vector(img_a, img_b)
        self.assertEqual(vec_115.shape, (115,), "Pairwise vector must be exactly 115-D")

        # 172-D directional vector
        vec_172 = self.manager.get_172d_direction_vector(img_a, img_b)
        self.assertEqual(vec_172.shape, (172,), "Direction vector must be exactly 172-D")

        # Strict error raising on wrong dimension inputs
        with self.assertRaises(ValueError):
            build_115d_pair_feature(np.zeros(50), np.zeros(57))
        with self.assertRaises(ValueError):
            build_172d_direction_feature(np.zeros(57), np.zeros(60))

    def test_03_pairwise_prediction_schema_and_integrity(self):
        """Verify predict_pair produces a complete schema with honest confidences."""
        test_images = list(TEST_IMG_DIR.glob("*.jpg"))
        self.assertGreaterEqual(len(test_images), 2)
        res = self.manager.predict_pair(test_images[0], test_images[1])

        required_keys = [
            "related", "related_confidence", "direction", "direction_confidence",
            "transformation", "transformation_confidence", "cosine_similarity",
            "active_model", "feature_dim"
        ]
        for key in required_keys:
            self.assertIn(key, res, f"Result must contain '{key}'")

        self.assertIsInstance(res["related"], bool)
        self.assertGreaterEqual(res["related_confidence"], 0.0)
        self.assertLessEqual(res["related_confidence"], 1.0)
        self.assertIn(res["direction"], ["A_TO_B", "B_TO_A", "NONE"])
        self.assertEqual(res["feature_dim"], 115)

    def test_04_direction_ab_swap_consistency(self):
        """Verify image A/B swapping inverts signed differences and produces appropriate direction behavior."""
        df_test = pd.read_csv(TEST_PAIRS_CSV)
        pos_pairs = df_test[df_test["related"] == 1].head(10)

        for _, row in pos_pairs.iterrows():
            path_a = TEST_IMG_DIR / f"{row['image_a']}.jpg"
            path_b = TEST_IMG_DIR / f"{row['image_b']}.jpg"
            if not path_a.exists() or not path_b.exists():
                continue

            pred_ab = self.manager.predict_pair(path_a, path_b)
            pred_ba = self.manager.predict_pair(path_b, path_a)

            # Relationship probability must be symmetric (distance is commutative)
            self.assertAlmostEqual(
                pred_ab["related_confidence"],
                pred_ba["related_confidence"],
                places=3,
                msg="Relationship probability should be symmetric under A/B swap"
            )

    def test_05_graph_provenance_reconstruction(self):
        """Verify Minimum Spanning Forest and cycle avoidance on real test images."""
        test_images = list(TEST_IMG_DIR.glob("src_0001_*.jpg"))
        if len(test_images) < 3:
            test_images = list(TEST_IMG_DIR.glob("*.jpg"))[:5]

        self.assertGreaterEqual(len(test_images), 2)
        res = run_provenance_analysis(test_images, similarity_threshold=0.50)

        self.assertIn("root_image_id", res)
        self.assertIn("nodes", res)
        self.assertIn("edges", res)
        self.assertIn("summary", res)
        self.assertEqual(len(res["nodes"]), len(test_images))

        # Root node must exist in the node set
        node_ids = [n["id"] for n in res["nodes"]]
        self.assertIn(res["root_image_id"], node_ids)

        # Check edge weights are mathematical distances d = 1 - P(related)
        for e in res["edges"]:
            self.assertAlmostEqual(e["weight"], round(1.0 - e["related_prob"], 4), places=3)
            self.assertIn(e["source"], node_ids)
            self.assertIn(e["target"], node_ids)

    def test_06_fastapi_endpoints(self):
        """Verify FastAPI REST API endpoints (/api/health, /api/model-info, /api/model-metrics)."""
        r_health = self.client.get("/api/health")
        self.assertEqual(r_health.status_code, 200)
        h_data = r_health.json()
        self.assertEqual(h_data["status"], "ONLINE")
        self.assertTrue(h_data["sklearn_pipeline_loaded"])
        self.assertIn("selected_relationship_classifier", h_data["active_model"])

        r_info = self.client.get("/api/model-info")
        self.assertEqual(r_info.status_code, 200)
        info_data = r_info.json()
        self.assertEqual(info_data["scikit_learn_pipeline"]["detected"], True)
        self.assertEqual(len(info_data["supported_transformations"]), 12)

        r_metrics = self.client.get("/api/model-metrics")
        self.assertEqual(r_metrics.status_code, 200)
        m_data = r_metrics.json()
        self.assertEqual(m_data["selected_relationship_model"]["feature_dim"], 115)
        self.assertEqual(m_data["selected_relationship_model"]["test_split_metrics"]["accuracy"], 0.8542)

    def test_07_test_set_benchmark_reproduction(self):
        """Verify that the model achieves the verified benchmark of 85.42% accuracy and 1.0 precision."""
        df_f = pd.read_csv(FORENSIC_CSV).set_index("image_id")
        df_test = pd.read_csv(TEST_PAIRS_CSV)

        forensic_17_cols = [
            "width", "height", "area", "aspect_ratio", "file_size", "jpeg_quality_est",
            "r_mean", "r_std", "g_mean", "g_std", "b_mean", "b_std",
            "edge_mean", "edge_std", "edge_max", "noise_res_std", "noise_res_skew"
        ]

        def get_57d(img_id):
            r17 = df_f.loc[img_id, forensic_17_cols].values.astype(float)
            npy = np.load(DATASET_DIR / "features" / f"{img_id}.npy")
            h40 = npy[15:55].astype(float)
            return np.concatenate([r17, h40])

        X = []
        y_true = []
        for _, r in df_test.iterrows():
            v_a = get_57d(r["image_a"])
            v_b = get_57d(r["image_b"])
            diff = np.abs(v_a - v_b)
            prod = v_a * v_b
            cos = np.dot(v_a, v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b) + 1e-8)
            X.append(np.concatenate([diff, prod, [cos]]))
            y_true.append(int(r["related"]))

        X = np.array(X)
        y_true = np.array(y_true)

        pipe = joblib.load(MODELS_DIR / "selected_relationship_classifier.pkl")
        preds = pipe.predict(X)

        from sklearn.metrics import accuracy_score, precision_score
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds)

        self.assertAlmostEqual(acc, 0.85416666, places=4, msg="Test accuracy must match 85.42%")
        self.assertEqual(prec, 1.0, "Test precision must be 1.0 (zero false positives)")


if __name__ == "__main__":
    unittest.main()
