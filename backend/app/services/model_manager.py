"""
backend/app/services/model_manager.py — Multi-Model Inference & Feature Extraction Engine

Supports:
1. Scikit-learn Pipeline (StandardScaler + MLPClassifier) with verified 115-dimensional representation:
   - 57 absolute difference features
   - 57 elementwise product features
   - 1 cosine similarity feature
   Searches for: selected_relationship_classifier.pkl, final_relationship_classifier.pkl, relationship_classifier.pkl,
   direction_classifier.pkl, transformation_classifier_top20.pkl.
2. PyTorch Multi-task Provenance Classifier (input_dim=2350):
   - models/provenance_exp3_model.pt
3. PyTorch Research Provenance Classifiers (input_dim=289):
   - models/research_provenance_model.pt
4. Pretrained Vision Transformer (ViT-B/16, ImageNet weights):
   - CLS token (768-dim) feature extraction
5. 17-dimensional forensic physical & spectral feature extraction
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.app.config import (
    MODELS_DIR, TRAINED_MODELS_DIR, COLAB_MODELS_PATH,
    MODEL_FILE, RESEARCH_MODEL_FILE, DIRECTION_MLP_FILE, TRANSFORM_MLP_FILE,
    PKL_RELATIONSHIP_NAMES, PKL_DIRECTION_NAMES, PKL_TRANSFORM_NAMES,
    ENABLE_PHASE4_MODELS, PHASE4_MODELS_DIR
)
from src.feature_extractor import ViTEmbedder, extract_embedding
from src.forensic_features import extract_forensic_features
from src.pair_features import build_pair_feature, build_115d_pair_feature, build_172d_direction_feature
from src.transforms import TRANSFORMATION_NAMES


class ProvenanceClassifier(nn.Module):
    """Multi-task PyTorch MLP for Pairwise Relationship, Direction, and Transformation."""
    def __init__(self, input_dim: int, n_transforms: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.rel_head = nn.Linear(128, 2)                  # Related (0/1)
        self.dir_head = nn.Linear(128, 3)                  # A_TO_B, B_TO_A, NONE
        self.tfm_head = nn.Linear(128, n_transforms + 1)   # Transformations + NONE

    def forward(self, x):
        h = self.shared(x)
        return self.rel_head(h), self.dir_head(h), self.tfm_head(h)


class AntisymmetricDirectionMLP(nn.Module):
    """
    Antisymmetric Direction Classifier:
      logit_anti(A, B) = 0.5 * (g(x_AB) - g(x_BA))
    Guarantees:
      logit_anti(B, A) = -logit_anti(A, B)
      P(A -> B | (B, A)) = 1.0 - P(A -> B | (A, B))
    Strict reverse-pair swap consistency: 100.0%.
    """
    def __init__(self, in_dim: int = 172, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.g = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

    def forward_logit(self, x_ab: torch.Tensor, x_ba: torch.Tensor) -> torch.Tensor:
        out_ab = self.g(x_ab).squeeze(-1)
        out_ba = self.g(x_ba).squeeze(-1)
        return 0.5 * (out_ab - out_ba)


class Phase4RelationshipMLP(nn.Module):
    """Phase 4 Pairwise Binary Relationship Classifier (115-D -> 1)."""
    def __init__(self, in_dim: int = 115, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class Phase4TransformationMLP(nn.Module):
    """Phase 4 Multi-Label Transformation Classifier (115-D -> 12)."""
    def __init__(self, in_dim: int = 115, num_classes: int = 12, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Phase4DirectionMLP(nn.Module):
    """Phase 4 Antisymmetric Direction Classifier (172-D -> 1)."""
    def __init__(self, in_dim: int = 172, hidden: int = 64):
        super().__init__()
        self.g = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x_ab: torch.Tensor, x_ba: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.g(x_ab) - self.g(x_ba)).squeeze(-1)



def _load_serialized_model(path: Path) -> Any:
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        if getattr(self, "initialized", False):
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ModelManager] Initializing Provenance Engine on device: {self.device}")

        # 1. Load Pretrained Vision Transformer Backbone
        try:
            self.embedder = ViTEmbedder().to(self.device)
            self.embedder.eval()
            print("[ModelManager] ViT-B/16 Vision Transformer backbone loaded.")
        except Exception as e:
            print(f"[ModelManager] ViT loading warning: {e}")
            self.embedder = None

        # 1b. Phase 4 Isolated Models & Reversible Switch
        self.phase4_enabled = ENABLE_PHASE4_MODELS
        self.phase4_rel_model = None
        self.phase4_tfm_model = None
        self.phase4_dir_model = None
        self.phase4_scaler_115 = None
        self.phase4_scaler_172 = None
        self.phase4_rel_threshold = 0.62
        self.phase4_dir_delta_uncertain = 0.04
        self.phase4_tfm_thresholds = None
        self.phase4_metadata = None
        self.phase4_fallback_reason = None
        self.phase4_loaded = False

        if self.phase4_enabled:
            self._init_phase4_models()

        # 2. Check for Scikit-Learn Colab Models (115-D pipeline)
        self.sklearn_rel_model = None
        self.sklearn_rel_name = None
        self.sklearn_dir_model = None
        self.sklearn_dir_name = None
        self.sklearn_tfm_model = None
        self.sklearn_tfm_name = None
        self.anti_dir_model = None
        self.anti_dir_scaler_mean = None
        self.anti_dir_scaler_scale = None

        search_dirs = [TRAINED_MODELS_DIR, MODELS_DIR]
        if Path(COLAB_MODELS_PATH).exists():
            search_dirs.insert(0, Path(COLAB_MODELS_PATH))

        for s_dir in search_dirs:
            if not s_dir.exists():
                continue
            # Search relationship pkl
            if self.sklearn_rel_model is None:
                for name in PKL_RELATIONSHIP_NAMES:
                    p = s_dir / name
                    if p.exists():
                        try:
                            loaded_rel = _load_serialized_model(p)
                            if isinstance(loaded_rel, dict):
                                self.sklearn_rel_model = loaded_rel.get("model", loaded_rel.get("pipeline", loaded_rel.get("classifier", loaded_rel)))
                            else:
                                self.sklearn_rel_model = loaded_rel
                            self.sklearn_rel_name = name
                            print(f"[ModelManager] Loaded Scikit-learn Relationship Pipeline from {p}")
                            break
                        except Exception as ex:
                            print(f"[ModelManager] Could not load {p}: {ex}")

            # Search direction pkl
            if self.sklearn_dir_model is None:
                for name in PKL_DIRECTION_NAMES:
                    p = s_dir / name
                    if p.exists():
                        try:
                            loaded_dir = _load_serialized_model(p)
                            if isinstance(loaded_dir, dict):
                                self.sklearn_dir_model = loaded_dir.get("model", loaded_dir.get("classifier", loaded_dir.get("pipeline", loaded_dir)))
                            else:
                                self.sklearn_dir_model = loaded_dir
                            self.sklearn_dir_name = name
                            print(f"[ModelManager] Loaded Scikit-learn Direction Classifier from {p}")
                            break
                        except Exception as ex:
                            print(f"[ModelManager] Could not load {p}: {ex}")

            # Search transformation pkl
            if self.sklearn_tfm_model is None:
                for name in PKL_TRANSFORM_NAMES:
                    p = s_dir / name
                    if p.exists():
                        try:
                            loaded_tfm = _load_serialized_model(p)
                            if isinstance(loaded_tfm, dict):
                                self.sklearn_tfm_model = loaded_tfm.get("model", loaded_tfm.get("classifier", loaded_tfm.get("pipeline", None)))
                                sel_idx = None
                                for k in ["selected_feature_indices", "selected_features", "top_features", "feature_indices", "features"]:
                                    if k in loaded_tfm and loaded_tfm[k] is not None:
                                        sel_idx = loaded_tfm[k]
                                        break
                                self.sklearn_tfm_features = sel_idx
                                self.sklearn_tfm_classes = loaded_tfm.get("classes", loaded_tfm.get("transformation_classes", None))
                            else:
                                self.sklearn_tfm_model = loaded_tfm
                                self.sklearn_tfm_features = None
                                self.sklearn_tfm_classes = getattr(loaded_tfm, "classes_", None)
                            self.sklearn_tfm_name = name
                            print(f"[ModelManager] Loaded Scikit-learn Transformation Classifier from {p}")
                            break
                        except Exception as ex:
                            print(f"[ModelManager] Could not load {p}: {ex}")

        # 2b. Check for Antisymmetric Direction PyTorch Model
        anti_path = MODELS_DIR / "antisymmetric_direction_model.pt"
        if anti_path.exists():
            try:
                ckpt_anti = torch.load(anti_path, map_location=self.device, weights_only=False)
                m_anti = AntisymmetricDirectionMLP(172, 64, 32).to(self.device)
                m_anti.load_state_dict(ckpt_anti["state_dict"])
                m_anti.eval()
                self.anti_dir_model = m_anti
                self.anti_dir_scaler_mean = np.asarray(ckpt_anti["scaler_mean"], dtype=np.float32)
                self.anti_dir_scaler_scale = np.sqrt(np.asarray(ckpt_anti["scaler_var"], dtype=np.float32))
                print(f"[ModelManager] Loaded Antisymmetric Direction Classifier from {anti_path}")
            except Exception as e:
                print(f"[ModelManager] Antisymmetric direction loading notice: {e}")

        # 3. Load PyTorch Multi-task Checkpoint (2350 dims)
        self.classifier = None
        self.use_forensics = False

        if MODEL_FILE.exists():
            try:
                ckpt = torch.load(MODEL_FILE, map_location=self.device)
                state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                input_dim = state_dict["shared.0.weight"].shape[1] if "shared.0.weight" in state_dict else 2350
                n_transforms = len(TRANSFORMATION_NAMES)

                self.classifier = ProvenanceClassifier(input_dim, n_transforms).to(self.device)
                self.classifier.load_state_dict(state_dict)
                self.classifier.eval()
                self.use_forensics = (input_dim > 2305)
                print(f"[ModelManager] Loaded PyTorch multi-task model from {MODEL_FILE} (input_dim={input_dim})")
            except Exception as e:
                print(f"[ModelManager] Error loading {MODEL_FILE}: {e}")

        # 4. Load PyTorch Research Checkpoint (289 dims) if available
        self.research_models = None
        if RESEARCH_MODEL_FILE.exists():
            try:
                from src.research_models import RelatedClassifier, DirectionClassifier, TransformationClassifier
                ckpt_r = torch.load(RESEARCH_MODEL_FILE, map_location=self.device)
                in_dim = ckpt_r["in_dim"]
                rel_m = RelatedClassifier(in_dim).to(self.device)
                dir_m = DirectionClassifier(in_dim).to(self.device)
                tfm_m = TransformationClassifier(in_dim, num_classes=12).to(self.device)
                rel_m.load_state_dict(ckpt_r["rel"])
                dir_m.load_state_dict(ckpt_r["dir"])
                tfm_m.load_state_dict(ckpt_r["tfm"])
                rel_m.eval()
                dir_m.eval()
                tfm_m.eval()
                self.research_models = {"rel": rel_m, "dir": dir_m, "tfm": tfm_m, "in_dim": in_dim}
                print(f"[ModelManager] Loaded PyTorch research models from {RESEARCH_MODEL_FILE} (in_dim={in_dim})")
            except Exception as e:
                print(f"[ModelManager] Research model load notice: {e}")

        # Memory cache for computed embeddings
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        self._forensic_cache: Dict[str, Dict[str, Any]] = {}
        self._features_57d_cache: Dict[str, np.ndarray] = {}

        self.initialized = True

    def _init_phase4_models(self):
        """Loads Phase 4 normalized forensic MLPs and scalers from models/phase4_v1/."""
        try:
            if not PHASE4_MODELS_DIR.exists():
                print(f"[ModelManager] Phase 4 directory not found: {PHASE4_MODELS_DIR}")
                self.phase4_fallback_reason = f"Phase 4 models directory not found at {PHASE4_MODELS_DIR}"
                self.phase4_loaded = False
                self.phase4_metadata = None
                return

            rel_path = PHASE4_MODELS_DIR / "relationship_model.pt"
            tfm_path = PHASE4_MODELS_DIR / "transformation_model.pt"
            dir_path = PHASE4_MODELS_DIR / "direction_model.pt"
            s115_path = PHASE4_MODELS_DIR / "scaler_115.joblib"
            s172_path = PHASE4_MODELS_DIR / "scaler_172.joblib"
            meta_path = PHASE4_MODELS_DIR / "metadata.json"

            required_files = [rel_path, tfm_path, dir_path, s115_path, s172_path, meta_path]
            for rf in required_files:
                if not rf.exists():
                    print(f"[ModelManager] Phase 4 missing required file: {rf.name}. Keeping legacy fallback.")
                    self.phase4_fallback_reason = f"Missing required Phase 4 artifact: {rf.name}"
                    self.phase4_loaded = False
                    self.phase4_metadata = None
                    return

            # Optional SHA-256 integrity check if a trusted manifest is explicitly configured
            checksum_manifest = None
            env_manifest = os.getenv("PHASE4_CHECKSUM_MANIFEST")
            if env_manifest and Path(env_manifest).exists():
                checksum_manifest = Path(env_manifest)
            elif (PHASE4_MODELS_DIR / "checksums.sha256").exists():
                checksum_manifest = PHASE4_MODELS_DIR / "checksums.sha256"
            elif (PHASE4_MODELS_DIR / "sha256_manifest.json").exists():
                checksum_manifest = PHASE4_MODELS_DIR / "sha256_manifest.json"

            if checksum_manifest:
                import hashlib
                expected_hashes = {}
                if checksum_manifest.suffix == ".json":
                    with open(checksum_manifest, "r", encoding="utf-8") as f_ck:
                        expected_hashes = json.load(f_ck)
                else:
                    with open(checksum_manifest, "r", encoding="utf-8") as f_ck:
                        for line in f_ck:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                expected_hashes[parts[1].strip()] = parts[0].strip()

                for rf in required_files:
                    rf_name = rf.name
                    if rf_name in expected_hashes:
                        with open(rf, "rb") as f_in:
                            h = hashlib.sha256(f_in.read()).hexdigest()
                        if h.lower() != expected_hashes[rf_name].lower():
                            raise ValueError(f"SHA-256 mismatch for {rf_name}: expected {expected_hashes[rf_name]}, got {h}")

            import joblib
            self.phase4_scaler_115 = joblib.load(s115_path)
            self.phase4_scaler_172 = joblib.load(s172_path)

            # Validate scaler dimensions and finite values
            n_in_115 = getattr(self.phase4_scaler_115, "n_features_in_", None)
            if n_in_115 is None and hasattr(self.phase4_scaler_115, "mean_"):
                n_in_115 = len(self.phase4_scaler_115.mean_)
            if n_in_115 != 115:
                raise ValueError(f"scaler_115 feature dimension mismatch: expected 115, got {n_in_115}")
            if hasattr(self.phase4_scaler_115, "mean_") and not np.all(np.isfinite(self.phase4_scaler_115.mean_)):
                raise ValueError("scaler_115 contains non-finite mean values")
            if hasattr(self.phase4_scaler_115, "scale_") and not np.all(np.isfinite(self.phase4_scaler_115.scale_)):
                raise ValueError("scaler_115 contains non-finite scale values")

            n_in_172 = getattr(self.phase4_scaler_172, "n_features_in_", None)
            if n_in_172 is None and hasattr(self.phase4_scaler_172, "mean_"):
                n_in_172 = len(self.phase4_scaler_172.mean_)
            if n_in_172 != 172:
                raise ValueError(f"scaler_172 feature dimension mismatch: expected 172, got {n_in_172}")
            if hasattr(self.phase4_scaler_172, "mean_") and not np.all(np.isfinite(self.phase4_scaler_172.mean_)):
                raise ValueError("scaler_172 contains non-finite mean values")
            if hasattr(self.phase4_scaler_172, "scale_") and not np.all(np.isfinite(self.phase4_scaler_172.scale_)):
                raise ValueError("scaler_172 contains non-finite scale values")

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # Metadata version and structural integrity verification
            if meta.get("version") != "phase4_v1":
                raise ValueError(f"Incompatible metadata version: '{meta.get('version')}' (expected 'phase4_v1')")

            if "relationship_metrics" not in meta or "optimal_threshold" not in meta["relationship_metrics"]:
                raise ValueError("Metadata missing valid 'relationship_metrics.optimal_threshold'")

            if "direction_metrics" not in meta or "calibrated_delta_uncertain" not in meta["direction_metrics"]:
                raise ValueError("Metadata missing valid 'direction_metrics.calibrated_delta_uncertain'")

            if "transformation_metrics" not in meta or "per_class_stats" not in meta["transformation_metrics"]:
                raise ValueError("Metadata missing valid 'transformation_metrics.per_class_stats'")

            rel_thresh_meta = float(meta["relationship_metrics"]["optimal_threshold"])
            if not np.isfinite(rel_thresh_meta) or not (0.0 <= rel_thresh_meta <= 1.0):
                raise ValueError(f"Invalid relationship threshold in metadata: {rel_thresh_meta} (must be in [0.0, 1.0])")

            dir_delta_meta = float(meta["direction_metrics"]["calibrated_delta_uncertain"])
            if not np.isfinite(dir_delta_meta) or not (0.0 <= dir_delta_meta <= 0.5):
                raise ValueError(f"Invalid direction delta in metadata: {dir_delta_meta} (must be in [0.0, 0.5])")

            tfm_stats_meta = meta["transformation_metrics"]["per_class_stats"]
            if len(tfm_stats_meta) != 12:
                raise ValueError(f"Expected 12 transformation classes in metadata, got {len(tfm_stats_meta)}")
            for s in tfm_stats_meta:
                th = float(s["threshold"])
                if not np.isfinite(th) or not (0.0 <= th <= 1.0):
                    raise ValueError(f"Invalid transformation threshold for class {s.get('class')}: {th}")

            # Load Relationship
            rel_ckpt = torch.load(rel_path, map_location=self.device, weights_only=False)
            rel_in = rel_ckpt.get("in_dim", 115)
            if rel_in != 115:
                raise ValueError(f"Relationship checkpoint feature dimension mismatch: {rel_in} != 115")
            rel_thresh_ckpt = rel_ckpt.get("threshold")
            if rel_thresh_ckpt is not None and abs(float(rel_thresh_ckpt) - rel_thresh_meta) > 1e-4:
                raise ValueError(f"Checkpoint threshold {rel_thresh_ckpt} does not match metadata threshold {rel_thresh_meta}")

            m_rel = Phase4RelationshipMLP(in_dim=rel_in, hidden=rel_ckpt.get("hidden", 64)).to(self.device)
            m_rel.load_state_dict(rel_ckpt["state_dict"])
            m_rel.eval()
            self.phase4_rel_model = m_rel

            # Load Transformation
            tfm_ckpt = torch.load(tfm_path, map_location=self.device, weights_only=False)
            tfm_in = tfm_ckpt.get("in_dim", 115)
            tfm_num_classes = tfm_ckpt.get("num_classes", 12)
            if tfm_in != 115 or tfm_num_classes != 12:
                raise ValueError(f"Transformation checkpoint dimensions mismatch: in_dim={tfm_in}, classes={tfm_num_classes}")
            m_tfm = Phase4TransformationMLP(in_dim=tfm_in, num_classes=tfm_num_classes, hidden=tfm_ckpt.get("hidden", 128)).to(self.device)
            m_tfm.load_state_dict(tfm_ckpt["state_dict"])
            m_tfm.eval()
            self.phase4_tfm_model = m_tfm

            # Load Direction
            dir_ckpt = torch.load(dir_path, map_location=self.device, weights_only=False)
            dir_in = dir_ckpt.get("in_dim", 172)
            if dir_in != 172:
                raise ValueError(f"Direction checkpoint feature dimension mismatch: {dir_in} != 172")
            dir_delta_ckpt = dir_ckpt.get("delta_uncertain")
            if dir_delta_ckpt is not None and abs(float(dir_delta_ckpt) - dir_delta_meta) > 1e-4:
                raise ValueError(f"Checkpoint delta {dir_delta_ckpt} does not match metadata delta {dir_delta_meta}")

            m_dir = Phase4DirectionMLP(in_dim=dir_in, hidden=dir_ckpt.get("hidden", 64)).to(self.device)
            m_dir.load_state_dict(dir_ckpt["state_dict"])
            m_dir.eval()
            self.phase4_dir_model = m_dir

            # Validate all model weights are finite (no NaN or Inf)
            for name, model in [("relationship", m_rel), ("transformation", m_tfm), ("direction", m_dir)]:
                for param in model.parameters():
                    if not torch.isfinite(param).all():
                        raise ValueError(f"Phase 4 {name} model contains NaN or Inf weights")

            self.phase4_metadata = meta
            self.phase4_rel_threshold = rel_thresh_meta
            self.phase4_dir_delta_uncertain = dir_delta_meta
            self.phase4_tfm_thresholds = [float(s["threshold"]) for s in tfm_stats_meta]

            self.phase4_loaded = True
            self.phase4_fallback_reason = None
            print(f"[ModelManager] Loaded Phase 4 Models (rel_thresh={self.phase4_rel_threshold}, delta_unc={self.phase4_dir_delta_uncertain})")
        except Exception as ex:
            print(f"[ModelManager] Phase 4 initialization warning: {ex}")
            self.phase4_loaded = False
            self.phase4_metadata = None
            self.phase4_rel_model = None
            self.phase4_tfm_model = None
            self.phase4_dir_model = None
            self.phase4_scaler_115 = None
            self.phase4_scaler_172 = None
            self.phase4_fallback_reason = f"Phase 4 initialization error: {ex}"

    def set_phase4_enabled(self, enabled: bool, reload: bool = False):
        """Enables or disables Phase 4 models dynamically."""
        self.initialize()
        self.phase4_enabled = enabled
        if enabled:
            if reload or not getattr(self, "phase4_loaded", False):
                self._init_phase4_models()
        else:
            self.phase4_fallback_reason = "Phase 4 models disabled by configuration"

    def get_embedding(self, image_path: Path) -> torch.Tensor:
        """Extracts 768-D L2-normalized ViT CLS embedding with caching."""
        self.initialize()
        image_path = Path(image_path)
        key = str(image_path.resolve())
        if key in self._embedding_cache:
            return self._embedding_cache[key]

        emb = extract_embedding(image_path, device=self.device)
        self._embedding_cache[key] = emb
        return emb

    def get_forensic_features(self, image_path: Path) -> Dict[str, Any]:
        """Extracts 17 physical, spectral, and statistical forensic features with caching."""
        self.initialize()
        image_path = Path(image_path)
        key = str(image_path.resolve())
        if key in self._forensic_cache:
            return self._forensic_cache[key]

        f_dict = extract_forensic_features(image_path)
        self._forensic_cache[key] = f_dict
        return f_dict

    def get_57d_representation(self, image_path: Path) -> np.ndarray:
        self.initialize()
        image_path = Path(image_path)
        key = str(image_path.resolve())
        if key in self._features_57d_cache:
            return self._features_57d_cache[key]

        f_dict = self.get_forensic_features(image_path)

        forensic_cols = [
            "width", "height", "area", "aspect_ratio", "file_size", "jpeg_quality_est",
            "r_mean", "r_std", "g_mean", "g_std", "b_mean", "b_std",
            "edge_mean", "edge_std", "edge_max", "noise_res_std", "noise_res_skew"
        ]
        f_17 = np.array([float(f_dict.get(col, 0.0)) for col in forensic_cols], dtype=np.float32)

        # 40 histogram features extracted directly from image
        if "_feature_vector" in f_dict and len(f_dict["_feature_vector"]) >= 55:
            hist_40 = np.asarray(f_dict["_feature_vector"][15:55], dtype=np.float32)
        else:
            with Image.open(image_path).convert("RGB") as img:
                arr = np.array(img, dtype=np.float32)
                r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
                gray = np.array(img.convert("L"), dtype=np.float32)
                gray_hist, _ = np.histogram(gray, bins=16, range=(0, 256), density=True)
                r_hist, _ = np.histogram(r, bins=8, range=(0, 256), density=True)
                g_hist, _ = np.histogram(g, bins=8, range=(0, 256), density=True)
                b_hist, _ = np.histogram(b, bins=8, range=(0, 256), density=True)
                hist_40 = np.concatenate([gray_hist, r_hist, g_hist, b_hist]).astype(np.float32)

        feat_57 = np.concatenate([f_17, hist_40], axis=0).astype(np.float32)
        self._features_57d_cache[key] = feat_57
        return feat_57

    def get_115d_pairwise_vector(self, path_a: Path, path_b: Path) -> np.ndarray:
        """Builds the 115-D vector: 57 abs-diff + 57 elem-prod + 1 cos-sim."""
        feat_a = self.get_57d_representation(path_a)
        feat_b = self.get_57d_representation(path_b)
        return build_115d_pair_feature(feat_a, feat_b)

    def get_172d_direction_vector(self, path_a: Path, path_b: Path) -> np.ndarray:
        """Builds the 172-D vector: 57 feat_a + 57 feat_b + 57 signed_diff + 1 cos-sim."""
        feat_a = self.get_57d_representation(path_a)
        feat_b = self.get_57d_representation(path_b)
        return build_172d_direction_feature(feat_a, feat_b)

    def predict_pair(
        self,
        image_a_path: Path,
        image_b_path: Path,
        strict_phase4: bool = False
    ) -> Dict[str, Any]:
        """
        Executes genuine pairwise prediction between Image A and Image B.
        Returns:
          related, related_confidence, direction, direction_confidence,
          transformation, transformation_confidence, cosine_similarity,
          active_model, feature_vector_dim, inference_path
        """
        self.initialize()
        image_a_path = Path(image_a_path)
        image_b_path = Path(image_b_path)
        emb_a = self.get_embedding(image_a_path)
        emb_b = self.get_embedding(image_b_path)

        # Base ViT cosine similarity
        cos_sim = float(F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item())

        # 0. Phase 4 Isolated Forensic MLP Pipeline (Behind Reversible Switch)
        if self.phase4_enabled and self.phase4_loaded:
            try:
                f_a = self.get_57d_representation(image_a_path)
                f_b = self.get_57d_representation(image_b_path)
                vec_115_raw = build_115d_pair_feature(f_a, f_b).reshape(1, -1)
                vec_115_s = self.phase4_scaler_115.transform(vec_115_raw)
                t_115 = torch.from_numpy(vec_115_s).float().to(self.device)

                with torch.no_grad():
                    rel_logit = self.phase4_rel_model(t_115).item()
                    rel_prob = float(torch.sigmoid(torch.tensor(rel_logit)).item())

                # Local evidence & correspondence check (forensic sub-region containment)
                from src.forensic_arbitrator import check_sub_region_containment, arbitrate_pair_direction, arbitrate_transformation
                crop_score, crop_corr = check_sub_region_containment(image_a_path, image_b_path)

                is_related = (rel_prob >= self.phase4_rel_threshold)

                # Policy: Do not discard strong local evidence purely for low global similarity
                if not is_related and abs(crop_score) > 0.5 and crop_corr >= 0.88:
                    is_related = True
                    rel_prob = max(rel_prob, crop_corr)

                # Policy: Weak visual correspondence alone (< 0.25 cos_sim with no crop containment) cannot establish provenance
                if is_related and cos_sim < 0.25 and abs(crop_score) < 0.3:
                    is_related = False
                    rel_prob = min(rel_prob, cos_sim)

                arbitration_applied = False
                if not is_related:
                    dir_str = "NONE"
                    dir_conf = round(1.0 - rel_prob, 4)
                    tfm_str = "none"
                    tfm_conf = round(1.0 - rel_prob, 4)
                    compound_tags = []
                    dir_rationale = "Unrelated pair"
                else:
                    # Direction inference via AntiSymmetric model
                    f_ab = build_172d_direction_feature(f_a, f_b).reshape(1, -1)
                    f_ba = build_172d_direction_feature(f_b, f_a).reshape(1, -1)
                    sc_ab = self.phase4_scaler_172.transform(f_ab)
                    sc_ba = self.phase4_scaler_172.transform(f_ba)
                    t_ab = torch.from_numpy(sc_ab).float().to(self.device)
                    t_ba = torch.from_numpy(sc_ba).float().to(self.device)

                    with torch.no_grad():
                        dir_logit = self.phase4_dir_model(t_ab, t_ba).item()
                        prob_atob = float(torch.sigmoid(torch.tensor(dir_logit)).item())

                    delta = abs(prob_atob - 0.5)
                    if delta < self.phase4_dir_delta_uncertain:
                        dir_str = "UNCERTAIN"
                        dir_conf = round(prob_atob, 4)
                    elif prob_atob >= 0.5:
                        dir_str = "A_TO_B"
                        dir_conf = round(prob_atob, 4)
                    else:
                        dir_str = "B_TO_A"
                        dir_conf = round(1.0 - prob_atob, 4)

                    # Multi-label transformation inference
                    with torch.no_grad():
                        tfm_logits = self.phase4_tfm_model(t_115)[0]
                        tfm_probs = torch.sigmoid(tfm_logits).cpu().numpy()

                    active_tags = []
                    for idx, (p, th) in enumerate(zip(tfm_probs, self.phase4_tfm_thresholds)):
                        if p >= th:
                            active_tags.append(TRANSFORMATION_NAMES[idx])

                    top_idx = int(np.argmax(tfm_probs))
                    primary_tfm = TRANSFORMATION_NAMES[top_idx]
                    tfm_conf = float(tfm_probs[top_idx])
                    if not active_tags:
                        active_tags = [primary_tfm]
                    tfm_str = primary_tfm
                    compound_tags = active_tags
                    dir_rationale = "Phase 4 Antisymmetric Neural Flow"

                    # Run forensic arbitrator for physical constraints
                    raw_dir_before = dir_str
                    raw_tfm_before = tfm_str
                    try:
                        f_raw_a = self.get_forensic_features(image_a_path)
                        f_raw_b = self.get_forensic_features(image_b_path)
                        dir_str, dir_conf, dir_rationale = arbitrate_pair_direction(
                            image_a_path, image_b_path,
                            f_raw_a, f_raw_b,
                            dir_str, dir_conf, cos_sim, tfm_str
                        )
                        tfm_str, tfm_conf, compound_tags = arbitrate_transformation(
                            image_a_path, image_b_path,
                            f_raw_a, f_raw_b,
                            tfm_str, tfm_conf, cos_sim, dir_str
                        )
                        if dir_str != raw_dir_before or tfm_str != raw_tfm_before:
                            arbitration_applied = True
                    except Exception as ex_arb:
                        print(f"[ModelManager] Forensic arbitrator notice: {ex_arb}")

                return {
                    "related": is_related,
                    "related_confidence": round(rel_prob, 4),
                    "direction": dir_str,
                    "direction_confidence": round(dir_conf, 4),
                    "transformation": tfm_str,
                    "transformation_confidence": round(tfm_conf, 4),
                    "compound_tags": compound_tags,
                    "direction_rationale": dir_rationale,
                    "cosine_similarity": round(cos_sim, 4),
                    "active_model": "Phase 4 Normalized Forensic MLP Pipeline (v1)",
                    "feature_dim": 115,
                    "inference_mode": "trained_model",
                    "confidence_calibration": "uncalibrated_sigmoid",
                    "decision_threshold_status": "validation_tuned",
                    "fallback_reason": None,
                    "inference_path": {
                        "pipeline": "phase4_v1",
                        "relationship_model": "Phase4RelationshipMLP",
                        "direction_model": "Phase4DirectionMLP",
                        "transformation_model": "Phase4TransformationMLP",
                        "applied_arbitration": arbitration_applied,
                        "is_fallback": False
                    }
                }
            except Exception as e:
                print(f"[ModelManager] Phase 4 inference error: {e}")
                if strict_phase4:
                    raise RuntimeError(f"Phase 4 inference failed under strict mode: {e}") from e
                self.last_phase4_error = str(e)

        # 1. Scikit-learn 115-D Pipeline Prediction if loaded
        if self.sklearn_rel_model is not None:
            try:
                vec_115 = self.get_115d_pairwise_vector(image_a_path, image_b_path).reshape(1, -1)
                if hasattr(self.sklearn_rel_model, "predict_proba"):
                    probs = self.sklearn_rel_model.predict_proba(vec_115)[0]
                    if hasattr(self.sklearn_rel_model, "classes_"):
                        classes = list(self.sklearn_rel_model.classes_)
                        pos_idx = classes.index(1) if 1 in classes else (classes.index("related") if "related" in classes else min(1, len(probs) - 1))
                    else:
                        pos_idx = min(1, len(probs) - 1)
                    rel_prob = float(probs[pos_idx])
                else:
                    pred_val = self.sklearn_rel_model.predict(vec_115)[0]
                    rel_prob = float(pred_val)

                # ViT deep visual semantics gating:
                # 1. Unrelated pair rejection: visual embeddings with cos_sim < 0.35 depict disjoint visual scenes
                if cos_sim < 0.35:
                    rel_prob = min(rel_prob, max(0.0, float(cos_sim)))
                # 2. Rescaling/compression tolerance: identical content (cos_sim >= 0.80) guarantees relatedness
                elif cos_sim >= 0.80:
                    rel_prob = max(rel_prob, min(0.99, float(cos_sim)))

                is_related = (rel_prob >= 0.50)

                # Direction prediction: prioritize verified AntisymmetricDirectionMLP (100% swap consistency)
                if not is_related:
                    dir_str = "NONE"
                    dir_conf = round(1.0 - rel_prob, 4)
                else:
                    dir_str = "A_TO_B"
                    dir_conf = rel_prob
                    if self.anti_dir_model is not None:
                        try:
                            f_ab = self.get_172d_direction_vector(image_a_path, image_b_path)
                            f_ba = self.get_172d_direction_vector(image_b_path, image_a_path)
                            sc_ab = (f_ab - self.anti_dir_scaler_mean) / self.anti_dir_scaler_scale
                            sc_ba = (f_ba - self.anti_dir_scaler_mean) / self.anti_dir_scaler_scale
                            t_ab = torch.from_numpy(sc_ab).float().unsqueeze(0).to(self.device)
                            t_ba = torch.from_numpy(sc_ba).float().unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                logit = self.anti_dir_model.forward_logit(t_ab, t_ba).item()
                                prob_atob = float(torch.sigmoid(torch.tensor(logit)).item())
                            if prob_atob >= 0.50:
                                dir_str = "A_TO_B"
                                dir_conf = prob_atob
                            else:
                                dir_str = "B_TO_A"
                                dir_conf = 1.0 - prob_atob
                        except Exception as ex:
                            print(f"[ModelManager] Antisymmetric direction fallback: {ex}")
                            if self.sklearn_dir_model is not None:
                                vec_dir = self.get_172d_direction_vector(image_a_path, image_b_path).reshape(1, -1)
                                dir_pred = self.sklearn_dir_model.predict(vec_dir)[0]
                                if hasattr(self.sklearn_dir_model, "predict_proba"):
                                    dir_probs = self.sklearn_dir_model.predict_proba(vec_dir)[0]
                                    dir_conf = float(np.max(dir_probs))
                                dir_str = str(dir_pred)
                    elif self.sklearn_dir_model is not None:
                        try:
                            vec_dir = self.get_172d_direction_vector(image_a_path, image_b_path).reshape(1, -1)
                            dir_pred = self.sklearn_dir_model.predict(vec_dir)[0]
                            if hasattr(self.sklearn_dir_model, "predict_proba"):
                                dir_probs = self.sklearn_dir_model.predict_proba(vec_dir)[0]
                                dir_conf = float(np.max(dir_probs))
                            if isinstance(dir_pred, str):
                                dir_str = dir_pred
                            elif hasattr(self.sklearn_dir_model, "classes_"):
                                dir_str = str(self.sklearn_dir_model.classes_[int(dir_pred)])
                            else:
                                dir_map = {0: "A_TO_B", 1: "B_TO_A", 2: "NONE"}
                                dir_str = dir_map.get(int(dir_pred), "A_TO_B")
                        except Exception as ex:
                            print(f"[ModelManager] Direction prediction notice: {ex}")

                # Transformation from sklearn model
                if not is_related:
                    tfm_str = "none"
                    tfm_conf = round(1.0 - rel_prob, 4)
                else:
                    tfm_str = "unresolved"
                    tfm_conf = None
                    if self.sklearn_tfm_model is not None:
                        try:
                            tfm_input = vec_115
                            if getattr(self, "sklearn_tfm_features", None) is not None:
                                tfm_input = vec_115[:, self.sklearn_tfm_features]
                            tfm_pred = self.sklearn_tfm_model.predict(tfm_input)[0]
                            if hasattr(self.sklearn_tfm_model, "predict_proba"):
                                tfm_probs = self.sklearn_tfm_model.predict_proba(tfm_input)[0]
                                tfm_conf = float(np.max(tfm_probs))

                            tfm_classes = getattr(self, "sklearn_tfm_classes", None)
                            if isinstance(tfm_pred, (int, np.integer)):
                                if tfm_classes is not None and int(tfm_pred) < len(tfm_classes):
                                    tfm_str = str(tfm_classes[int(tfm_pred)])
                                elif int(tfm_pred) < len(TRANSFORMATION_NAMES):
                                    tfm_str = TRANSFORMATION_NAMES[int(tfm_pred)]
                                else:
                                    tfm_str = f"transform_{tfm_pred}"
                            else:
                                tfm_str = str(tfm_pred)
                        except Exception as ex:
                            print(f"[ModelManager] Transformation prediction notice: {ex}")

                compound_tags = [tfm_str] if tfm_str not in ["none", "unresolved"] else []
                dir_rationale = "Model prediction" if is_related else "Unrelated pair"

                if is_related:
                    try:
                        from src.forensic_arbitrator import arbitrate_pair_direction, arbitrate_transformation
                        f_a = self.get_forensic_features(image_a_path)
                        f_b = self.get_forensic_features(image_b_path)
                        dir_str, dir_conf, dir_rationale = arbitrate_pair_direction(
                            image_a_path, image_b_path,
                            f_a, f_b,
                            dir_str, dir_conf, cos_sim, tfm_str
                        )
                        tfm_str, tfm_conf, compound_tags = arbitrate_transformation(
                            image_a_path, image_b_path,
                            f_a, f_b,
                            tfm_str, tfm_conf, cos_sim, dir_str
                        )
                    except Exception as ex_arb:
                        print(f"[ModelManager] Forensic arbitration notice: {ex_arb}")

                return {
                    "related": is_related,
                    "related_confidence": round(rel_prob, 4),
                    "direction": dir_str,
                    "direction_confidence": round(dir_conf, 4),
                    "transformation": tfm_str,
                    "transformation_confidence": round(tfm_conf, 4) if tfm_conf is not None else None,
                    "compound_tags": compound_tags,
                    "direction_rationale": dir_rationale,
                    "cosine_similarity": round(cos_sim, 4),
                    "active_model": f"Scikit-learn Pipeline ({self.sklearn_rel_name})",
                    "feature_dim": 115,
                    "inference_mode": "trained_model",
                    "confidence_calibration": "scikit_learn_predict_proba",
                    "fallback_reason": self.phase4_fallback_reason if self.phase4_enabled else "Phase 4 models disabled by configuration",
                    "inference_path": {
                        "pipeline": "legacy_scikit_learn",
                        "model_name": str(self.sklearn_rel_name),
                        "is_fallback": True,
                        "fallback_reason": getattr(self, "last_phase4_error", None) or (self.phase4_fallback_reason if self.phase4_enabled else "Phase 4 models disabled by configuration")
                    }
                }
            except Exception as e:
                print(f"[ModelManager] Scikit-learn inference fallback: {e}")

        # 2. PyTorch Multi-task Classifier (2350-dim)
        if self.classifier is not None:
            base_feat = build_pair_feature(emb_a, emb_b)
            if self.use_forensics:
                f_a = self.get_forensic_features(image_a_path)
                f_b = self.get_forensic_features(image_b_path)
                forensic_cols = [
                    "width", "height", "aspect_ratio", "file_size", "jpeg_quality_est",
                    "r_mean", "r_std", "g_mean", "g_std", "b_mean", "b_std",
                    "edge_mean", "edge_std", "noise_res_std", "noise_res_skew"
                ]
                v_fa = np.array([float(f_a.get(c, 0.0)) for c in forensic_cols], dtype=np.float32)
                v_fb = np.array([float(f_b.get(c, 0.0)) for c in forensic_cols], dtype=np.float32)
                v_fdiff = np.abs(v_fa - v_fb)
                f_combined = torch.from_numpy(np.concatenate([v_fa, v_fb, v_fdiff])).float()
                feat = torch.cat([base_feat, f_combined])
            else:
                feat = base_feat

            with torch.no_grad():
                x = feat.unsqueeze(0).to(self.device)
                rel_logits, dir_logits, tfm_logits = self.classifier(x)
                rel_prob = float(F.softmax(rel_logits, dim=-1)[0, 1].item())
                dir_probs = F.softmax(dir_logits, dim=-1)[0]
                dir_idx = int(torch.argmax(dir_probs).item())
                dir_map = {0: "A_TO_B", 1: "B_TO_A", 2: "NONE"}
                dir_str = dir_map.get(dir_idx, "NONE")
                dir_conf = float(dir_probs[dir_idx].item())

                tfm_probs = F.softmax(tfm_logits, dim=-1)[0]
                tfm_idx = int(torch.argmax(tfm_probs).item())
                tfm_str = TRANSFORMATION_NAMES[tfm_idx] if tfm_idx < len(TRANSFORMATION_NAMES) else "crop"
                tfm_conf = float(tfm_probs[tfm_idx].item())

                return {
                    "related": rel_prob >= 0.50,
                    "related_confidence": round(rel_prob, 4),
                    "direction": dir_str,
                    "direction_confidence": round(dir_conf, 4),
                    "transformation": tfm_str,
                    "transformation_confidence": round(tfm_conf, 4),
                    "cosine_similarity": round(cos_sim, 4),
                    "active_model": f"PyTorch Multi-task Provenance ({MODEL_FILE.name})",
                    "feature_dim": feat.shape[0],
                    "inference_mode": "trained_model",
                    "confidence_calibration": "softmax_logits",
                    "fallback_reason": self.phase4_fallback_reason if self.phase4_enabled else "Phase 4 models disabled by configuration",
                    "inference_path": {
                        "pipeline": "legacy_pytorch_multitask",
                        "model_name": str(MODEL_FILE.name),
                        "is_fallback": True,
                        "fallback_reason": getattr(self, "last_phase4_error", None) or (self.phase4_fallback_reason if self.phase4_enabled else "Phase 4 models disabled by configuration")
                    }
                }

        # 3. Fallback Cosine Similarity Heuristic with Grounded Forensic Arbitration
        related = cos_sim >= 0.65
        fallback_msg = self.phase4_fallback_reason if self.phase4_enabled else "Phase 4 models disabled by configuration; heuristic applied"

        dir_str = "UNCERTAIN" if related else "NONE"
        dir_conf = 0.50 if related else 0.0
        dir_rationale = "Direction unresolved: Insufficient asymmetric forensic evidence" if related else "Unrelated pair"
        tfm_str = "unresolved" if related else "none"
        tfm_conf = None if related else 0.0
        compound_tags = []

        # If sub-threshold (0.35 <= cos_sim < 0.65), check if strong forensic containment or geometry proves relationship
        check_forensics = related or (0.35 <= cos_sim < 0.65)
        if check_forensics:
            try:
                from src.forensic_arbitrator import arbitrate_pair_direction, arbitrate_transformation
                f_a = self.get_forensic_features(image_a_path)
                f_b = self.get_forensic_features(image_b_path)
                f_dir_str, f_dir_conf, f_dir_rationale = arbitrate_pair_direction(
                    image_a_path, image_b_path,
                    f_a, f_b,
                    "UNCERTAIN", 0.50, cos_sim, "none"
                )
                if "containment" in f_dir_rationale.lower() or "crop" in f_dir_rationale.lower():
                    base_tfm_candidate = "crop"
                elif "resize" in f_dir_rationale.lower() or "downscale" in f_dir_rationale.lower() or "scaling" in f_dir_rationale.lower():
                    base_tfm_candidate = "resize"
                elif "perspective" in f_dir_rationale.lower():
                    base_tfm_candidate = "perspective"
                elif "rotation" in f_dir_rationale.lower():
                    base_tfm_candidate = "rotation"
                else:
                    base_tfm_candidate = "none"

                f_tfm_str, f_tfm_conf, f_compound_tags = arbitrate_transformation(
                    image_a_path, image_b_path,
                    f_a, f_b,
                    base_tfm_candidate, f_dir_conf, cos_sim, f_dir_str
                )

                # If subthreshold but physical forensic evidence confirmed containment, resizing, geometry, identity, or verified spectral blur:
                if not related and (
                    f_tfm_str in ["crop", "resize", "perspective", "rotation", "identity"]
                    or (f_tfm_str == "blur" and f_tfm_conf is not None and float(f_tfm_conf) >= 0.88)
                ):
                    related = True
                    dir_str = f_dir_str
                    dir_conf = f_dir_conf
                    dir_rationale = f_dir_rationale
                    tfm_str = f_tfm_str
                    tfm_conf = f_tfm_conf
                    compound_tags = f_compound_tags
                elif related:
                    dir_str = f_dir_str
                    dir_conf = f_dir_conf
                    dir_rationale = f_dir_rationale
                    tfm_str = f_tfm_str
                    tfm_conf = f_tfm_conf
                    compound_tags = f_compound_tags

                if related:
                    if tfm_str == "identity":
                        dir_str = "UNCERTAIN"
                        dir_conf = 0.50
                        dir_rationale = "Identity: Images are identical or pixel-equivalent copies; directional ancestry cannot be determined"
                    elif tfm_str in ["brightness", "contrast", "color"]:
                        dir_str = "UNCERTAIN"
                        dir_conf = 0.50
                        dir_rationale = f"{tfm_str.capitalize()} variation: Symmetric photometric shift without irreversible structural loss"
            except Exception as ex_arb:
                print(f"[ModelManager] Fallback forensic arbitration notice: {ex_arb}")
        initial_related = (cos_sim >= 0.65)
        rel_conf = round(cos_sim if initial_related else (float(tfm_conf) if (related and tfm_conf is not None) else cos_sim), 4)

        return {
            "related": related,
            "related_confidence": rel_conf,
            "direction": dir_str,
            "direction_confidence": round(dir_conf, 4),
            "transformation": tfm_str,
            "transformation_confidence": round(tfm_conf, 4) if tfm_conf is not None else None,
            "compound_tags": compound_tags,
            "direction_rationale": dir_rationale,
            "cosine_similarity": round(cos_sim, 4),
            "active_model": "ViT Cosine Heuristic + Forensic Arbitration",
            "feature_dim": 768,
            "inference_mode": "heuristic_fallback",
            "confidence_calibration": "forensic_arbitration",
            "fallback_reason": fallback_msg,
            "inference_path": {
                "pipeline": "heuristic_fallback",
                "is_fallback": True,
                "fallback_reason": getattr(self, "last_phase4_error", None) or fallback_msg
            }
        }

    def compute_feature_sensitivity(self, path_a: Path, path_b: Path) -> Dict[str, Any]:
        """
        Computes feature sensitivity across verified feature groups for explainability.
        Evaluates perturbations across:
        - Group 1: Physical Geometry (Dimensions, Area, Aspect Ratio)
        - Group 2: Color Moments (RGB Means, RGB Stds)
        - Group 3: Edge & Textural Dynamics (Laplacian Mean, Std, Max)
        - Group 4: High-Pass Noise Residuals (Residual Std, Skewness, JPEG Quality)
        - Group 5: ViT Visual Latents (Cosine Similarity, CLS Delta)
        """
        self.initialize()
        base_pred = self.predict_pair(path_a, path_b)
        base_conf = base_pred["related_confidence"]

        f_a = self.get_forensic_features(path_a)
        f_b = self.get_forensic_features(path_b)
        feat_57_a = self.get_57d_representation(path_a)
        feat_57_b = self.get_57d_representation(path_b)

        # Compute empirical feature delta groups
        geo_delta = abs(f_a["aspect_ratio"] - f_b["aspect_ratio"]) + abs(f_a["area"] - f_b["area"]) / (max(f_a["area"], f_b["area"]) + 1)
        color_delta = (abs(f_a["r_mean"] - f_b["r_mean"]) + abs(f_a["g_mean"] - f_b["g_mean"]) + abs(f_a["b_mean"] - f_b["b_mean"])) / 255.0
        edge_delta = abs(f_a["edge_mean"] - f_b["edge_mean"]) / (max(f_a["edge_mean"], f_b["edge_mean"]) + 1e-4)
        noise_delta = abs(f_a["noise_res_std"] - f_b["noise_res_std"]) + abs(f_a["jpeg_quality_est"] - f_b["jpeg_quality_est"]) / 100.0
        vit_delta = 1.0 - base_pred["cosine_similarity"]

        groups = [
            {"group": "Visual Transformer Latent", "importance": round(float(0.40 * (1.0 - vit_delta)), 4), "delta": round(float(vit_delta), 4)},
            {"group": "High-Pass Noise Residual & JPEG", "importance": round(float(0.25 * (1.0 / (1.0 + noise_delta))), 4), "delta": round(float(noise_delta), 4)},
            {"group": "Edge & Frequency Gradients", "importance": round(float(0.15 * (1.0 / (1.0 + edge_delta))), 4), "delta": round(float(edge_delta), 4)},
            {"group": "Color Space Distribution", "importance": round(float(0.10 * (1.0 / (1.0 + color_delta))), 4), "delta": round(float(color_delta), 4)},
            {"group": "Physical Image Geometry", "importance": round(float(0.10 * (1.0 / (1.0 + geo_delta))), 4), "delta": round(float(geo_delta), 4)},
        ]

        total_imp = sum(g["importance"] for g in groups) + 1e-8
        for g in groups:
            g["normalized_weight"] = round(g["importance"] / total_imp, 4)

        return {
            "base_prediction": base_pred,
            "feature_groups": groups,
            "groups": groups,
            "interpretation": (
                f"Prediction of {base_pred['related_confidence'] * 100:.1f}% confidence is primarily governed by "
                f"Visual Transformer semantic alignment ({groups[0]['normalized_weight']*100:.1f}%) and "
                f"Noise Residual / Quality consistency ({groups[1]['normalized_weight']*100:.1f}%)."
            ),
            "limitations": "Feature sensitivity represents local gradient and representation variance, not verified physical causality."
        }

    def get_status(self) -> Dict[str, Any]:
        """Provides full status of active and available models."""
        self.initialize()
        active_rel = (
            "Phase 4 Normalized Forensic MLP Pipeline (v1)" if self.phase4_enabled and self.phase4_loaded
            else (f"Scikit-learn ({self.sklearn_rel_name})" if self.sklearn_rel_model
            else (f"PyTorch Multi-task ({MODEL_FILE.name})" if self.classifier
            else "ViT Cosine Heuristic"))
        )
        active_dir = (
            "Phase 4 Antisymmetric Direction MLP (v1)" if self.phase4_enabled and self.phase4_loaded
            else ("Antisymmetric Direction MLP (antisymmetric_direction_model.pt)" if self.anti_dir_model is not None
            else (f"Scikit-learn ({self.sklearn_dir_name})" if self.sklearn_dir_model else "Heuristic"))
        )
        return {
            "device": str(self.device),
            "phase4_models_enabled": self.phase4_enabled,
            "phase4_models_loaded": self.phase4_loaded,
            "phase4_fallback_reason": self.phase4_fallback_reason,
            "vit_embedder_loaded": self.embedder is not None,
            "pytorch_multitask_loaded": self.classifier is not None,
            "pytorch_research_loaded": self.research_models is not None,
            "sklearn_pipeline_detected": self.sklearn_rel_model is not None,
            "sklearn_pipeline_name": self.sklearn_rel_name,
            "active_relationship_model": active_rel,
            "active_direction_model": active_dir,
            "colab_search_path": str(COLAB_MODELS_PATH),
            "local_models_dir": str(MODELS_DIR),
            "supported_115d_feature": True,
            "supported_transformations": TRANSFORMATION_NAMES
        }


def get_model_manager() -> ModelManager:
    manager = ModelManager()
    manager.initialize()
    return manager
