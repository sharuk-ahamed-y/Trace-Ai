"""
backend/app/services/model_manager.py — Pre-trained Model Preloader & Inference Service
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

from backend.app.config import MODEL_FILE, BASE_DIR
from src.feature_extractor import ViTEmbedder, extract_embedding
from src.pair_features import build_pair_feature
from src.transforms import TRANSFORMATION_NAMES

class ProvenanceClassifier(nn.Module):
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
        self.rel_head = nn.Linear(128, 2)            # Related (0/1)
        self.dir_head = nn.Linear(128, 3)            # A_TO_B, B_TO_A, NONE
        self.tfm_head = nn.Linear(128, n_transforms + 1)  # 12 transforms + NONE

    def forward(self, x):
        h = self.shared(x)
        return self.rel_head(h), self.dir_head(h), self.tfm_head(h)


class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        if self.initialized:
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ModelManager] Initializing on device: {self.device}")

        # Load ViT Embedder
        self.embedder = ViTEmbedder().to(self.device)
        self.embedder.eval()

        # Load Provenance Classifier
        self.classifier = None
        self.use_forensics = False
        
        if MODEL_FILE.exists():
            try:
                ckpt = torch.load(MODEL_FILE, map_location=self.device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    state_dict = ckpt["model_state_dict"]
                else:
                    state_dict = ckpt
                
                input_dim = state_dict["shared.0.weight"].shape[1] if "shared.0.weight" in state_dict else 2350
                n_transforms = len(TRANSFORMATION_NAMES)
                
                self.classifier = ProvenanceClassifier(input_dim, n_transforms).to(self.device)
                self.classifier.load_state_dict(state_dict)
                self.classifier.eval()
                self.use_forensics = (input_dim > 2305)
                print(f"[ModelManager] Loaded trained model from {MODEL_FILE} (input_dim={input_dim}, forensics={self.use_forensics})")
            except Exception as e:
                print(f"[ModelManager] Error loading model checkpoint: {e}. Will fall back to similarity heuristics.")
        else:
            print(f"[ModelManager] Warning: Checkpoint {MODEL_FILE} not found. Will use heuristic inference.")

        self.initialized = True

    def get_embedding(self, image_path: Path) -> torch.Tensor:
        self.initialize()
        return extract_embedding(image_path, device=self.device)

    def predict_pair(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        f_combined: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        self.initialize()
        
        # Base pairwise feature (2305 dims)
        base_feat = build_pair_feature(emb_a, emb_b)
        
        if self.use_forensics and f_combined is not None:
            feat = torch.cat([base_feat, f_combined.float()])
        else:
            feat = base_feat

        if self.classifier is not None:
            with torch.no_grad():
                x = feat.unsqueeze(0).to(self.device)
                rel_logits, dir_logits, tfm_logits = self.classifier(x)

                rel_prob = F.softmax(rel_logits, dim=-1)[0, 1].item()
                dir_probs = F.softmax(dir_logits, dim=-1)[0]
                dir_idx = torch.argmax(dir_probs).item()

                tfm_probs = F.softmax(tfm_logits, dim=-1)[0]
                tfm_idx = torch.argmax(tfm_probs).item()

                dir_map = {0: "A_TO_B", 1: "B_TO_A", 2: "NONE"}
                dir_str = dir_map.get(dir_idx, "NONE")

                if tfm_idx < len(TRANSFORMATION_NAMES):
                    tfm_str = TRANSFORMATION_NAMES[tfm_idx]
                else:
                    tfm_str = "identity"

                tfm_conf = tfm_probs[tfm_idx].item()

                return {
                    "related": rel_prob >= 0.5,
                    "related_confidence": float(rel_prob),
                    "direction": dir_str,
                    "direction_confidence": float(dir_probs[dir_idx].item()),
                    "transformation": tfm_str,
                    "transformation_confidence": float(tfm_conf),
                }

        # Fallback Cosine Similarity Heuristic
        cos_sim = F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
        related = cos_sim >= 0.70
        return {
            "related": related,
            "related_confidence": float(cos_sim),
            "direction": "A_TO_B" if related else "NONE",
            "direction_confidence": float(cos_sim),
            "transformation": "crop" if related else "identity",
            "transformation_confidence": float(cos_sim),
        }

def get_model_manager() -> ModelManager:
    manager = ModelManager()
    manager.initialize()
    return manager
