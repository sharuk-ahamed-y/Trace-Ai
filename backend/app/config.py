"""
backend/app/config.py — Global Configuration & Environment Settings
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
REPORT_DIR = DATA_DIR / "reports"
VISUALIZATION_DIR = DATA_DIR / "visualizations"
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "image_provenance_dataset"
if not (DATASET_DIR / "graphs").exists() and (DATA_DIR / "provenance_dataset_v2").exists():
    DATASET_DIR = DATA_DIR / "provenance_dataset_v2"
DB_PATH = BASE_DIR / "provenance_app.db"

# Models Paths
TRAINED_MODELS_PATH = os.getenv("TRAINED_MODELS_PATH", str(MODELS_DIR))
TRAINED_MODELS_DIR = Path(TRAINED_MODELS_PATH)
COLAB_MODELS_PATH = os.getenv("COLAB_MODELS_PATH", "/content/drive/MyDrive/dataset_RE/dataset/trained_models")

MODEL_FILE = MODELS_DIR / "provenance_exp3_model.pt"
RESEARCH_MODEL_FILE = MODELS_DIR / "research_provenance_model.pt"
DIRECTION_MLP_FILE = MODELS_DIR / "direction_mlp.pt"
TRANSFORM_MLP_FILE = MODELS_DIR / "transform_mlp.pt"

# Known Colab / Scikit-learn Classifier filenames
PKL_RELATIONSHIP_NAMES = [
    "selected_relationship_classifier.pkl",
    "final_relationship_classifier.pkl",
    "relationship_classifier.pkl"
]
PKL_DIRECTION_NAMES = [
    "direction_classifier.pkl",
    "selected_direction_classifier.pkl"
]
PKL_TRANSFORM_NAMES = [
    "transformation_classifier_top20.pkl",
    "transformation_classifier.pkl",
    "selected_transformation_classifier.pkl"
]

# Ensure runtime directories exist
for path in [DATA_DIR, UPLOAD_DIR, REPORT_DIR, VISUALIZATION_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Settings
SECRET_KEY = os.getenv("SECRET_KEY", "provex-ai-cyber-forensics-secret-key-2026")
DEFAULT_SIMILARITY_THRESHOLD = 0.55
DEFAULT_CONFIDENCE_FILTER = 0.40
MAX_IMAGE_UPLOAD_COUNT = 30
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}

# Phase 4 Isolated Models & Reversible Feature Switch (Default: Disabled for safety)
ENABLE_PHASE4_MODELS = os.getenv("ENABLE_PHASE4_MODELS", "false").lower() in ("true", "1", "yes")
PHASE4_MODELS_DIR = BASE_DIR / "models" / "phase4_v1"

