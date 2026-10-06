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
DB_PATH = BASE_DIR / "provenance_app.db"

MODEL_FILE = MODELS_DIR / "provenance_exp3_model.pt"

# Ensure runtime directories exist
for path in [DATA_DIR, UPLOAD_DIR, REPORT_DIR, VISUALIZATION_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Settings
SECRET_KEY = os.getenv("SECRET_KEY", "provex-ai-cyber-forensics-secret-key-2026")
DEFAULT_SIMILARITY_THRESHOLD = 0.65
DEFAULT_CONFIDENCE_FILTER = 0.50
MAX_IMAGE_UPLOAD_COUNT = 30
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
