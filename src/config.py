"""
config.py — Central configuration for the Image Provenance System.
All paths, hyperparameters, and seeds live here.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Data directories ──────────────────────────────────────────────────────────
DATA_DIR       = ROOT / "data"
ORIGINALS_DIR  = DATA_DIR / "originals"
GENERATED_DIR  = DATA_DIR / "generated"
LABELS_DIR     = DATA_DIR / "labels"
EDGES_CSV      = LABELS_DIR / "edges.csv"
GRAPHS_JSON    = LABELS_DIR / "graphs.json"    # ground-truth tree structures
SPLIT_JSON     = LABELS_DIR / "split.json"     # train/val/test original IDs

# ── Model / output directories ────────────────────────────────────────────────
MODELS_DIR     = ROOT / "models"
OUTPUTS_DIR    = ROOT / "outputs"

# ── Feature caches ────────────────────────────────────────────────────────────
EMBEDDING_CACHE = MODELS_DIR / "embedding_cache.pkl"   # CLS embeddings (float32)
PATCH_CACHE     = MODELS_DIR / "patch_cache.npz"       # patch embeddings (float16)

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Image settings ────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIT_IMAGE_SIZE   = 224          # ViT-B/16 expects 224×224

# ── Transformation vocabulary (8 transforms, paper Sec III) ──────────────────
#   Order matters: JPEG is always applied LAST when combined with another
#   transform so the round-trip artefact lands on the leaf.
TRANSFORMATION_NAMES = [
    "crop",           # real dimension change — no resize-back
    "resize",         # real dimension change — no upscale-back
    "rotation",       # in-plane rotation, canvas keeps original size
    "brightness",     # pixel-level colour shift
    "contrast",       # pixel-level contrast shift
    "blur",           # Gaussian blur
    "text_insertion", # visible text watermark overlay
    "jpeg",           # lossy JPEG recompression
]
N_TRANSFORMS = len(TRANSFORMATION_NAMES)

# ── Transform parameter ranges ────────────────────────────────────────────────
TRANSFORM_PARAMS = {
    "crop": {
        "min_frac": 0.10, "max_frac": 0.30,   # fraction of dimension removed
    },
    "resize": {
        "min_scale": 0.40, "max_scale": 0.85,  # downscale factor; NOT upscaled back
    },
    "rotation": {
        "min_angle": -30.0, "max_angle": 30.0, # degrees
    },
    "brightness": {
        "min_factor": 0.50, "max_factor": 1.60,
    },
    "contrast": {
        "min_factor": 0.50, "max_factor": 1.60,
    },
    "blur": {
        "min_radius": 1.0, "max_radius": 5.0,
    },
    "text_insertion": {
        # Text strings to sample from; position chosen randomly
        "texts": ["COPY", "DRAFT", "SAMPLE", "WATERMARK", "CONFIDENTIAL",
                  "PREVIEW", "TEST", "DO NOT USE"],
        "min_opacity": 80, "max_opacity": 200,
    },
    "jpeg": {
        "min_quality": 15, "max_quality": 70,
    },
}

# ── Tree generation parameters (Phase 1) ─────────────────────────────────────
TREES_PER_ORIGINAL  = 3     # number of independent trees per original image
TREE_MIN_NODES      = 4     # minimum nodes per tree (including root)
TREE_MAX_NODES      = 8     # maximum nodes per tree
TREE_MAX_DEPTH      = 3     # maximum depth (root = depth 0)
TREE_MAX_BRANCHING  = 3     # maximum children per node

# Fraction of edges that get TWO transforms instead of one
DOUBLE_TRANSFORM_PROB = 0.35

# ── ViT feature extractor ─────────────────────────────────────────────────────
VIT_MODEL_NAME   = "vit_b_16"
EMBEDDING_DIM    = 768                  # ViT-B/16 CLS token dim
PATCH_DIM        = 768                  # per-patch embedding dim
N_PATCHES        = 196                  # 14×14 patches for 224×224 input

# ── Low-level feature dimensions (Phase 2, Eq. 7) ────────────────────────────
LOW_LEVEL_DIM    = 10       # aspect-ratio, size, JPEG block, hist×3, bright,
                            # contrast, Laplacian ratio, edge density
EVIDENCE_DIM     = EMBEDDING_DIM + PATCH_DIM + LOW_LEVEL_DIM  # = 1546

# ── GAT model hyperparameters (Phase 3) ──────────────────────────────────────
GAT_HEADS        = 4        # number of attention heads
GAT_HIDDEN       = 256      # hidden dimension per head
GAT_TAU          = 0.50     # cosine similarity threshold for candidate mask (Eq. 1)
GAT_KNN          = 3        # minimum guaranteed neighbours per node (Eq. 1)
THETA_PRUNE      = 0.30     # confidence threshold for edge pruning (Eq. 10)

# ── Loss lambdas (Eq. 11) ────────────────────────────────────────────────────
LAMBDA_EDGE      = 1.0
LAMBDA_TRANSFORM = 1.0
LAMBDA_BRIER     = 0.5

# ── Training stage epochs (Phase 3) ──────────────────────────────────────────
STAGE_A_EPOCHS   = 20       # transform head warm-up
STAGE_B_EPOCHS   = 20       # GAT (edge classification)
STAGE_C_EPOCHS   = 30       # joint fine-tune
QUICK_EPOCHS     = (10, 10, 15)  # quick CPU config

# ── Legacy MLP classifier settings (kept for ablation baseline) ───────────────
MLP_HIDDEN_DIM   = 256
MLP_DROPOUT      = 0.30
MLP_LR           = 1e-3
MLP_EPOCHS       = 60
MLP_BATCH_SIZE   = 32

# ── Graph / reconstruction settings ─────────────────────────────────────────
MST_WEIGHT_ATTR      = "weight"
CONFIDENCE_THRESHOLD = 0.50

# ── Dataset split fractions ───────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC  = 1 - TRAIN_FRAC - VAL_FRAC = 0.15

# ── Ensure directories exist at import time ───────────────────────────────────
for _d in [ORIGINALS_DIR, GENERATED_DIR, LABELS_DIR, MODELS_DIR, OUTPUTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
