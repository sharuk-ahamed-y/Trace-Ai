"""
feature_extractor.py — Pretrained ViT-B/16 embedding extractor.

Uses torchvision's ViT-B/16 with pretrained ImageNet weights.
The classification head is removed; the CLS token (dim=768) is returned
as the image embedding.

Embeddings are L2-normalised and cached to disk to avoid recomputation.
"""

import pickle
import sys
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

from src.config import (
    VIT_MODEL_NAME, EMBEDDING_DIM, VIT_IMAGE_SIZE, EMBEDDING_CACHE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing pipeline (matches ImageNet stats used during ViT training)
# ─────────────────────────────────────────────────────────────────────────────

_PREPROCESS = T.Compose([
    T.Resize((VIT_IMAGE_SIZE, VIT_IMAGE_SIZE), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# ViT wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ViTEmbedder(nn.Module):
    """
    Wraps torchvision ViT-B/16 and exposes only the CLS embedding.
    Weights are frozen; the model is eval-only.
    """

    def __init__(self) -> None:
        super().__init__()
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torchvision.models as tvm
            weights_cls = getattr(tvm, "ViT_B_16_Weights", None)
            if weights_cls is None:
                raise ImportError(
                    "torchvision.models.ViT_B_16_Weights not found. "
                    "Please upgrade torchvision:\n"
                    "    pip install --upgrade torchvision\n"
                )
            weights = weights_cls.IMAGENET1K_V1
            print("[ViT] Loading pretrained ViT-B/16 weights …")
            backbone = tvm.vit_b_16(weights=weights)
        except Exception as exc:
            print("\n" + "=" * 70)
            print("  ERROR: Could not load pretrained ViT-B/16 weights.")
            print("=" * 70)
            print(f"\n  Reason: {exc}\n")
            print(
                "  This prototype requires pretrained ViT-B/16 weights from\n"
                "  torchvision (ImageNet-1k).  To download them manually:\n\n"
                "      python - <<'EOF'\n"
                "      import torchvision.models as m\n"
                "      m.vit_b_16(weights=m.ViT_B_16_Weights.IMAGENET1K_V1)\n"
                "      EOF\n\n"
                "  Make sure you have an active internet connection and that\n"
                "  torchvision >= 0.13 is installed:\n"
                "      pip install torch torchvision --index-url "
                "https://download.pytorch.org/whl/cpu\n"
            )
            print("=" * 70 + "\n")
            sys.exit(1)

        # Remove classification head; keep encoder + patch embeddings
        self.patch_embed  = backbone.conv_proj
        self.class_token  = backbone.class_token
        self.encoder      = backbone.encoder
        self.seq_length   = backbone.seq_length   # 197 for ViT-B/16 224

        # Freeze everything
        for p in self.parameters():
            p.requires_grad_(False)

        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, 224, 224)
        returns : (B, 768)  — L2-normalised CLS token
        """
        # Patch embedding
        x = self.patch_embed(x)                     # (B, 768, 14, 14)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)            # (B, 196, 768)

        # Prepend class token
        cls = self.class_token.expand(B, -1, -1)    # (B, 1, 768)
        x = torch.cat([cls, x], dim=1)              # (B, 197, 768)

        # Transformer encoder (handles position embedding internally)
        x = self.encoder(x)                         # (B, 197, 768)

        # Extract CLS token and L2-normalise
        cls_out = x[:, 0]                            # (B, 768)
        cls_out = nn.functional.normalize(cls_out, dim=-1)
        return cls_out


# ─────────────────────────────────────────────────────────────────────────────
# Caching extractor
# ─────────────────────────────────────────────────────────────────────────────

_EMBEDDER: ViTEmbedder | None = None


def _get_embedder() -> ViTEmbedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = ViTEmbedder()
    return _EMBEDDER


def extract_embedding(
    image_path: Union[str, Path],
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Extract a 768-dim L2-normalised ViT embedding for a single image.

    Returns a CPU tensor of shape (768,).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedder = _get_embedder().to(device)
    img = Image.open(image_path).convert("RGB")
    tensor = _PREPROCESS(img).unsqueeze(0).to(device)   # (1, 3, 224, 224)
    emb = embedder(tensor).squeeze(0).cpu()             # (768,)
    return emb


def extract_embeddings_cached(
    image_paths: list[Path],
    device: torch.device | None = None,
    cache_path: Path = EMBEDDING_CACHE,
) -> dict[str, torch.Tensor]:
    """
    Extract embeddings for all image_paths, using/updating a disk cache.

    Returns a dict mapping str(path) → tensor(768,).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load existing cache
    cache: dict[str, torch.Tensor] = {}
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"[Cache] Loaded {len(cache)} cached embeddings from {cache_path.name}")

    # Only compute missing
    missing = [p for p in image_paths if str(p) not in cache]
    if missing:
        print(f"[ViT] Extracting embeddings for {len(missing)} image(s) …")
        for p in missing:
            key = str(p)
            cache[key] = extract_embedding(p, device=device)

        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"[Cache] Saved {len(cache)} embeddings -> {cache_path.name}")

    return {str(p): cache[str(p)] for p in image_paths}
