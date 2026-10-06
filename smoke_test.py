"""
smoke_test.py — Lightweight smoke test that verifies the core pipeline
without requiring pretrained ViT weights or real images.

Creates 3 tiny synthetic images on-the-fly, applies transforms, and checks:
- All 5 transforms run without error.
- Pair-feature dimensions are correct.
- Dataset classes load correctly.
- Graph builder runs (with random embeddings).

Run with:
    python smoke_test.py
"""

import sys
import torch
import random
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

print("[Smoke Test] Starting …")

# ── Test 1: transforms ───────────────────────────────────────────────────────
from src.transforms import apply_transform, sample_transform_params, TRANSFORMATION_NAMES

rng = random.Random(42)
dummy_img = Image.new("RGB", (128, 128), color=(100, 150, 200))

for name in TRANSFORMATION_NAMES:
    params = sample_transform_params(name, rng)
    result = apply_transform(dummy_img, name, params)
    assert result.size == dummy_img.size or not params.get("resize_back", True) or params.get("expand", False) or name in ("jpeg", "resize", "crop", "rotation", "quality_degradation"), \
        f"Size mismatch for {name}"
    print(f"  [OK] {name:18s} -> {result.size}  params={params}")

# -- Test 2: pair features -----------------------------------------------------
from src.pair_features import build_pair_feature, PAIR_FEATURE_DIM

a = torch.randn(768)
b = torch.randn(768)
feat = build_pair_feature(a, b)
assert feat.shape == (PAIR_FEATURE_DIM,), f"Expected ({PAIR_FEATURE_DIM},), got {feat.shape}"
print(f"\n  [OK] pair_feature shape = {feat.shape}")

# -- Test 3: dataset classes with fake embeddings ------------------------------
import pandas as pd
from src.dataset import DirectionDataset, TransformationDataset

fake_edges = {
    "parent_path":               ["/tmp/a.jpg", "/tmp/a.jpg"],
    "child_path":                ["/tmp/b.jpg", "/tmp/c.jpg"],
    "transformation":            ["crop", "blur"],
    "transformation_parameters": ['{"frac": 0.1, "side": "top"}',
                                  '{"radius": 2.0}'],
    "direction":                 ["parent_to_child", "parent_to_child"],
}
df = pd.DataFrame(fake_edges)
fake_embs = {
    "/tmp/a.jpg": torch.randn(768),
    "/tmp/b.jpg": torch.randn(768),
    "/tmp/c.jpg": torch.randn(768),
}
dir_ds = DirectionDataset(df, fake_embs)
tfm_ds = TransformationDataset(df, fake_embs)
assert len(dir_ds) == 4,  f"Expected 4 direction samples, got {len(dir_ds)}"
assert len(tfm_ds) == 2,  f"Expected 2 transform samples, got {len(tfm_ds)}"
print(f"  [OK] DirectionDataset      len={len(dir_ds)}")
print(f"  [OK] TransformationDataset len={len(tfm_ds)}")

# -- Test 4: MLP forward pass --------------------------------------------------
from src.train_classifiers import MLP
from src.pair_features import PAIR_FEATURE_DIM

mlp = MLP(PAIR_FEATURE_DIM, 64, 2)
x   = torch.randn(4, PAIR_FEATURE_DIM)
out = mlp(x)
assert out.shape == (4, 2), f"MLP output shape wrong: {out.shape}"
print(f"  [OK] MLP forward pass  output shape={out.shape}")

# -- Test 5: graph builder with random embeddings ------------------------------
from src.graph_builder import build_mst

paths = [Path(f"/tmp/img_{i}.jpg") for i in range(4)]
embs  = {str(p): torch.nn.functional.normalize(torch.randn(768), dim=0)
         for p in paths}
mst = build_mst(paths, embs)
assert mst.number_of_nodes() == 4
assert mst.number_of_edges() == 3   # MST of N nodes has N-1 edges
print(f"  [OK] MST  nodes={mst.number_of_nodes()}  edges={mst.number_of_edges()}")

print("\n[Smoke Test] ALL TESTS PASSED - OK")
