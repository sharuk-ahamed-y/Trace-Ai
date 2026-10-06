"""
demo.py — End-to-end demonstration using the FIRST original image.

Picks the first original image, applies all 5 transformations, picks
3–6 of the resulting images, runs the full pipeline, and saves:
    outputs/provenance_graph.png
    outputs/predictions.csv
    outputs/predictions.json

Usage
-----
    python demo.py

Requirements
------------
1. At least one image in data/originals/
2. Trained models in models/  (run `python train.py` first)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch

from src.config import (
    ORIGINALS_DIR, GENERATED_DIR, MODELS_DIR,
    IMAGE_EXTENSIONS, OUTPUTS_DIR, TRANSFORMATION_NAMES,
)
from src.feature_extractor import extract_embeddings_cached
from src.train_classifiers import load_direction_model, load_transform_model
from src.graph_builder import build_mst, orient_mst
from src.visualize_graph import render_provenance_graph


def pick_demo_images() -> list[Path]:
    """
    Return a list of 4–6 images: the first original + its generated variants.
    """
    originals = sorted([
        p for p in ORIGINALS_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    if not originals:
        print("[ERROR] No images found in data/originals/. "
              "Please add images and run `python train.py` first.")
        sys.exit(1)

    base = originals[0]
    stem = base.stem

    # Collect generated children for this original
    children = sorted([
        p for p in GENERATED_DIR.iterdir()
        if p.stem.startswith(f"{stem}__")
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not children:
        print("[ERROR] No generated images found. Run `python train.py` first.")
        sys.exit(1)

    # Demo uses parent + all 5 children (or fewer if some are missing)
    demo_images = [base] + children[:5]
    demo_images = demo_images[:6]   # cap at 6 for readability

    print(f"[Demo] Selected {len(demo_images)} images:")
    for p in demo_images:
        print(f"  {p.name}")

    return demo_images


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}\n")

    # ── Check trained models ───────────────────────────────────────────────────
    if not (MODELS_DIR / "direction_mlp.pt").exists():
        print("[ERROR] Trained models not found in models/.")
        print("        Please run:  python train.py")
        sys.exit(1)

    # ── Pick demo images ───────────────────────────────────────────────────────
    image_paths = pick_demo_images()

    # ── Extract embeddings ─────────────────────────────────────────────────────
    print("\n[Step 1] Extracting ViT embeddings …")
    embeddings = extract_embeddings_cached(image_paths, device=device)

    # ── Load classifiers ───────────────────────────────────────────────────────
    print("[Step 2] Loading trained classifiers …")
    dir_model, dir_names = load_direction_model(device)
    tfm_model, tfm_names = load_transform_model(device)

    # ── Build MST ──────────────────────────────────────────────────────────────
    print("[Step 3] Building MST provenance graph …")
    mst = build_mst(image_paths, embeddings)
    print(f"  MST nodes: {mst.number_of_nodes()} | edges: {mst.number_of_edges()}")

    # ── Orient edges ───────────────────────────────────────────────────────────
    print("[Step 4] Orienting edges …")
    DG, records = orient_mst(
        mst, embeddings,
        dir_model, dir_names,
        tfm_model, tfm_names,
        device,
    )

    # ── Display predictions ────────────────────────────────────────────────────
    print("\n[Predictions]")
    print(f"  {'Source':<30}  {'Target':<30}  {'Transform':<12}  "
          f"{'TfmConf':>7}  {'DirConf':>7}")
    print("  " + "-" * 95)
    for r in records:
        src = Path(r["source"]).name[:28]
        tgt = Path(r["target"]).name[:28]
        print(f"  {src:<30}  {tgt:<30}  {r['predicted_transformation']:<12}  "
              f"{r['transformation_confidence']:>7.4f}  "
              f"{r['direction_confidence']:>7.4f}")

    # ── Save outputs ────────────────────────────────────────────────────────────
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    graph_path = OUTPUTS_DIR / "provenance_graph.png"
    render_provenance_graph(
        DG,
        output_path=graph_path,
        title="Demo — Image Provenance Graph",
    )

    csv_path  = OUTPUTS_DIR / "predictions.csv"
    json_path = OUTPUTS_DIR / "predictions.json"

    pd.DataFrame(records).to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\n[Done] Outputs saved:")
    print(f"  {graph_path}")
    print(f"  {csv_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
