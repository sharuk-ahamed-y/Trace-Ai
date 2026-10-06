"""
infer.py — Run inference on a folder of images to produce a provenance graph.

Usage
-----
    python infer.py --images <folder_or_glob>
    python infer.py --images data/generated/photo1__*.jpg data/originals/photo1.jpg
    python infer.py --images data/originals/ data/generated/ --output-dir outputs/

Requirements
------------
- Trained models must exist in models/.  Run train.py first.
- Images must be .jpg / .jpeg / .png files.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch

from src.config import (
    IMAGE_EXTENSIONS, OUTPUTS_DIR, MODELS_DIR,
)
from src.feature_extractor import extract_embeddings_cached
from src.train_classifiers import load_direction_model, load_transform_model
from src.graph_builder import build_mst, orient_mst
from src.visualize_graph import render_provenance_graph


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a provenance graph from a set of images"
    )
    p.add_argument(
        "--images", nargs="+", required=True,
        help="Image files or directories to include",
    )
    p.add_argument(
        "--output-dir", type=Path, default=OUTPUTS_DIR,
        help="Where to write provenance_graph.png, predictions.csv, predictions.json",
    )
    return p.parse_args()


def collect_images(sources: list[str]) -> list[Path]:
    paths: list[Path] = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            paths.extend(sorted([
                f for f in p.iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS
            ]))
        elif p.exists() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
        else:
            print(f"[WARN] Skipping {src} — not found or unsupported format")
    return sorted(set(paths))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # ── Collect images ────────────────────────────────────────────────────────
    image_paths = collect_images(args.images)
    if len(image_paths) < 2:
        print("[ERROR] Need at least 2 images to build a graph.")
        sys.exit(1)
    print(f"[Infer] Processing {len(image_paths)} images …")

    # ── Check models exist ─────────────────────────────────────────────────────
    if not (MODELS_DIR / "direction_mlp.pt").exists():
        print("[ERROR] No trained direction model found. Run train.py first.")
        sys.exit(1)

    # ── Extract embeddings ────────────────────────────────────────────────────
    embeddings = extract_embeddings_cached(image_paths, device=device)

    # ── Load models ───────────────────────────────────────────────────────────
    dir_model,  dir_names  = load_direction_model(device)
    tfm_model,  tfm_names  = load_transform_model(device)

    # ── Build MST ─────────────────────────────────────────────────────────────
    mst = build_mst(image_paths, embeddings)

    # ── Orient edges ──────────────────────────────────────────────────────────
    DG, records = orient_mst(
        mst, embeddings,
        dir_model, dir_names,
        tfm_model, tfm_names,
        device,
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    render_provenance_graph(DG, output_path=out_dir / "provenance_graph.png")

    df_out = pd.DataFrame(records)
    df_out.to_csv(out_dir / "predictions.csv", index=False)
    print(f"[Save] predictions.csv -> {out_dir / 'predictions.csv'}")

    with open(out_dir / "predictions.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"[Save] predictions.json -> {out_dir / 'predictions.json'}")

    print("\n[Done] Inference complete.")


if __name__ == "__main__":
    main()
