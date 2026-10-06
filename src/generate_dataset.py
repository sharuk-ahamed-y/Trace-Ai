"""
generate_dataset.py — Create controlled child images and write edges.csv.

Usage (from project root):
    python -m src.generate_dataset
or:
    python train.py   (calls this internally)
"""

import csv
import json
import random
import sys
from pathlib import Path

from PIL import Image

from src.config import (
    ORIGINALS_DIR, GENERATED_DIR, LABELS_DIR, EDGES_CSV,
    IMAGE_EXTENSIONS, TRANSFORMATION_NAMES, TRANSFORM_PARAMS, RANDOM_SEED,
)
from src.transforms import apply_transform, sample_transform_params


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

MIN_ORIGINALS = 3   # lower bar for smoke-tests; README asks for 20+


def _discover_originals() -> list[Path]:
    """Return sorted list of original image paths."""
    files = sorted([
        p for p in ORIGINALS_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    return files


def _check_originals(files: list[Path]) -> None:
    """Print clear instructions if not enough originals are present."""
    if len(files) == 0:
        print("\n" + "=" * 70)
        print("  NO IMAGES FOUND IN data/originals/")
        print("=" * 70)
        print(
            "\n  Please add at least 20 ordinary natural images (JPEG or PNG)\n"
            "  to the following directory:\n"
            f"\n      {ORIGINALS_DIR}\n\n"
            "  You can use any public-domain photos, e.g. from\n"
            "  https://unsplash.com/  or  https://www.pexels.com/ (free licence)\n\n"
            "  Accepted formats : .jpg  .jpeg  .png\n"
            "  Minimum count    : 20 images (≥3 for smoke-tests)\n\n"
            "  After placing the images, re-run:\n"
            "      python train.py\n"
        )
        print("=" * 70 + "\n")
        sys.exit(0)

    if len(files) < MIN_ORIGINALS:
        print(f"[WARNING] Only {len(files)} image(s) found in data/originals/. "
              f"The README recommends at least 20 for meaningful training.\n"
              f"Continuing with {len(files)} image(s).")


# ─────────────────────────────────────────────────────────────────────────────
# main generation logic
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(seed: int = RANDOM_SEED, overwrite: bool = False) -> list[dict]:
    """
    Generate child images for every (original, transformation) pair.

    Returns a list of edge dicts that is also written to EDGES_CSV.

    Parameters
    ----------
    seed      : random seed for reproducibility
    overwrite : if False, skip already-generated files
    """
    rng = random.Random(seed)
    files = _discover_originals()
    _check_originals(files)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    edges: list[dict] = []

    for orig_path in files:
        try:
            parent_img = Image.open(orig_path).convert("RGB")
        except Exception as exc:
            print(f"[SKIP] Cannot open {orig_path}: {exc}")
            continue

        for transform_name in TRANSFORMATION_NAMES:
            # Draw random parameters (reproducible per orig+transform combo)
            params = sample_transform_params(
                transform_name, rng, TRANSFORM_PARAMS[transform_name]
            )

            # Build output path
            stem = orig_path.stem
            suffix = ".jpg" if transform_name == "jpeg" else orig_path.suffix
            child_name = f"{stem}__{transform_name}{suffix}"
            child_path = GENERATED_DIR / child_name

            # Apply transformation
            if child_path.exists() and not overwrite:
                print(f"  [SKIP existing] {child_name}")
            else:
                child_img = apply_transform(parent_img, transform_name, params)
                if transform_name == "jpeg":
                    child_img.save(child_path, format="JPEG",
                                   quality=params.get("quality", 85))
                else:
                    child_img.save(child_path)
                print(f"  [GEN] {child_name}")

            edges.append({
                "parent_path":               str(orig_path.resolve()),
                "child_path":                str(child_path.resolve()),
                "transformation":            transform_name,
                "transformation_parameters": json.dumps(params),
                "direction":                 "parent_to_child",
            })

    # Write CSV
    fieldnames = [
        "parent_path", "child_path", "transformation",
        "transformation_parameters", "direction",
    ]
    with open(EDGES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges)

    print(f"\n[DONE] Generated {len(edges)} edges from {len(files)} originals.")
    print(f"       Labels saved -> {EDGES_CSV}")
    return edges


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_dataset()
