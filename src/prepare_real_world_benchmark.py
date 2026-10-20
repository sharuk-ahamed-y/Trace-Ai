"""
src/prepare_real_world_benchmark.py — Controlled Real-World Benchmark Dataset Preparator

Generates controlled-transformation benchmark pairs from authorized real-world originals.

Safeguards & Compliance:
1. Reuses existing transformations from src/transforms.py.
2. Preserves existing contributor-confirmed real-world benchmark pairs without modification.
3. Outputs to a dedicated directory (data/real_world_benchmark/controlled_transforms/), completely separate
   from synthetic training data (data/provenance_dataset_v2/).
4. Strictly marks generated derivatives as 'controlled_transform' (never misrepresenting them as independent captures).
5. Enforces source-grouped split independence (zero source leakage across splits).
6. Generates negative pairs ONLY across confirmed distinct source images (never assuming lineage).
7. Validates manifest schema, duplicate pairs, file existence, and PIL readability before completion.
"""

import os
import sys
import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transforms import (
    TRANSFORMATION_NAMES, apply_transform, sample_transform_params
)
from src.evaluate_real_world import validate_manifest


DEFAULT_TRANSFORMS = [
    "crop", "resize", "jpeg", "blur", "brightness",
    "contrast", "noise", "perspective", "sharpen", "color"
]

# Contributor-confirmed mapping of authorized original images to independent physical scenes
DEFAULT_CONFIRMED_SCENE_MAPPING: Dict[str, str] = {
    "WhatsApp Image 2026-10-13 at 1.31.11 AM.jpeg": "real_src_0001",
    "WhatsApp Image 2026-10-13 at 1.31.12 AM.jpeg": "real_src_0001",
    "WhatsApp Image 2026-10-18 at 2.20.04 PM (1).jpeg": "real_src_0002",
    "WhatsApp Image 2026-10-18 at 2.20.04 PM (2).jpeg": "real_src_0003",
    "WhatsApp Image 2026-10-18 at 2.20.04 PM.jpeg": "real_src_0004",
    "WhatsApp Image 2026-10-18 at 2.20.05 PM (1).jpeg": "real_src_0005",
    "WhatsApp Image 2026-10-18 at 2.20.05 PM.jpeg": "real_src_0006",
    "WhatsApp Image 2026-10-18 at 2.20.06 PM (1).jpeg": "real_src_0007",
    "WhatsApp Image 2026-10-18 at 2.20.06 PM.jpeg": "real_src_0008",
    "WhatsApp Image 2026-10-18 at 2.20.07 PM (1).jpeg": "real_src_0009",
    "WhatsApp Image 2026-10-18 at 2.20.07 PM (2).jpeg": "real_src_0010",
    "WhatsApp Image 2026-10-18 at 2.20.07 PM.jpeg": "real_src_0011",
    "WhatsApp Image 2026-10-18 at 2.20.08 PM (1).jpeg": "real_src_0012",
    "WhatsApp Image 2026-10-18 at 2.20.08 PM (2).jpeg": "real_src_0013",
    "WhatsApp Image 2026-10-18 at 2.20.08 PM (3).jpeg": "real_src_0014",
    "WhatsApp Image 2026-10-18 at 2.20.08 PM.jpeg": "real_src_0015",
    "WhatsApp Image 2026-10-18 at 2.20.09 PM (1).jpeg": "real_src_0016",
    "WhatsApp Image 2026-10-18 at 2.20.09 PM (2).jpeg": "real_src_0017",
    "WhatsApp Image 2026-10-18 at 2.20.09 PM (3).jpeg": "real_src_0018",
    "WhatsApp Image 2026-10-18 at 2.20.09 PM.jpeg": "real_src_0019",
    "WhatsApp Image 2026-10-18 at 2.20.10 PM (1).jpeg": "real_src_0020",
    "WhatsApp Image 2026-10-18 at 2.20.10 PM (2).jpeg": "real_src_0021",
    "WhatsApp Image 2026-10-18 at 2.20.10 PM.jpeg": "real_src_0022",
    "WhatsApp Image 2026-10-18 at 2.20.11 PM (1).jpeg": "real_src_0023",
    "WhatsApp Image 2026-10-18 at 2.20.11 PM.jpeg": "real_src_0024",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a controlled-transform real-world benchmark dataset from authorized originals."
    )
    parser.add_argument(
        "--input-images-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "real_world_benchmark" / "images"),
        help="Directory containing authorized original real-world benchmark images."
    )
    parser.add_argument(
        "--existing-manifest",
        type=str,
        default=str(PROJECT_ROOT / "data" / "real_world_benchmark" / "real_pairs_manifest.csv"),
        help="Path to existing manifest containing contributor-confirmed pairs to preserve."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "real_world_benchmark" / "controlled_transforms"),
        help="Output directory for controlled-transform benchmark files."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic transformation parameter sampling."
    )
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=DEFAULT_TRANSFORMS,
        help="List of transformations to apply."
    )
    parser.add_argument(
        "--source-mapping",
        type=str,
        default=None,
        help="Optional path to JSON file specifying custom image-to-source-ID mapping."
    )
    parser.add_argument(
        "--negative-strategy",
        type=str,
        default="ring",
        choices=["ring", "none"],
        help="Strategy for generating cross-scene negative pairs across distinct confirmed physical sources."
    )
    return parser.parse_args()


def prepare_benchmark_dataset(
    input_images_dir: Path,
    existing_manifest_path: Path,
    output_dir: Path,
    seed: int = 42,
    transforms: List[str] = None,
    source_mapping: Dict[str, str] = None,
    negative_strategy: str = "ring"
) -> Tuple[int, int, Path]:
    """
    Builds the controlled-transform benchmark dataset.
    Returns (num_positive_pairs, num_negative_pairs, manifest_path).
    """
    if transforms is None:
        transforms = DEFAULT_TRANSFORMS
    if source_mapping is None:
        source_mapping = DEFAULT_CONFIRMED_SCENE_MAPPING

    rng = random.Random(seed)

    images_out_dir = output_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out_path = output_dir / "manifest.csv"
    params_log_path = output_dir / "transform_params.json"

    print("=" * 80)
    print("TRACE AI — Real-World Benchmark Dataset Preparation (Controlled Transforms)")
    print("=" * 80)
    print(f"[BenchmarkPrep] Source directory: {input_images_dir}")
    print(f"[BenchmarkPrep] Output directory: {output_dir}")
    print(f"[BenchmarkPrep] Deterministic Seed: {seed}")

    # Step 1: Discover and verify authorized base images
    if not input_images_dir.exists():
        raise FileNotFoundError(f"Input images directory does not exist: {input_images_dir}")

    # Discover images
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    input_images = sorted([f for f in input_images_dir.iterdir() if f.suffix.lower() in allowed_exts])

    if not input_images:
        raise ValueError(f"No valid authorized image files found in {input_images_dir}")

    print(f"[BenchmarkPrep] Found {len(input_images)} input image(s):")
    for img_p in input_images:
        print(f"  - {img_p.name} ({img_p.stat().st_size:,} bytes)")

    # Copy input images into output/images/ to ensure self-contained directory
    for img_p in input_images:
        dest_p = images_out_dir / img_p.name
        if not dest_p.exists() or dest_p.stat().st_size != img_p.stat().st_size:
            shutil.copy2(img_p, dest_p)

    # Step 2: Read and preserve existing contributor-confirmed pairs
    preserved_rows = []
    seen_pair_ids = set()
    seen_pair_tuples = set()
    sources_detected = set()

    if existing_manifest_path.exists():
        import pandas as pd
        df_exist = pd.read_csv(existing_manifest_path)
        print(f"[BenchmarkPrep] Reading existing verified manifest: {existing_manifest_path} ({len(df_exist)} pairs)")
        for idx, row in df_exist.iterrows():
            pid = str(row["pair_id"]).strip()
            # Skip placeholder rows if any
            if pid.startswith("placeholder_"):
                continue
            row_dict = {
                "pair_id": pid,
                "image_a": str(row["image_a"]).strip(),
                "image_b": str(row["image_b"]).strip(),
                "source_a": str(row["source_a"]).strip(),
                "source_b": str(row["source_b"]).strip(),
                "related": int(row["related"]),
                "ground_truth_direction": str(row["ground_truth_direction"]).strip().upper(),
                "applied_transformations": str(row.get("applied_transformations", "crop")).strip(),
                "split": str(row.get("split", "benchmark_test")).strip(),
                "notes": str(row.get("notes", "Contributor-confirmed real-world pair")).strip()
            }
            preserved_rows.append(row_dict)
            seen_pair_ids.add(pid)
            pair_key = tuple(sorted([row_dict["image_a"], row_dict["image_b"]]))
            seen_pair_tuples.add(pair_key)
            sources_detected.add(row_dict["source_a"])
            sources_detected.add(row_dict["source_b"])

    print(f"[BenchmarkPrep] Preserving {len(preserved_rows)} contributor-confirmed pair(s).")

    # Map sources and detect canonical base images from input images
    source_to_base_image: Dict[str, str] = {}
    for img_p in input_images:
        src_id = source_mapping.get(img_p.name)
        if src_id:
            sources_detected.add(src_id)
            if src_id not in source_to_base_image:
                source_to_base_image[src_id] = img_p.name
            elif src_id == "real_src_0001" and img_p.name == "WhatsApp Image 2026-10-13 at 1.31.12 AM.jpeg":
                # Prefer full uncropped root capture with key for real_src_0001
                source_to_base_image[src_id] = img_p.name

    # Step 3: Select base image for controlled transformations
    base_image_name = "WhatsApp Image 2026-10-13 at 1.31.12 AM.jpeg"
    base_image_path = input_images_dir / base_image_name
    if not base_image_path.exists():
        # Fallback to first available image
        base_image_path = input_images[0]
        base_image_name = base_image_path.name

    base_source_id = "real_src_0001"
    sources_detected.add(base_source_id)

    print(f"[BenchmarkPrep] Base image for controlled transforms: {base_image_name} (Source: {base_source_id})")

    with Image.open(base_image_path).convert("RGB") as base_pil:
        orig_w, orig_h = base_pil.size
        print(f"[BenchmarkPrep] Base dimensions: {orig_w}x{orig_h} px")

        new_pairs = []
        applied_params_log = {}
        pair_counter = len(preserved_rows) + 1

        for tfm_name in transforms:
            if tfm_name not in TRANSFORMATION_NAMES:
                print(f"[BenchmarkPrep] WARNING: Skipping unsupported transformation '{tfm_name}'")
                continue

            params = sample_transform_params(tfm_name, rng)
            # For realistic benchmark: when applying crop, do not force resize_back so resolution changes naturally
            if tfm_name == "crop":
                params["resize_back"] = False

            transformed_img = apply_transform(base_pil, tfm_name, params)
            out_filename = f"ct_{base_source_id}_{tfm_name}.jpg"
            out_filepath = images_out_dir / out_filename

            # Save transformed image
            # If JPEG transform, use quality from params; otherwise quality 90
            q = int(params.get("quality", 90)) if tfm_name == "jpeg" else 90
            transformed_img.save(out_filepath, format="JPEG", quality=q)

            applied_params_log[out_filename] = {
                "base_image": base_image_name,
                "source_id": base_source_id,
                "transformation": tfm_name,
                "parameters": params,
                "output_dimensions": list(transformed_img.size)
            }

            # Create Positive Pair: Base -> Transformed
            pair_id = f"ct_pair_{pair_counter:04d}"
            pair_key = tuple(sorted([base_image_name, out_filename]))
            if pair_key not in seen_pair_tuples:
                pair_dict = {
                    "pair_id": pair_id,
                    "image_a": base_image_name,
                    "image_b": out_filename,
                    "source_a": base_source_id,
                    "source_b": base_source_id,
                    "related": 1,
                    "ground_truth_direction": "A_TO_B",
                    "applied_transformations": tfm_name,
                    "split": "benchmark_test",
                    "notes": f"controlled_transform: {tfm_name} from {base_image_name}"
                }
                new_pairs.append(pair_dict)
                seen_pair_ids.add(pair_id)
                seen_pair_tuples.add(pair_key)
                pair_counter += 1

    # Step 4: Negative Pairs (Distinct Authorized Sources Only)
    unrelated_pairs = []
    distinct_sources = sorted(list(source_to_base_image.keys()))

    if len(distinct_sources) >= 2 and negative_strategy == "ring":
        print(f"[BenchmarkPrep] Detected {len(distinct_sources)} distinct confirmed sources. Generating cross-source negative pairs...")
        neg_counter = 1
        n_src = len(distinct_sources)
        for i in range(n_src):
            s_a = distinct_sources[i]
            s_b = distinct_sources[(i + 1) % n_src]
            img_a = source_to_base_image[s_a]
            img_b = source_to_base_image[s_b]

            pair_key = tuple(sorted([img_a, img_b]))
            if pair_key not in seen_pair_tuples:
                pair_id = f"neg_pair_{neg_counter:04d}"
                pair_dict = {
                    "pair_id": pair_id,
                    "image_a": img_a,
                    "image_b": img_b,
                    "source_a": s_a,
                    "source_b": s_b,
                    "related": 0,
                    "ground_truth_direction": "NONE",
                    "applied_transformations": "none",
                    "split": "benchmark_test",
                    "notes": f"cross_source_negative: distinct authorized physical scenes ({s_a} vs {s_b})"
                }
                unrelated_pairs.append(pair_dict)
                seen_pair_ids.add(pair_id)
                seen_pair_tuples.add(pair_key)
                neg_counter += 1
        print(f"[BenchmarkPrep] Generated {len(unrelated_pairs)} authorized cross-source negative pairs across {len(distinct_sources)} independent scenes.")
    else:
        print(f"[BenchmarkPrep] Notice: Less than 2 distinct authorized sources present or negative strategy is 'none'.")
        print(f"[BenchmarkPrep] Safeguard Enforced: 0 negative pairs generated. Never assume images are unrelated.")

    # Step 5: Assemble and write manifest
    all_rows = preserved_rows + new_pairs + unrelated_pairs

    import pandas as pd
    manifest_df = pd.DataFrame(all_rows)
    manifest_df.to_csv(manifest_out_path, index=False)
    print(f"[BenchmarkPrep] Written manifest to {manifest_out_path} ({len(manifest_df)} total pairs)")

    # Save parameters log
    with open(params_log_path, "w", encoding="utf-8") as f:
        json.dump(applied_params_log, f, indent=2)
    print(f"[BenchmarkPrep] Logged transformation parameters to {params_log_path}")

    # Step 6: Validate generated manifest
    is_valid, validation_errors = validate_manifest(manifest_df, images_out_dir)
    if not is_valid:
        raise ValueError(f"Generated manifest failed validation: {validation_errors}")

    # Verify PIL readability for every image in manifest
    for _, row in manifest_df.iterrows():
        pa = images_out_dir / str(row["image_a"])
        pb = images_out_dir / str(row["image_b"])
        if not pa.exists() or not pb.exists():
            raise FileNotFoundError(f"Missing referenced image: {pa} or {pb}")
        with Image.open(pa) as im_a, Image.open(pb) as im_b:
            im_a.verify()
            im_b.verify()

    print("[BenchmarkPrep] Manifest and image file verification: 100% PASSED.")

    n_pos = sum(1 for r in all_rows if r["related"] == 1)
    n_neg = sum(1 for r in all_rows if r["related"] == 0)
    n_ct = sum(1 for r in all_rows if str(r.get("notes", "")).startswith("controlled_transform:"))
    n_real_pos = n_pos - n_ct
    n_scenes = len(distinct_sources) if distinct_sources else len(sources_detected)

    print("=" * 80)
    print(f"[BenchmarkPrep] Summary: {len(all_rows)} total pairs prepared.")
    print(f"  - Independent physical scenes: {n_scenes}")
    print(f"  - Positive pairs: {n_pos} ({n_real_pos} contributor-confirmed + {n_ct} controlled_transforms)")
    print(f"  - Negative pairs: {n_neg} (Authorized cross-scene pairs across distinct sources)")
    print("=" * 80)

    return n_pos, n_neg, manifest_out_path


def main():
    args = parse_args()
    source_mapping = None
    if args.source_mapping:
        with open(args.source_mapping, "r", encoding="utf-8") as f:
            source_mapping = json.load(f)

    prepare_benchmark_dataset(
        input_images_dir=Path(args.input_images_dir),
        existing_manifest_path=Path(args.existing_manifest),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        transforms=args.transforms,
        source_mapping=source_mapping,
        negative_strategy=args.negative_strategy
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
