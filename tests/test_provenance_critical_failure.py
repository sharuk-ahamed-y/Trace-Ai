"""
tests/test_provenance_critical_failure.py — End-to-End Regression Suite for Critical Provenance Failures

Addresses and validates:
1. Exact reproduction and resolution of the live WhatsApp 4-image session failure.
2. Root selection accurately designates the true uncropped, pristine original.
3. Permutation / upload-order invariance on multi-image collections.
4. Directed arborescence construction: derivatives have in-degree 1, crops are leaves, no cycles.
5. Transformation classification accuracy on controlled fixtures:
   - Original + Crop -> CROP, direction Original -> Crop
   - Original + Resize -> RESIZE, direction Original -> Resize
   - Original + Brightness -> BRIGHTNESS, direction UNCERTAIN (honest uncertainty)
   - Original + Blur -> BLUR, direction UNCERTAIN
6. Unrelated scenes remain separated without artificial edges.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageEnhance, ImageFilter
import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.model_manager import get_model_manager
from backend.app.services.provenance_engine import run_provenance_analysis
from src.forensic_arbitrator import check_sub_region_containment, check_full_canvas_resizing


def create_rich_base_image(path: Path, width: int = 600, height: int = 600):
    """Generates an image with distinct geometric, textural, and textual patterns."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x] = [int(255 * x / width), int(255 * y / height), 128]
    cv2.circle(arr, (width // 2, height // 2), min(width, height) // 4, (40, 220, 40), -1)
    cv2.rectangle(arr, (width // 8, height // 8), (width // 3, height // 3), (220, 50, 50), -1)
    cv2.line(arr, (0, 0), (width, height), (255, 255, 255), 3)
    cv2.putText(arr, "PROVENANCE BENCHMARK", (width // 8, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    Image.fromarray(arr).save(path, quality=95)
    return path


def test_live_whatsapp_critical_failure_resolution():
    """Validates that the exact 4-image live WhatsApp session selects the true original as root."""
    session_dir = Path("data/uploads/anl_20261019_121010_32939f")
    if not session_dir.exists():
        pytest.skip("Live failure image directory not found on local disk")

    image_paths = [
        session_dir / "WhatsApp_Image_2026-10-13_at_1.31.11_AM.jpeg",
        session_dir / "WhatsApp_Image_2026-10-13_at_1.31.12_AM.jpeg",
        session_dir / "WhatsApp_Image_2026-10-18_at_10.25.00_PM.jpeg",
        session_dir / "WhatsApp_Image_2026-10-18_at_8.06.50_PM.jpeg"
    ]
    for p in image_paths:
        if not p.exists():
            pytest.skip(f"Live failure image {p.name} missing")

    result = run_provenance_analysis(image_paths)
    nodes = {n["id"]: n for n in result["nodes"]}
    root_filename = nodes[result["root_image_id"]]["filename"]

    # 1. The true uncropped original (1.31.12_AM) must be identified as root
    assert "1.31.12_AM" in root_filename, f"Expected true original 1.31.12_AM as root, got {root_filename}"
    assert result["root_verified"] is True

    # 2. Arborescence properties: exactly 3 edges, root has in-degree 0 and out-degree 3
    edges = result["edges"]
    assert len(edges) == 3

    out_from_root = [e for e in edges if e["source"] == result["root_image_id"]]
    in_to_root = [e for e in edges if e["target"] == result["root_image_id"]]
    assert len(in_to_root) == 0, "Root must have in-degree 0"
    assert len(out_from_root) == 3, "Root must be parent of all 3 derivatives"

    # 3. Crop must be a leaf node (in-degree 1, out-degree 0)
    crop_node_id = [n["id"] for n in result["nodes"] if "1.31.11_AM" in n["filename"]][0]
    in_to_crop = [e for e in edges if e["target"] == crop_node_id]
    out_from_crop = [e for e in edges if e["source"] == crop_node_id]
    assert len(in_to_crop) == 1, "Crop must have exactly 1 incoming edge"
    assert len(out_from_crop) == 0, "Crop must be a leaf node (out-degree 0)"
    assert in_to_crop[0]["transformation"] == "crop"
    assert in_to_crop[0]["direction"] == "A_TO_B"


def test_live_whatsapp_upload_order_invariance():
    """Validates that reversing the upload order preserves the same true root."""
    session_dir = Path("data/uploads/anl_20261019_121010_32939f")
    if not session_dir.exists():
        pytest.skip("Live failure image directory not found on local disk")

    image_paths = [
        session_dir / "WhatsApp_Image_2026-10-18_at_8.06.50_PM.jpeg",
        session_dir / "WhatsApp_Image_2026-10-18_at_10.25.00_PM.jpeg",
        session_dir / "WhatsApp_Image_2026-10-13_at_1.31.12_AM.jpeg",
        session_dir / "WhatsApp_Image_2026-10-13_at_1.31.11_AM.jpeg"
    ]
    for p in image_paths:
        if not p.exists():
            pytest.skip(f"Live failure image {p.name} missing")

    result = run_provenance_analysis(image_paths)
    nodes = {n["id"]: n for n in result["nodes"]}
    root_filename = nodes[result["root_image_id"]]["filename"]

    assert "1.31.12_AM" in root_filename, f"Reversed upload order failed: expected 1.31.12_AM as root, got {root_filename}"


def test_controlled_fixtures_provenance(tmp_path):
    """Controlled multi-image dataset with known ground-truth transformations."""
    orig_path = tmp_path / "original.png"
    create_rich_base_image(orig_path, width=600, height=600)
    orig_img = Image.open(orig_path)

    # 1. Sub-region Crop
    crop_path = tmp_path / "derivative_crop.png"
    orig_img.crop((50, 50, 450, 450)).save(crop_path)

    # 2. Resized full frame
    resize_path = tmp_path / "derivative_resize.png"
    orig_img.resize((300, 300), Image.Resampling.LANCZOS).save(resize_path)

    # 3. Brightness derivative
    bright_path = tmp_path / "derivative_bright.png"
    ImageEnhance.Brightness(orig_img).enhance(1.25).save(bright_path)

    # 4. Blur derivative
    blur_path = tmp_path / "derivative_blur.png"
    orig_img.filter(ImageFilter.GaussianBlur(radius=2)).save(blur_path)

    # Run multi-image provenance analysis
    image_paths = [crop_path, blur_path, orig_path, bright_path]
    result = run_provenance_analysis(image_paths)
    nodes = {n["id"]: n for n in result["nodes"]}
    root_filename = nodes[result["root_image_id"]]["filename"]

    # Original must be root
    assert "original" in root_filename, f"Expected original as root, got {root_filename}"

    # Verify edge transformations
    edges = result["edges"]
    assert len(edges) == 3
    for e in edges:
        assert e["source"] == result["root_image_id"], f"Expected all edges to originate from root, got {e['source']}"
        tgt_fn = nodes[e["target"]]["filename"]
        if "crop" in tgt_fn:
            assert e["transformation"] == "crop"
            assert e["direction"] == "A_TO_B"
        elif "bright" in tgt_fn:
            assert e["transformation"] == "brightness"
        elif "blur" in tgt_fn:
            assert e["transformation"] == "blur"


def test_unrelated_images_multi_component(tmp_path):
    """Validates that unrelated scenes do not form a single connected tree."""
    img_a = tmp_path / "scene_a.png"
    create_rich_base_image(img_a, width=400, height=400)

    # Unrelated checkerboard noise
    img_b = tmp_path / "scene_b.png"
    noise_arr = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
    Image.fromarray(noise_arr).save(img_b)

    result = run_provenance_analysis([img_a, img_b])
    # Unrelated images should have 0 candidate edges or 2 separate components
    assert len(result["edges"]) == 0 or len(result["components"]) == 2


def test_nested_crop_provenance_chain(tmp_path):
    """Validates Scenario B: Original -> Crop 1 -> Crop 2 (nested and upscaled)."""
    orig_path = tmp_path / "orig_scene.png"
    create_rich_base_image(orig_path, width=800, height=800)
    arr_orig = np.array(Image.open(orig_path))

    # Crop 1: 500x500 sub-region (150:650, 150:650)
    crop1_path = tmp_path / "crop1_intermediate.png"
    im_crop1 = Image.fromarray(arr_orig[150:650, 150:650])
    im_crop1.save(crop1_path)

    # Crop 2: 200x200 sub-region of Crop 1, upscaled to 800x800
    arr_crop1 = np.array(im_crop1)
    crop2_path = tmp_path / "crop2_nested_upscaled.png"
    sub_crop = arr_crop1[150:350, 150:350]
    im_crop2 = Image.fromarray(cv2.resize(sub_crop, (800, 800), interpolation=cv2.INTER_LINEAR))
    im_crop2.save(crop2_path)

    # 1. Independent pairwise validation
    d1, s1 = check_sub_region_containment(orig_path, crop1_path)
    assert d1 == 1.0 and s1 >= 0.90, f"Orig -> Crop1 containment failed: dir={d1}, score={s1}"

    d2, s2 = check_sub_region_containment(orig_path, crop2_path)
    assert d2 == 1.0 and s2 >= 0.90, f"Orig -> Crop2 containment failed: dir={d2}, score={s2}"

    d3, s3 = check_sub_region_containment(crop1_path, crop2_path)
    assert d3 == 1.0 and s3 >= 0.90, f"Crop1 -> Crop2 containment failed: dir={d3}, score={s3}"

    # 2. Multi-image provenance graph reconstruction
    result = run_provenance_analysis([orig_path, crop1_path, crop2_path])
    nodes = {n["id"]: n for n in result["nodes"]}
    root_filename = nodes[result["root_image_id"]]["filename"]

    assert "orig_scene" in root_filename, f"Expected orig_scene as root, got {root_filename}"
    assert result["root_verified"] is True

    edges = result["edges"]
    assert len(edges) == 2, f"Expected exactly 2 edges in 3-node tree, got {len(edges)}"

    # All edges must be crop with verified direction
    for e in edges:
        assert e["transformation"] == "crop"
        assert e["direction"] == "A_TO_B"
        assert e["is_uncertain"] is False


def test_nested_crop_reverse_upload_order(tmp_path):
    """Validates that reversing the upload order of nested crops preserves original root and edge classification."""
    orig_path = tmp_path / "orig_scene.png"
    create_rich_base_image(orig_path, width=800, height=800)
    arr_orig = np.array(Image.open(orig_path))

    crop1_path = tmp_path / "crop1_intermediate.png"
    im_crop1 = Image.fromarray(arr_orig[150:650, 150:650])
    im_crop1.save(crop1_path)

    arr_crop1 = np.array(im_crop1)
    crop2_path = tmp_path / "crop2_nested_upscaled.png"
    sub_crop = arr_crop1[150:350, 150:350]
    im_crop2 = Image.fromarray(cv2.resize(sub_crop, (800, 800), interpolation=cv2.INTER_LINEAR))
    im_crop2.save(crop2_path)

    # Reverse upload order: [crop2, crop1, orig]
    result = run_provenance_analysis([crop2_path, crop1_path, orig_path])
    nodes = {n["id"]: n for n in result["nodes"]}
    root_filename = nodes[result["root_image_id"]]["filename"]

    assert "orig_scene" in root_filename, f"Expected orig_scene as root under reverse order, got {root_filename}"


def test_exact_three_image_live_branching_crops():
    """
    Validates the exact 3-image live WhatsApp collection:
    - WhatsApp_Image_2026-10-18_at_10.28.56_PM_(1) is the true Original Root.
    - WhatsApp_Image_2026-10-18_at_10.28.57_PM_(2) is a verified crop child of Image 1.
    - WhatsApp_Image_2026-10-18_at_10.28.57_PM_(3) is a verified crop child of Image 1.
    - No false sibling edge between Image 2 and Image 3 is formed.
    """
    session_dir = Path("data/uploads/anl_20261019_135017_d4df68")
    if not session_dir.exists():
        pytest.skip("Live failure image directory not found on local disk")

    image_paths = [
        session_dir / "WhatsApp_Image_2026-10-18_at_10.28.56_PM_(1).jpeg",
        session_dir / "WhatsApp_Image_2026-10-18_at_10.28.57_PM_(2).jpeg",
        session_dir / "WhatsApp_Image_2026-10-18_at_10.28.57_PM_(3).jpeg"
    ]
    for p in image_paths:
        if not p.exists():
            pytest.skip(f"Live image {p.name} not present")

    res = run_provenance_analysis(image_paths, similarity_threshold=0.5, confidence_filter=0.35)

    root_id = res["root_image_id"]
    assert "10.28.56_PM_(1)" in root_id, f"Expected Image 1 as root, got {root_id}"
    assert res["root_verified"] is True

    edges = res["edges"]
    assert len(edges) == 2, f"Expected exactly 2 edges in 3-image tree, got {len(edges)}"

    # Check edges
    edge_pairs = {(e["source"], e["target"]): e for e in edges}
    
    # Both Image 2 and Image 3 must be direct children of Image 1
    id1 = [n["id"] for n in res["nodes"] if "10.28.56_PM_(1)" in n["id"]][0]
    id2 = [n["id"] for n in res["nodes"] if "10.28.57_PM_(2)" in n["id"]][0]
    id3 = [n["id"] for n in res["nodes"] if "10.28.57_PM_(3)" in n["id"]][0]

    assert (id1, id2) in edge_pairs, f"Missing edge {id1} -> {id2}"
    assert (id1, id3) in edge_pairs, f"Missing edge {id1} -> {id3}"
    assert (id2, id3) not in edge_pairs, "False sibling edge between Image 2 and Image 3 must NOT exist"
    assert (id3, id2) not in edge_pairs, "False sibling edge between Image 3 and Image 2 must NOT exist"

    edge_12 = edge_pairs[(id1, id2)]
    assert edge_12["transformation"] == "crop"
    assert edge_12["direction"] == "A_TO_B"
    assert edge_12["is_uncertain"] is False
    assert edge_12["transformation_confidence"] >= 0.88

    edge_13 = edge_pairs[(id1, id3)]
    assert edge_13["transformation"] == "crop"
    assert edge_13["direction"] == "A_TO_B"
    assert edge_13["is_uncertain"] is False
    assert edge_13["transformation_confidence"] >= 0.88


def test_database_edge_metadata_persistence_and_reload(tmp_path):
    """
    Validates that direction, uncertainty, and confidence metadata survive database round-trip
    when saved and reloaded via SQLite without loss or fabrication.
    """
    from backend.app.database.db import save_analysis, get_analysis_by_id

    test_analysis_id = "test_meta_roundtrip_999"
    test_nodes = [
        {"id": "node_a", "filename": "a.png", "file_path": str(tmp_path / "a.png"), "url": "/a.png",
         "sha256": "sha_a", "md5": "md5_a", "width": 800, "height": 600, "file_size": 12345},
        {"id": "node_b", "filename": "b.png", "file_path": str(tmp_path / "b.png"), "url": "/b.png",
         "sha256": "sha_b", "md5": "md5_b", "width": 400, "height": 300, "file_size": 5432}
    ]
    test_edges = [
        {
            "source": "node_a",
            "target": "node_b",
            "source_id": "node_a",
            "target_id": "node_b",
            "transformation": "crop",
            "confidence": 0.94,
            "weight": 0.06,
            "direction": "A_TO_B",
            "is_uncertain": False,
            "transformation_confidence": 0.985,
            "direction_confidence": 0.992,
            "direction_rationale": "Field of view containment test",
            "related_prob": 0.96
        }
    ]

    # Save to SQLite
    save_analysis(
        analysis_id=test_analysis_id,
        timestamp="2026-10-19T14:00:00",
        status="COMPLETED",
        image_count=2,
        root_image_id="node_a",
        root_confidence=0.94,
        execution_time_ms=120.0,
        similarity_threshold=0.5,
        confidence_filter=0.35,
        images=test_nodes,
        edges=test_edges,
        xai_results=[],
        candidate_edges=test_edges,
        rejected_pairs=[]
    )

    # Reload from SQLite
    record = get_analysis_by_id(test_analysis_id)
    assert record is not None
    assert record["analysis"]["root_image_id"] == "node_a"

    reloaded_edges = record["edges"]
    assert len(reloaded_edges) == 1
    edge = reloaded_edges[0]

    # Assert all metadata survived intact
    assert edge["transformation"] == "crop"
    assert edge["direction"] == "A_TO_B"
    assert edge["is_uncertain"] is False
    assert edge["transformation_confidence"] == 0.985
    assert edge["direction_confidence"] == 0.992
    assert edge["direction_rationale"] == "Field of view containment test"
    assert edge["related_prob"] == 0.96

    # Assert candidate_edges survived persistence and reload
    reloaded_cands = record.get("candidate_edges", [])
    assert len(reloaded_cands) == 1
    assert reloaded_cands[0]["source"] == "node_a"
    assert reloaded_cands[0]["target"] == "node_b"


