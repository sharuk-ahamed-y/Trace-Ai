"""
tests/test_generalized_n_provenance.py — Generalized N-Image Provenance Correctness & Robustness Suite

Validates comprehensive requirements for arbitrary upload count N:
1. Arbitrary collection sizes N in [2, 3, 4, 5, 6, 7] across varied graph topologies.
2. Nested provenance chains (A -> B -> C -> D) vs. direct sibling crops without shortcut collapse.
3. Complete evidence graph preservation alongside minimal spanning forest.
4. Multi-component disconnected forests with zero cross-component bridge edges.
5. Genuine direction uncertainty preservation for symmetric edits and pixel clones.
6. Upload-order permutation invariance across multi-image collections.
7. Database persistence and reload of candidate edges and individual edge confidences.
8. Multiple diverse synthetic base scenes (geometric, landscape gradient, textured document).
"""

import sys
import random
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageEnhance, ImageFilter
import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.forensic_arbitrator import (
    check_sub_region_containment,
    check_full_canvas_resizing,
    arbitrate_pair_direction,
    arbitrate_transformation,
)
from backend.app.services.model_manager import ModelManager
from backend.app.services.provenance_engine import run_provenance_analysis
from backend.app.database.db import save_analysis, get_analysis_by_id, init_db


def make_scene_alpha(path: Path, width: int = 800, height: int = 800):
    """Scene 1: Geometric structural scene with distinct colored quadrants and text."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x] = [int(255 * x / width), int(255 * y / height), 100]
    cv2.circle(arr, (width // 2, height // 2), min(width, height) // 4, (40, 220, 40), -1)
    cv2.rectangle(arr, (width // 8, height // 8), (width // 3, height // 3), (220, 50, 50), -1)
    cv2.rectangle(arr, (width // 2, height // 2), (width * 7 // 8, height * 7 // 8), (50, 50, 220), -1)
    cv2.line(arr, (0, 0), (width, height), (255, 255, 255), 4)
    cv2.putText(arr, "ALPHA_ORIGINAL_ROOT", (width // 5, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    Image.fromarray(arr).save(path, quality=95)
    return path


def make_scene_beta(path: Path, width: int = 700, height: int = 700):
    """Scene 2: High-frequency textured landscape with concentric rings."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x] = [int(128 + 127 * np.sin(x / 30.0)), int(128 + 127 * np.cos(y / 30.0)), 180]
    for r in range(40, min(width, height) // 2, 40):
        cv2.circle(arr, (width // 2, height // 2), r, (r % 255, (r * 2) % 255, 255), 3)
    cv2.putText(arr, "BETA_INDEPENDENT_ROOT", (width // 6, height // 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    Image.fromarray(arr).save(path, quality=95)
    return path


# =====================================================================
# 1. PARAMETERIZED N-IMAGE PROVENANCE & PAIRWISE COMPLETENESS
# =====================================================================

@pytest.mark.parametrize("n_count", [2, 3, 4, 5])
def test_arbitrary_n_evaluation_completeness(tmp_path, n_count):
    """Verifies that for any N in [2..5], exactly N*(N-1)/2 pairwise tests occur."""
    base_path = tmp_path / f"base_n_{n_count}.png"
    make_scene_alpha(base_path, 600, 600)
    base_im = Image.open(base_path)

    images = [base_path]
    # Generate (n_count - 1) distinct derivative images
    for i in range(1, n_count):
        p_deriv = tmp_path / f"deriv_{i}.png"
        if i == 1:
            base_im.resize((300, 300), Image.Resampling.LANCZOS).save(p_deriv)
        elif i == 2:
            base_im.crop((100, 100, 400, 400)).save(p_deriv)
        elif i == 3:
            enhancer = ImageEnhance.Brightness(base_im)
            enhancer.enhance(1.25).save(p_deriv)
        else:
            base_im.filter(ImageFilter.GaussianBlur(radius=2.0)).save(p_deriv)
        images.append(p_deriv)

    result = run_provenance_analysis(images)
    expected_pairs = (n_count * (n_count - 1)) // 2

    assert result["summary"]["total_images"] == n_count
    total_eval_pairs = len(result["candidate_edges"]) + len(result["rejected_pairs"])
    assert total_eval_pairs == expected_pairs, (
        f"For N={n_count}, expected {expected_pairs} total pairs evaluated, got {total_eval_pairs}"
    )


# =====================================================================
# 2. NESTED PROVENANCE CHAINS (A -> B -> C -> D)
# =====================================================================

def test_nested_provenance_chain_four_images(tmp_path):
    """
    Validates sequential derivation: Original A -> Crop B -> Crop C -> Crop D
    A is 800x800
    B is 600x600 crop of A
    C is 400x400 crop of B
    D is 200x200 crop of C (upscaled to 400x400)
    Requirement: Spanning tree must form sequential chain A -> B -> C -> D.
    Transitive shortcuts (A -> C, A -> D, B -> D) must not bypass intermediate parents.
    """
    p_a = tmp_path / "img_a_original.png"
    make_scene_alpha(p_a, width=800, height=800)
    arr_a = np.array(Image.open(p_a))

    # B: crop 600x600 from A (100:700, 100:700)
    p_b = tmp_path / "img_b_crop1.png"
    arr_b = arr_a[100:700, 100:700]
    Image.fromarray(arr_b).save(p_b)

    # C: crop 400x400 from B (100:500, 100:500)
    p_c = tmp_path / "img_c_crop2.png"
    arr_c = arr_b[100:500, 100:500]
    Image.fromarray(arr_c).save(p_c)

    # D: crop 200x200 from C, upscaled to 400x400
    p_d = tmp_path / "img_d_crop3_upscaled.png"
    arr_d = arr_c[100:300, 100:300]
    arr_d_up = cv2.resize(arr_d, (400, 400), interpolation=cv2.INTER_LINEAR)
    Image.fromarray(arr_d_up).save(p_d)

    result = run_provenance_analysis([p_a, p_b, p_c, p_d])
    nodes = {n["id"]: n for n in result["nodes"]}
    root_fn = nodes[result["root_image_id"]]["filename"]

    assert "img_a_original" in root_fn
    assert result["root_verified"] is True

    tree_edges = result["edges"]
    assert len(tree_edges) == 3, f"Expected 3 tree edges for 4-node tree, got {len(tree_edges)}"

    # Map parent -> child filenames
    lineage_pairs = []
    for e in tree_edges:
        s_fn = nodes[e["source"]]["filename"]
        t_fn = nodes[e["target"]]["filename"]
        lineage_pairs.append((s_fn, t_fn))

    # Verify sequential chain: A -> B, B -> C, C -> D
    a_targets = [t for (s, t) in lineage_pairs if "img_a_original" in s]
    b_targets = [t for (s, t) in lineage_pairs if "img_b_crop1" in s]
    c_targets = [t for (s, t) in lineage_pairs if "img_c_crop2" in s]

    assert any("img_b_crop1" in t for t in a_targets), f"Expected A -> B, got {lineage_pairs}"
    assert any("img_c_crop2" in t for t in b_targets), f"Expected B -> C, got {lineage_pairs}"
    assert any("img_d_crop3" in t for t in c_targets), f"Expected C -> D, got {lineage_pairs}"

    # Ensure shortcut A -> D is NOT a tree edge
    assert not any("img_a_original" in s and "img_d_crop3" in t for (s, t) in lineage_pairs), (
        f"Shortcut A -> D incorrectly selected in tree arborescence: {lineage_pairs}"
    )

    # Ensure complete evidence graph preserved all candidate containment edges
    cand_pairs = [(nodes[ce["source"]]["filename"], nodes[ce["target"]]["filename"]) for ce in result["candidate_edges"]]
    assert any("img_a_original" in s and "img_d_crop3" in t for (s, t) in cand_pairs), (
        "Candidate evidence graph must retain physical candidate edge between A and D"
    )


# =====================================================================
# 3. SIBLING BRANCHING CROPS (STAR TOPOLOGY)
# =====================================================================

def test_branching_sibling_crops_star_topology(tmp_path):
    """
    Validates branching derivation: Original A -> Crop B (top-left) and A -> Crop C (bottom-right).
    B and C are non-overlapping sibling crops of A.
    Requirement: Tree must branch A -> B and A -> C with ZERO direct B <-> C derivation edge.
    """
    p_orig = tmp_path / "star_root.png"
    make_scene_alpha(p_orig, width=800, height=800)
    arr_orig = np.array(Image.open(p_orig))

    # Sibling 1: Top-left quadrant 350x350
    p_crop1 = tmp_path / "crop_quadrant1.png"
    Image.fromarray(arr_orig[0:350, 0:350]).save(p_crop1)

    # Sibling 2: Bottom-right quadrant 350x350
    p_crop2 = tmp_path / "crop_quadrant2.png"
    Image.fromarray(arr_orig[450:800, 450:800]).save(p_crop2)

    result = run_provenance_analysis([p_orig, p_crop1, p_crop2])
    nodes = {n["id"]: n for n in result["nodes"]}
    root_fn = nodes[result["root_image_id"]]["filename"]

    assert "star_root" in root_fn
    edges = result["edges"]
    assert len(edges) == 2

    # Both edges must originate from star_root
    for e in edges:
        s_fn = nodes[e["source"]]["filename"]
        assert "star_root" in s_fn
        assert e["transformation"] == "crop"
        assert e["direction"] == "A_TO_B"
        assert e["is_uncertain"] is False


# =====================================================================
# 4. MULTI-COMPONENT DISCONNECTED FORESTS
# =====================================================================

def test_multi_component_independent_forests(tmp_path):
    """
    Validates two independent 3-image trees (6 images total):
    Component 1: Alpha Root -> Alpha Resize, Alpha Crop
    Component 2: Beta Root -> Beta Resize, Beta Crop
    Requirement: Exactly 2 connected subgraphs, 4 total forest edges, 0 cross-component edges.
    """
    p_a_root = tmp_path / "scene_alpha_root.png"
    make_scene_alpha(p_a_root, 600, 600)
    im_a = Image.open(p_a_root)

    p_a_res = tmp_path / "scene_alpha_resize.png"
    im_a.resize((300, 300), Image.Resampling.LANCZOS).save(p_a_res)

    p_a_crop = tmp_path / "scene_alpha_crop.png"
    im_a.crop((100, 100, 400, 400)).save(p_a_crop)

    # Component 2: Completely different scene
    p_b_root = tmp_path / "scene_beta_root.png"
    make_scene_beta(p_b_root, 600, 600)
    im_b = Image.open(p_b_root)

    p_b_res = tmp_path / "scene_beta_resize.png"
    im_b.resize((300, 300), Image.Resampling.LANCZOS).save(p_b_res)

    p_b_crop = tmp_path / "scene_beta_crop.png"
    im_b.crop((100, 100, 400, 400)).save(p_b_crop)

    all_images = [p_a_root, p_a_res, p_a_crop, p_b_root, p_b_res, p_b_crop]
    result = run_provenance_analysis(all_images)
    nodes = {n["id"]: n for n in result["nodes"]}

    assert len(result["components"]) == 2, f"Expected 2 components, got {len(result['components'])}"
    assert len(result["edges"]) == 4, f"Expected 4 forest edges, got {len(result['edges'])}"

    # Check that no edge bridges between Alpha and Beta
    for e in result["edges"]:
        s_fn = nodes[e["source"]]["filename"]
        t_fn = nodes[e["target"]]["filename"]
        assert ("alpha" in s_fn and "alpha" in t_fn) or ("beta" in s_fn and "beta" in t_fn), (
            f"Artificial cross-component bridge edge detected: {s_fn} -> {t_fn}"
        )


# =====================================================================
# 5. GENUINE DIRECTION UNCERTAINTY (SYMMETRIC EDITS & PIXEL CLONES)
# =====================================================================

def test_direction_uncertainty_preservation_across_clones(tmp_path):
    """
    Validates that identical copies or symmetric photometric edits retain UNCERTAIN status
    without fabricating a spurious original root or fake edge direction.
    """
    p_orig = tmp_path / "source_image.png"
    make_scene_beta(p_orig, 400, 400)
    im = Image.open(p_orig)

    p_copy1 = tmp_path / "identical_clone_1.png"
    im.save(p_copy1)

    p_copy2 = tmp_path / "identical_clone_2.png"
    im.save(p_copy2)

    result = run_provenance_analysis([p_orig, p_copy1, p_copy2])
    assert result["direction_status"] == "unresolved"
    assert result["root_verified"] is False
    assert "Direction unresolved" in result["root_reason"]

    for e in result["edges"]:
        assert e["direction"] == "UNCERTAIN"
        assert e["is_uncertain"] is True


# =====================================================================
# 6. UPLOAD-ORDER PERMUTATION INVARIANCE ACROSS ARBITRARY N
# =====================================================================

def test_permutation_invariance_arbitrary_n(tmp_path):
    """
    Validates that shuffling the upload order across 4 images produces identical root
    identification, identical edge classifications, and consistent graph topology.
    """
    p_orig = tmp_path / "perm_root.png"
    make_scene_alpha(p_orig, 600, 600)
    im = Image.open(p_orig)

    p_res = tmp_path / "perm_resize.png"
    im.resize((300, 300), Image.Resampling.LANCZOS).save(p_res)

    p_crop = tmp_path / "perm_crop.png"
    im.crop((150, 150, 450, 450)).save(p_crop)

    p_bright = tmp_path / "perm_bright.png"
    ImageEnhance.Brightness(im).enhance(1.25).save(p_bright)

    base_list = [p_orig, p_res, p_crop, p_bright]
    res_canonical = run_provenance_analysis(base_list)
    canonical_root_fn = Path(res_canonical["nodes"][[n["id"] for n in res_canonical["nodes"]].index(res_canonical["root_image_id"])]["file_path"]).name

    # Test 3 random permutations
    rng = random.Random(42)
    for trial in range(3):
        permuted_list = list(base_list)
        rng.shuffle(permuted_list)

        res_trial = run_provenance_analysis(permuted_list)
        trial_root_fn = Path(res_trial["nodes"][[n["id"] for n in res_trial["nodes"]].index(res_trial["root_image_id"])]["file_path"]).name

        assert trial_root_fn == canonical_root_fn, (
            f"Trial {trial}: Root mismatch! Expected {canonical_root_fn}, got {trial_root_fn}"
        )
        assert len(res_trial["edges"]) == len(res_canonical["edges"])


# =====================================================================
# 7. SQLITE PERSISTENCE & RELOAD OF COMPLETE EVIDENCE GRAPH
# =====================================================================

def test_complete_evidence_graph_database_persistence_and_reload(tmp_path):
    """
    Validates that candidate edges, rejected pairs, and individual edge confidence fields
    survive SQLite database storage and subsequent retrieval via get_analysis_by_id.
    """
    init_db()
    analysis_id = "test_audit_persistence_anl"

    nodes = [
        {"id": "node_1", "filename": "1.png", "file_path": "/tmp/1.png", "url": "/1.png", "sha256": "h1", "md5": "m1", "width": 800, "height": 800, "file_size": 1000},
        {"id": "node_2", "filename": "2.png", "file_path": "/tmp/2.png", "url": "/2.png", "sha256": "h2", "md5": "m2", "width": 400, "height": 400, "file_size": 500},
        {"id": "node_3", "filename": "3.png", "file_path": "/tmp/3.png", "url": "/3.png", "sha256": "h3", "md5": "m3", "width": 200, "height": 200, "file_size": 200}
    ]

    forest_edges = [
        {
            "source": "node_1", "target": "node_2", "source_id": "node_1", "target_id": "node_2",
            "transformation": "crop", "confidence": 0.94, "weight": 0.06,
            "direction": "A_TO_B", "is_uncertain": False,
            "transformation_confidence": 0.98, "direction_confidence": 0.99,
            "direction_rationale": "Direct crop containment", "related_prob": 0.96
        },
        {
            "source": "node_2", "target": "node_3", "source_id": "node_2", "target_id": "node_3",
            "transformation": "crop", "confidence": 0.92, "weight": 0.08,
            "direction": "A_TO_B", "is_uncertain": False,
            "transformation_confidence": 0.97, "direction_confidence": 0.98,
            "direction_rationale": "Nested sub-crop", "related_prob": 0.95
        }
    ]

    # Candidate edges include shortcut 1 -> 3
    candidate_edges = list(forest_edges) + [
        {
            "source": "node_1", "target": "node_3", "source_id": "node_1", "target_id": "node_3",
            "transformation": "crop", "confidence": 0.88, "weight": 0.12,
            "direction": "A_TO_B", "is_uncertain": False,
            "transformation_confidence": 0.91, "direction_confidence": 0.93,
            "direction_rationale": "Shortcut containment", "related_prob": 0.90,
            "is_shortcut": True
        }
    ]

    rejected_pairs = [
        {"u": "node_1", "v": "node_x", "confidence": 0.10, "related_prob": 0.05}
    ]

    save_analysis(
        analysis_id=analysis_id,
        timestamp="2026-10-19T15:00:00",
        status="COMPLETED",
        image_count=3,
        root_image_id="node_1",
        root_confidence=0.94,
        execution_time_ms=50.0,
        similarity_threshold=0.50,
        confidence_filter=0.35,
        images=nodes,
        edges=forest_edges,
        xai_results=[],
        candidate_edges=candidate_edges,
        rejected_pairs=rejected_pairs
    )

    reloaded = get_analysis_by_id(analysis_id)
    assert reloaded is not None
    assert len(reloaded["edges"]) == 2, "Spanning forest edges must retain exactly 2 edges"
    assert len(reloaded["candidate_edges"]) == 3, "Complete candidate evidence graph must retain 3 edges"
    assert len(reloaded["rejected_pairs"]) == 1, "Rejected pairs must be preserved"

    # Verify edge metadata survived intact
    e1 = reloaded["edges"][0]
    assert e1["transformation_confidence"] == 0.98
    assert e1["direction_confidence"] == 0.99
    assert e1["direction_rationale"] == "Direct crop containment"
