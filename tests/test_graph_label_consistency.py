"""
Focused Regression Test: Graph Edge Label Consistency Across API Paths
========================================================================
Verifies that:
1. GET /api/analysis/{id} and GET /api/analysis/{id}/graph produce consistent,
   identically formatted labels for the same edge.
2. Both metrics are preserved separately:
   - Relationship / Pairwise Match Confidence = `confidence`
   - Transformation Classification Confidence = `transformation_confidence`
3. Neither metric is silently substituted for the other:
   - Relationship confidence is explicitly named ('Match: XX%')
   - Transformation classification confidence is explicitly named ('Tfm: XX%')
4. Photometric / uncertain edges retain `is_uncertain=True` and append
   '[Lineage: Parent → Derivative]' while directed edges do not.
5. Candidate edges consistently include explicit match confidence.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.api.endpoints import format_edge_label, format_candidate_edge_label


@pytest.fixture
def client():
    return TestClient(app)


def test_format_edge_label_unit():
    """Unit test for format_edge_label helper with various confidence states."""
    # Directed edge with both confidences
    lbl_crop = format_edge_label("crop", 0.997, 0.9985, is_uncertain=False)
    assert lbl_crop == "CROP (Match: 100%, Tfm: 100%)"
    assert "Lineage:" not in lbl_crop

    # Uncertain edge with divergent confidences (e.g. Contrast: Match 83%, Tfm 96%)
    lbl_contrast = format_edge_label("contrast", 0.8333, 0.96, is_uncertain=True)
    assert lbl_contrast == "CONTRAST (Match: 83%, Tfm: 96%) [Lineage: Parent → Derivative]"
    assert "Match: 83%" in lbl_contrast
    assert "Tfm: 96%" in lbl_contrast

    # Edge with missing transformation confidence
    lbl_unresolved = format_edge_label("unresolved", 0.80, None, is_uncertain=True)
    assert lbl_unresolved == "UNRESOLVED (Match: 80%, Tfm: Unavailable) [Lineage: Parent → Derivative]"

    # Edge with missing match confidence
    lbl_no_match = format_edge_label("blur", None, 0.95, is_uncertain=False)
    assert lbl_no_match == "BLUR (Match: Unavailable, Tfm: 95%)"


def test_format_candidate_edge_label_unit():
    """Unit test for format_candidate_edge_label helper."""
    cand_lbl = format_candidate_edge_label("blur", 0.784)
    assert cand_lbl == "[CANDIDATE] BLUR (Match: 78%)"

    cand_lbl_none = format_candidate_edge_label("related", None)
    assert cand_lbl_none == "[CANDIDATE] RELATED"


def test_graph_label_consistency_across_api_paths(client):
    """End-to-end API consistency test against saved analysis."""
    analysis_id = "anl_20261019_201200_039ee6"

    # 1. Fetch analysis dossier via GET /api/analysis/{id}
    res_dossier = client.get(f"/api/analysis/{analysis_id}")
    assert res_dossier.status_code == 200, f"Analysis {analysis_id} not found"
    dossier_data = res_dossier.json()
    edges_dossier = dossier_data["edges"]
    assert len(edges_dossier) > 0

    # 2. Fetch Cytoscape graph via GET /api/analysis/{id}/graph
    res_graph = client.get(f"/api/analysis/{analysis_id}/graph")
    assert res_graph.status_code == 200
    graph_data = res_graph.json()
    cy_elements = graph_data["cytoscape_elements"]
    cy_edges = cy_elements["edges"]
    assert len(cy_edges) == len(edges_dossier)

    # 3. Build lookup map for cy_edges by edge ID
    cy_edge_map = {e["data"]["id"]: e["data"] for e in cy_edges}

    # 4. Verify 1:1 match and exact label consistency for each edge
    for edge in edges_dossier:
        edge_id = edge["id"]
        assert edge_id in cy_edge_map, f"Edge {edge_id} missing from /graph endpoint"
        cy_edge = cy_edge_map[edge_id]

        # Exact label consistency
        assert edge["label"] == cy_edge["label"], (
            f"Label mismatch for edge {edge_id}: "
            f"'{edge['label']}' (dossier) vs '{cy_edge['label']}' (graph)"
        )

        # Both metrics preserved separately
        assert edge["confidence"] == cy_edge["confidence"]
        assert edge["transformation_confidence"] == cy_edge["transformation_confidence"]

        # Explicit metric naming in label
        rel_pct_str = f"{edge['confidence'] * 100:.0f}%"
        assert f"Match: {rel_pct_str}" in cy_edge["label"], (
            f"Label '{cy_edge['label']}' does not explicitly name Match confidence {rel_pct_str}"
        )

        if edge["transformation_confidence"] is not None:
            tfm_pct_str = f"{edge['transformation_confidence'] * 100:.0f}%"
            assert f"Tfm: {tfm_pct_str}" in cy_edge["label"], (
                f"Label '{cy_edge['label']}' does not explicitly name Tfm confidence {tfm_pct_str}"
            )

        # Lineage and uncertainty consistency
        if edge.get("is_uncertain"):
            assert "[Lineage: Parent → Derivative]" in cy_edge["label"]
            assert cy_edge["is_uncertain"] is True
            assert cy_edge["pairwise_direction"] == "UNCERTAIN"
        else:
            assert "[Lineage: Parent → Derivative]" not in cy_edge["label"]
            assert cy_edge["is_uncertain"] is False


def test_contrast_edge_metric_distinction(client):
    """Specifically verifies that Contrast edge preserves Match 83% vs Tfm 96% explicitly."""
    analysis_id = "anl_20261019_201200_039ee6"
    res = client.get(f"/api/analysis/{analysis_id}/graph")
    assert res.status_code == 200
    cy_edges = res.json()["cytoscape_elements"]["edges"]

    contrast_edge = None
    for e in cy_edges:
        if e["data"]["transformation"] == "contrast":
            contrast_edge = e["data"]
            break

    assert contrast_edge is not None, "Contrast edge not found in graph"
    assert contrast_edge["confidence"] == pytest.approx(0.8333, abs=0.01)
    assert contrast_edge["transformation_confidence"] == pytest.approx(0.96, abs=0.01)

    # Ensure label explicitly shows BOTH without conflation
    label = contrast_edge["label"]
    assert "Match: 83%" in label
    assert "Tfm: 96%" in label
    assert "[Lineage: Parent → Derivative]" in label
