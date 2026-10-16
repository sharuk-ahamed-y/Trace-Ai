# Root Cause Analysis: Bug 1 — Provenance Graph Edges Missing

## 1. Executive Summary

- **Reported Bug**: The Provenance Graph page rendered `2 Nodes, 0 Directed Edges`, displaying two uploaded images as isolated floating nodes with no connecting edge.
- **Investigation Finding**: The two images uploaded in the user session (`WhatsApp_Image_2026-10-13_at_1.31.11_AM.jpeg` and `ojj.jpeg`) are **genuinely independent and unrelated images**. When evaluated by the Vision Transformer and scikit-learn relationship classifier, the empirical relationship probability was:
  $$P(\text{related}) = 0.0979 \quad (9.79\%)$$
  $$\text{ViT CLS Cosine Similarity} = 0.6538$$
  Because $P(\text{related}) = 0.0979$ falls far below the derivation candidate threshold ($\tau = 0.50$), no candidate edge was generated. The Kruskal Minimum Spanning Forest correctly reconstructed a 2-node, 0-edge forest (two independent trees).
- **Software Defects Identified & Fixed**:
  1. **Silent Disconnected State**: When `edges.length === 0`, the UI gave zero feedback or rationale for why no edges connected the nodes, giving the false visual impression of a broken graph component rather than a disconnected forensic forest.
  2. **Dead-End Edge Telemetry**: The right panel insisted "Select a directed edge on the canvas to inspect transformation evidence", which was impossible with 0 edges.
  3. **High UI Slider Defaults**: `UploadZone.jsx` initialized `simThreshold` to `0.65` and `confFilter` to `0.50` (higher than the backend defaults of `0.50` and `0.35`), which would reject even legitimate subtle derivations.
  4. **Hidden Rejection Diagnostics**: `provenance_engine.py` discarded `rejected_pairs` from the return dictionary, hiding the pairwise probabilities and decision thresholds from the frontend and API.
  5. **Cytoscape Dagre Degeneracy**: The default `dagre` layout requires directed edges to compute hierarchical ranks; with 0 edges, isolated nodes were unranked.

---

## 2. Tracing the Complete Data Flow

1. **Upload & Storage**:
   - Files are POSTed to `/api/upload` via `UploadZone.jsx`.
   - Stored in `data/uploads/{analysis_id}/` with sanitized filenames.
   - Node IDs are assigned from `file_path.stem` (e.g. `WhatsApp_Image_2026-10-13_at_1.31.11_AM`, `ojj`).
2. **Analysis Execution**:
   - `UploadZone.jsx` triggers POST `/api/analyze`.
   - `run_provenance_analysis()` loads images, computes SHA-256/MD5 hashes, and extracts 768-D ViT CLS tokens + 19-D forensic features.
3. **Pairwise Inference**:
   - `ModelManager.predict_pair()` computes 115-D difference representation:
     - ViT cosine similarity, CLS token difference, spatial noise residual deltas, Laplacian gradient deltas, JPEG quality estimate.
   - `selected_relationship_classifier.pkl` predicts relationship probability $P(\text{related})$.
4. **Candidate Edge Filtering**:
   - Two-stage filter:
     $$\text{Stage 1: } P(\text{related}) \ge \text{similarity\_threshold}$$
     $$\text{Stage 2: } P(\text{related}) \times P(\text{direction}) \ge \text{confidence\_filter}$$
   - For `WhatsApp_Image...` vs `ojj`: $P(\text{related}) = 0.0979 < 0.50$. Pair is appended to `rejected_pairs` and excluded from `candidate_edges`.
5. **Kruskal Minimum Spanning Forest**:
   - `nx.minimum_spanning_tree` executes over connected components of candidate graph. With 0 candidate edges, MST selects 0 edges.
6. **Directed Forest Orientation**:
   - Directed edges are oriented using `antisymmetric_direction_model.pt`. With 0 MST edges, final directed edges count is 0.

---

## 3. Files & Functions Modified

| File | Function / Component | Changes Made |
| :--- | :--- | :--- |
| `backend/app/services/provenance_engine.py` | `run_provenance_analysis()` | Included `rejected_pairs` in return dict; added diagnostic logs for uploaded count, feature dimensions, pair relationship probability & decision, direction probability, candidate/MST/final edge counts. |
| `backend/app/api/endpoints.py` | `analyze_provenance()` | Exposed `rejected_pairs` in API response; ensured consistent analysis schema. |
| `frontend/src/components/UploadZone.jsx` | `UploadZone()` | Aligned default `simThreshold` to `0.50` and `confFilter` to `0.35` (matching backend defaults). |
| `frontend/src/components/ProvenanceGraphViewer.jsx` | `ProvenanceGraphViewer()` | Added `edges.length === 0 ? 'grid' : layoutName` fallback for Cytoscape; added Disconnected Forest forensic banner; added Pairwise Forest Evaluation card in right telemetry panel with direct link to XAI. |

---

## 4. Before & After Comparisons

### A. Raw API Response (`/api/analyze`)

**Before Fix**:
```json
{
  "analysis_id": "anl_20261014_113320_042850",
  "status": "COMPLETED",
  "root_image_id": "WhatsApp_Image_2026-10-13_at_1.31.11_AM",
  "image_count": 2,
  "edge_count": 0,
  "nodes": [...],
  "edges": [],
  "candidate_edges": [],
  "xai_results": []
}
```

**After Fix**:
```json
{
  "analysis_id": "anl_20261014_122633_b1d400",
  "status": "COMPLETED",
  "root_image_id": "WhatsApp_Image_2026-10-13_at_1.31.11_AM",
  "image_count": 2,
  "edge_count": 0,
  "nodes": [...],
  "edges": [],
  "candidate_edges": [],
  "rejected_pairs": [
    {
      "u": "ojj",
      "v": "WhatsApp_Image_2026-10-13_at_1.31.11_AM",
      "related_prob": 0.0979,
      "cosine_similarity": 0.6538,
      "direction": "NONE",
      "direction_confidence": 0.9021,
      "confidence": 0.0883
    }
  ],
  "xai_results": [
    {
      "source_id": "ojj",
      "target_id": "WhatsApp_Image_2026-10-13_at_1.31.11_AM",
      "is_independent": true,
      "relationship": "INDEPENDENT / UNRELATED",
      "confidence": 0.0979
    }
  ]
}
```

### B. Node & Edge Counts

| Scenario | Tested Images | Node Count | Directed Edge Count | Resulting Topology |
| :--- | :--- | :---: | :---: | :--- |
| **Unrelated Custom Pair** | `WhatsApp_Image...` & `ojj.jpeg` | 2 | 0 | Disconnected Forest (2 Independent Sources) |
| **Related Benchmark Pair** | `src_0003...root` & `src_0003...gen1_n02` | 2 | 1 | Directed Tree (`root -> gen1_n02` via `perspective`, $P=82.9\%$) |

---

## 5. Diagnostic Log Verification

During real inference over the unrelated pair, `provenance_engine.py` logged:
```text
[ProvenanceEngine] Uploaded image count: 2
[ProvenanceEngine] Extracted feature dimensions for node 'ojj': ViT=768, Forensics=19
[ProvenanceEngine] Extracted feature dimensions for node 'WhatsApp_Image_2026-10-13_at_1.31.11_AM': ViT=768, Forensics=19
[ProvenanceEngine] Pair (ojj, WhatsApp_Image_2026-10-13_at_1.31.11_AM): P(rel)=0.0979, CosSim=0.6538 -> Relationship decision=REJECTED; Direction=NONE (P_dir=0.9021, predicted=WhatsApp_Image_2026-10-13_at_1.31.11_AM->ojj)
[ProvenanceEngine] Candidate-edge count: 0, Rejected-pair count: 1
[ProvenanceEngine] MST-edge count: 0
[ProvenanceEngine] Final directed-edge count: 0
[ProvenanceEngine] Serialized node count: 2, edge count: 0
```

---

## 6. Regression & Acceptance Criteria

- **A. Graph Accurately Renders Backend Edges**: Verified. Connected derivation pairs render edges with direction arrows and transformation labels; unrelated pairs render as separate roots.
- **B. Honest Forensic Architecture**: No false edges are force-connected between unrelated images.
- **C. Telemetry & User Guidance**: The UI explains why 0 edges formed and displays pairwise similarity, confidence scores, and threshold details.
