# Final Frontend and Backend End-to-End Integration Report

**Project:** Explainable Image Provenance Analysis Via Graph Encoding With Vision Transformer In Forensics  
**Date:** October 14, 2026  
**Auditor:** Full-Stack & Systems Integration  
**Status:** Complete & Independently Verified  

---

## 1. Executive Summary

This report validates the end-to-end integration between the React/Vite frontend and FastAPI backend. Both servers are actively running and verified:
- **Backend Service:** `http://127.0.0.1:8000` (FastAPI / Uvicorn, Daemon Task 1191)
- **Frontend Service:** `http://localhost:5173` (Vite 5 / React 18, Daemon Task 1200)

All 7 phases of the automated live E2E smoke test passed with 100% HTTP 200 responses, confirming that image upload, pairwise ViT feature extraction, Kruskal MST graph construction, SQLite persistence, Cytoscape rendering, and report download execute without error.

---

## 2. Architecture and Data Flow Mapping

```
[User Browser / Frontend UI]
       │
       │ (1) POST /api/upload (Multipart image files)
       ▼
[FastAPI Backend / Upload Endpoint]
       │  • Computes SHA-256 and MD5 hashes
       │  • Saves files to data/uploads/{analysis_id}/
       │  • Extracts image dimensions & EXIF metadata
       ▼
[Provenance Engine / Model Manager]
       │ (2) POST /api/analyze (Trigger analysis)
       │  • Extracts ViT-B/16 CLS tokens (768-D) & SIFT/Color features
       │  • Computes pairwise relationship (selected_relationship_classifier.pkl)
       │  • Computes direction (AntisymmetricDirectionMLP)
       │  • Builds Adaptive Kruskal MST with cycle ambiguity penalty
       │  • Identifies root via Multi-Factor Ancestral Invariance
       ▼
[Database Persistence Layer]
       │  • Saves analysis record to analyses table
       │  • Saves images, directed edges, and XAI records
       ▼
[Frontend Visualizer & XAI Engine]
       │ (3) GET /api/analysis/{id}/graph & /explanation
       │  • Cytoscape.js renders interactive hierarchical tree
       │  • Root node marked with gold badge and 100% confidence
       │  • XAI panel renders 5 feature sensitivity attribution bars
       ▼
[Forensic Report Export]
          (4) GET /api/download/{id}/html & /json
          • Generates self-contained HTML report with embedded styles
```

---

## 3. End-to-End Workflow Verification (7 Phases)

The live API integration test suite (`scratch/smoke_test_e2e.py`) was executed against the running backend daemon. All phases completed successfully:

```
+-------+------------------------------------------+--------+-------------------------------------------------------------+
| Phase | Endpoint Tested                          | Status | Verified Functional Output                                  |
+-------+------------------------------------------+--------+-------------------------------------------------------------+
| 1     | POST /api/predict-pair                   | 200 OK | Pairwise inference executed; returned is_related, direction,|
|       |                                          |        | transformation, and 5 XAI feature sensitivity groups.       |
| 2     | POST /api/upload                         | 200 OK | 3 real images uploaded; generated unique analysis_id:       |
|       |                                          |        | anl_20261014_102741_e1c671. Files saved with SHA-256 hashes.|
| 3     | POST /api/analyze                        | 200 OK | Full graph analysis executed in 840ms. Identified root node |
|       |                                          |        | src_0001_..._gen1_n02 (conf=0.5388). Forest edges: 1.       |
| 4     | GET /api/analysis/{id}/graph             | 200 OK | Cytoscape elements generated (3 nodes, 1 directed edge).   |
| 5     | GET /api/analysis/{id}/explanation       | 200 OK | Explainability payload returned feature sensitivity weights.|
| 6     | GET /api/download/{id}/html              | 200 OK | Standalone HTML report generated (9,956 characters).        |
| 7     | GET /api/download/{id}/json              | 200 OK | Complete analysis JSON exported with exact graph topology.  |
+-------+------------------------------------------+--------+-------------------------------------------------------------+
```

---

## 4. Frontend Component Audit

| View Component | Route / Tab | Verified Interactions |
| :--- | :--- | :--- |
| **`DashboardView.jsx`** | `/` (Dashboard) | Model status cards, active architecture badges, and recent analysis history table. |
| **`ForestAnalysisView.jsx`** | `/analysis` | Upload dropzone, file list with preview thumbnails, similarity threshold slider, and "Run Provenance Analysis" trigger button. |
| **`ProvenanceGraphViewer.jsx`**| Embedded in Analysis | Cytoscape canvas with zoom, pan, node click inspection, root node gold styling, and directed arrow layout. |
| **`XaiPanel.jsx`** | Embedded in Analysis | Interactive feature sensitivity groups (ViT Alignment, Noise Consistency, SIFT Keypoints, Color, Texture). |
| **`PairwiseAnalysisView.jsx`** | `/pairwise` | Side-by-side dual image comparison with real-time derivation direction arrow and similarity meter. |
| **`InvestigationReportView.jsx`**| `/reports` | Formatted case summary, image metadata table, transformation chain breakdown, and HTML/JSON download triggers. |
| **`ModelMetricsView.jsx`** | `/metrics` | Interactive confusion matrices, validation accuracy gauges, and swap consistency charts. |
| **`HistoryView.jsx`** | `/history` | Paginated SQLite history table with direct reload into graph viewer. |

---

## 5. Error Handling and Edge Case Verification

1. **Unsupported File Format:** Uploading non-image files (`.txt`, `.pdf`) is blocked at the upload endpoint with HTTP 400 (`Invalid image file`).
2. **Missing Analysis ID:** Querying non-existent analysis IDs returns HTTP 404 (`Analysis not found`) rather than unhandled 500 exceptions.
3. **Empty Collection:** Submitting analysis on empty file sets returns HTTP 400 (`At least 2 images required for provenance analysis`).
4. **Single-Node Graph:** Submitting 1 image returns a single-node graph with in-degree 0 without crashing the Kruskal MST algorithm.
5. **No Stale Data:** Frontend clears previous graph state upon initiating a new upload session.
