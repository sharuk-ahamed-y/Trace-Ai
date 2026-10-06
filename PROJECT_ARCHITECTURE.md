# PROJECT ARCHITECTURE: PROVEX AI (Image Provenance Intelligence Platform)

This document provides a comprehensive technical reference for the PROVEX AI web application, detailing the existing architecture, the upgraded full-stack AI system, data flow, model flow, API routes, database schema, and Explainable AI (XAI) pipeline.

---

## 1. High-Level Architecture Overview

PROVEX AI combines Vision Transformer (ViT-B/16) visual feature extraction, statistical & spectral forensic analysis, pairwise machine learning classification, Minimum Spanning Tree (MST) graph reconstruction, and XAI attribution maps into a single full-stack web platform.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             PROVEX AI FRONTEND                              │
│         (React 18 + Vite + Tailwind CSS + Cytoscape.js + Lucide)            │
│                     Theme: Cyber-Forensics (Black/Green)                    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTP REST APIs / JSON / Multipart
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             PROVEX AI BACKEND                               │
│                         (Python 3.13 + FastAPI)                             │
│  ┌───────────────────────┐  ┌─────────────────────┐  ┌───────────────────┐  │
│  │   ModelManager        │  │  ProvenanceEngine   │  │   XAIEngine       │  │
│  │  (ViT-B/16 + MLPs)    │  │ (Pairwise + MST)    │  │ (Heatmaps + Evidence)│
│  └───────────────────────┘  └─────────────────────┘  └───────────────────┘  │
└──────────────┬───────────────────────┬──────────────────────┬───────────────┘
               │                       │                      │
               ▼                       ▼                      ▼
┌──────────────────────────┐┌──────────────────────┐┌────────────────────────┐
│     SQLite Database      ││    Local Storage     ││   Dataset Benchmark    │
│    (provenance_app.db)   ││(data/uploads, reports││  (100-Original SIPGD)  │
└──────────────────────────┘└──────────────────────┘└────────────────────────┘
```

---

## 2. End-to-End Data & Execution Flow

1. **User Input / Upload**:
   - The user drags and drops one or more image files (PNG, JPG, WEBP, TIFF) into the frontend Upload Zone.
   - Files are validated client-side (file size, extension) and transmitted via `POST /api/upload`.
2. **Local Storage & Session Setup**:
   - The backend validates file signatures, computes MD5 & SHA-256 cryptographic hashes, extracts basic image dimensions, and stores raw files in `data/uploads/{analysis_id}/`.
   - An `analysis_id` (e.g. `ANL-2026-04821`) is generated and registered in SQLite database `provenance_app.db`.
3. **ML Inference Pipeline**:
   - **Feature Extraction**: Pretrained ViT-B/16 extracts 768-dim L2-normalized CLS token embeddings for each image.
   - **Forensic Extraction**: `src/forensic_features.py` extracts 15 numerical statistics (width, height, aspect ratio, file size, JPEG quality estimates, RGB means/stds, Laplacian edge intensity, spatial noise residual std/skewness).
   - **Pairwise Prediction**: `ModelManager` computes pairwise features $\mathbf{x}_{AB} = [\mathbf{e}_A \,\|\, \mathbf{e}_B \,\|\, |\mathbf{e}_A - \mathbf{e}_B| \,\|\, \text{cos\_sim}(\mathbf{e}_A, \mathbf{e}_B)]$ and feeds them to the 2-branch MLP classifiers.
   - **Relationship & Transformation Classification**: Predicts `related` (0/1), `direction` (`A_TO_B`, `B_TO_A`), and `transformation` (12 types).
4. **Graph Reconstruction & Root Identification**:
   - Builds a weighted candidate graph where edge weights are cosine distances $d(A, B) = 1 - \text{cos\_sim}(\mathbf{e}_A, \mathbf{e}_B)$, filtered by edge confidence $\ge 0.50$.
   - NetworkX Minimum Spanning Tree (MST) reconstructs the acyclic directed tree topology.
   - Root estimation identifies the node with in-degree 0 and maximum descendant reach as `Probable Root`.
5. **XAI Explanation Generation**:
   - `XAIEngine` generates spatial noise residual difference heatmaps and builds factual evidence statements.
6. **Report Generation & Storage**:
   - HTML and downloadable PDF reports are generated and stored in `data/reports/`.
   - Results, predictions, and graph edge records are persisted to SQLite.

---

## 3. API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/upload` | Upload multiple images for analysis |
| `POST` | `/api/analyze` | Trigger real ML provenance analysis pipeline |
| `GET` | `/api/analysis/{analysis_id}` | Fetch structured analysis results |
| `GET` | `/api/analysis/{analysis_id}/graph` | Fetch Cytoscape.js interactive graph JSON |
| `GET` | `/api/analysis/{analysis_id}/explanation` | Fetch XAI attribution maps & evidence text |
| `GET` | `/api/history` | Query SQLite analysis history |
| `GET` | `/api/dataset` | Browse 100-Original dataset metadata |
| `GET` | `/api/model-info` | Fetch ViT and MLP model specifications |
| `GET` | `/api/health` | Live system health (CPU, RAM, GPU, DB status) |
| `GET` | `/api/download/{analysis_id}` | Download generated analysis report |
| `POST` | `/api/demo` | Execute pre-existing benchmark dataset demo |

---

## 4. Database Schema (SQLite: `provenance_app.db`)

- **`analyses`**: `analysis_id` (PRIMARY KEY), `timestamp`, `num_images`, `status`, `probable_root`, `root_confidence`, `overall_confidence`
- **`images`**: `id`, `analysis_id`, `filename`, `filepath`, `width`, `height`, `file_size`, `sha256`, `role`
- **`predictions`**: `id`, `analysis_id`, `image_a`, `image_b`, `related`, `direction`, `transformation`, `confidence`
- **`graph_edges`**: `id`, `analysis_id`, `source`, `target`, `transformation`, `weight`, `confidence`
- **`xai_results`**: `id`, `analysis_id`, `image_a`, `image_b`, `explanation_text`, `heatmap_path`

---

## 5. Explainable AI (XAI) Pipeline

- **Visual Attribution**: Spatial Noise Residual Difference Heatmaps obtained by computing 3x3 local median high-pass filter differences between Parent and Child images.
- **Empirical Evidence**: Textual rationale generated strictly from model output logits, cosine similarity, Laplacian gradient drops, and JPEG quality drop scores. No fabricated numbers.
