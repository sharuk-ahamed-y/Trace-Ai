# Explainable Image Provenance Analysis Via Graph Encoding With Vision Transformer In Forensics

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org)
[![React 18](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-6.4-646CFF.svg)](https://vitejs.dev)
[![License: Academic Research](https://img.shields.io/badge/License-Academic_Research-green.svg)](#)

> **Official Research Benchmark & Application Suite**  
> **Authors:** Mrs. R. Rathika, Dhaya Anandh D., Sharuk Ahamed Y., Mohan Raj R.  
> **Affiliation:** Department of Artificial Intelligence and Data Science, Dr. Mahalingam College of Engineering and Technology, Pollachi, Tamil Nadu, India.

---

## 1. System Overview

**TRACEAI** is a forensic intelligence platform designed to reconstruct multi-generation parent-child image derivation graphs (Directed Acyclic Graphs, DAGs) from collections of suspect digital images. 

Modern social media platforms, messaging applications, and image-editing suites routinely re-encode, compress, rescale, crop, and filter digital assets. Determining the true original root asset and establishing chronological edit chains is a foundational challenge in digital media forensics and court evidentiary validation.

This platform integrates:
1. **Vision Transformer (ViT-B/16)** deep semantic CLS embeddings (768-D).
2. **High-Frequency Noise Residual & Spectral Spatial Forensics** (17-D) capturing Laplacian variance, JPEG DCT quantization drop, edge sharpness, and color spatial distributions.
3. **115-Dimensional Pairwise Feature Fusion Representation** trained via scikit-learn Pipelines (`MLPClassifier` + `StandardScaler`) and PyTorch multi-task networks.
4. **Component-Aware Minimum Spanning Forest (MSF)** topology using Kruskal's algorithm with topological cycle reversal safeguards and edge distance metric $d = 1 - P(\text{related})$.
5. **Empirical Explainable AI (XAI)** generating spatial noise difference heatmaps (JET colormap overlay), MSE/PSNR/SSIM metrics, and plain-language court-admissible forensic rationale.

---

## 2. Mathematical & Algorithmic Architecture

### 2.1 115-Dimensional Feature Fusion Representation

For any suspect image pair $(A, B)$ with 57-dimensional concatenated forensic embeddings $\mathbf{v}_A, \mathbf{v}_B$:
$$\mathbf{f}_{\text{pair}}(A, B) = \big[ |\mathbf{v}_A - \mathbf{v}_B|_{57}, \; (\mathbf{v}_A \odot \mathbf{v}_B)_{57}, \; \text{CosineSim}(\mathbf{v}_A, \mathbf{v}_B)_{1} \big] \in \mathbb{R}^{115}$$

- **57 Absolute Difference Features ($|\mathbf{v}_A - \mathbf{v}_B|$):** Directly quantifies magnitude deltas in noise variance, edge sharpness, and feature dimensions.
- **57 Element-wise Product Features ($\mathbf{v}_A \odot \mathbf{v}_B$):** Encodes feature cross-correlations and co-occurrence patterns.
- **1 Cosine Similarity Metric:** Measures global semantic orientation alignment in latent space.

### 2.2 Component-Aware Minimum Spanning Forest

Unlike standard Minimum Spanning Trees (MST) that force false bridge edges between disconnected clusters, TRACEAI preserves **Component Independence**:
- For candidate edge $(u, v)$ with predicted relatedness probability $P(\text{related}(u, v)) \ge \tau$:
  $$d(u, v) = 1.0 - P(\text{related}(u, v))$$
- Kruskal's algorithm extracts the minimum distance forest using Disjoint-Set Union-Find.
- **Cycle Elimination:** Cycles are prevented structurally during spanning forest construction; directed edges are oriented based on degradation markers, with cycle reversals automatically triggered if directed loops emerge.
- **Root Identification:** The original root image is identified by evaluating incoming ancestral edge weights—the master root node exhibits minimal incoming parent weight and pristine noise variance.

---

## 3. UI Design & Aesthetics

The web application follows two reference visual directions:
1. **Pulse Dark Neon Dashboard (Reference 1):**
   - Palette: Deep slate canvas (`#070909`, `#111916`, `#1A2822`), neon-lime green accents (`#39FF88`, `#D7FF32`), and cyan (`#35D9FF`).
   - Features: Activity heatmap matrix (6 weeks), circular SVG radial gauge, recent analysis telemetry, real-time hardware compute monitors.
2. **Forensic DAG Canvas (Reference 2):**
   - Palette: Technical grid background (`.forensic-grid`), glowing Cytoscape DAG nodes, animated directional arrows, waveform spectral analysis widget, zoom/pan telemetry inspector, and transformation badges.

---

## 4. Platform Pages & Capabilities

| Page # | Route | Module | Purpose |
|---|---|---|---|
| **1** | `/` (Dashboard) | `DashboardView.jsx` | Overview of system metrics, 6-week activity heatmap, model distributions, and quick demo triggers. |
| **2** | `upload` | `UploadZone.jsx` | Multi-image drag-and-drop, thumbnail cards, parameter sliders ($\tau$, confidence filter), and real-time execution progress. |
| **3** | `pairwise` | `PairwiseAnalysisView.jsx` | Isolated pair comparison, $P(\text{related})$ probability, directional hypothesis ($A \to B$ vs $B \to A$), and transformation class. |
| **4** | `graph` | `ProvenanceGraphViewer.jsx` | Interactive Cytoscape DAG canvas, layout switcher (Dagre, Breadthfirst, Concentric), node/edge inspector, and waveform telemetry. |
| **5** | `forest` | `ForestAnalysisView.jsx` | Dense Candidate Graph vs. Kruskal Spanning Forest comparison, connected components table, and cycle resolution breakdown. |
| **6** | `xai` | `XaiPanel.jsx` | Spatial noise residual difference heatmap (JET colormap), MSE/PSNR/SSIM metrics, and 115-D feature sensitivity bar charts. |
| **7** | `report` | `InvestigationReportView.jsx` | Formal forensic dossier with Case ID, SHA-256 integrity hashes, derivation timeline table, print triggering, and HTML report download. |
| **8** | `dataset` | `DatasetExplorer.jsx` | Browser for the SIPGD dataset (100 ground-truth DAGs and 795 images) with one-click inference execution. |

---

## 5. Machine Learning Models & Colab Integration

TRACEAI automatically searches for trained model checkpoints across multiple directories.

### Supported Model Checkpoints:
1. **Scikit-learn Pipeline Models (`.pkl`):**
   - Expected filenames: `selected_relationship_classifier.pkl` or `relationship_classifier.pkl`
   - Architecture: `StandardScaler` + `MLPClassifier` trained on 115-D feature vectors.
2. **PyTorch Multi-task Provenance Network (`provenance_exp3_model.pt`):**
   - Multitask network predicting relatedness, direction, and 12-class transformation simultaneously (2350-D concatenated embeddings).
3. **PyTorch Research Classifier (`research_provenance_model.pt`):**
   - 289-D feature classifier trained across 12 image transformations.

### Configuring Colab-Trained Model Paths:
You can specify the path to your Colab-trained model file in two ways:

#### Option A: Via Environment Variable (`.env`)
Create or edit `.env` in the project root:
```env
TRAINED_MODELS_PATH=C:/Users/devad/Downloads/my_colab_models/selected_relationship_classifier.pkl
COLAB_MODELS_PATH=C:/Users/devad/Downloads/my_colab_models
```

#### Option B: Via `backend/app/config.py`
Edit the search path in `backend/app/config.py`:
```python
COLAB_MODELS_PATH = Path(r"C:\Users\devad\Downloads\my_colab_models")
```
When placed in `models/` or the configured directory, the Model Manager automatically detects the model on startup and logs:
```text
[ModelManager] Loaded scikit-learn relationship pipeline from: models/selected_relationship_classifier.pkl
```

---

## 6. Project Directory Structure

```text
image-provenance-using-graph-main/
|-- backend/
|   |-- app/
|   |   |-- api/
|   |   |   `-- endpoints.py               # Comprehensive REST endpoints
|   |   |-- database/
|   |   |   `-- db.py                      # SQLite database persistence layer
|   |   |-- services/
|   |   |   |-- model_manager.py           # Multi-model manager (ViT + sklearn 115D + PyTorch)
|   |   |   |-- provenance_engine.py       # Kruskal Minimum Spanning Forest & BFS engine
|   |   |   |-- xai_engine.py              # Empirical noise residual heatmaps & metrics
|   |   |   |-- report_generator.py        # Formal HTML dossier generator
|   |   |   `-- research_provenance_service.py # Research model bridge
|   |   |-- config.py                      # Configurable paths & hyperparameters
|   |   `-- main.py                        # FastAPI application entrypoint
|   `-- backup/                            # Backups of original files prior to reconstruction
|-- frontend/
|   |-- src/
|   |   |-- components/
|   |   |   |-- Sidebar.jsx                # Pulse vertical sidebar navigation
|   |   |   |-- DashboardView.jsx          # Page 1: Pulse Intelligence Dashboard
|   |   |   |-- UploadZone.jsx             # Page 2: Multi-image ingestion & reconstruction
|   |   |   |-- PairwiseAnalysisView.jsx   # Page 3: Pairwise relationship & transformation
|   |   |   |-- ProvenanceGraphViewer.jsx  # Page 4: Interactive Cytoscape DAG Canvas
|   |   |   |-- ForestAnalysisView.jsx     # Page 5: Candidate Graph vs. MST Forest
|   |   |   |-- XaiPanel.jsx               # Page 6: Explainable AI Forensic Evidence
|   |   |   |-- InvestigationReportView.jsx # Page 7: Investigation Dossier & Export
|   |   |   `-- DatasetExplorer.jsx        # Page 8: Benchmark Dataset Catalog
|   |   |-- App.jsx                        # Master application container
|   |   |-- index.css                      # Google Fonts, forensic grid, scrollbars
|   |   `-- main.jsx                       # React entrypoint
|   |-- tailwind.config.js                 # Pulse and Cyber color themes
|   `-- package.json
|-- image_provenance_dataset/              # SIPGD Research Benchmark
|   |-- graphs/                            # 100 Ground-truth DAG JSON specifications
|   |-- images/                            # 795 Benchmark images across test/train/val
|   `-- forensic_features.csv              # Ground-truth forensic metadata
|-- models/                                # Model storage directory
|   |-- provenance_exp3_model.pt           # PyTorch multi-task weights
|   `-- research_provenance_model.pt       # PyTorch research weights
|-- src/
|   |-- feature_extractor.py               # Pretrained ViT-B/16 CLS embedder
|   |-- pair_features.py                   # 115-D feature vector generator
|   `-- forensic_features.py               # Laplacian & DCT forensic extractor
|-- tests/
|   |-- test_pipeline.py                   # Integration test suite
|   `-- test_e2e_full.py                   # Exhaustive 10-module end-to-end test suite
|-- provenance_app.db                      # SQLite database
|-- requirements.txt                       # Python dependencies
`-- README.md                              # Project documentation
```

---

## 7. Installation & Quick Start

### Prerequisites
- Python 3.10 or higher
- Node.js 18 or higher & npm

### Step 1: Clone and Set Up Virtual Environment

```bash
git clone https://github.com/YOUR_USERNAME/image-provenance-using-graph.git
cd image-provenance-using-graph-main

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Start the FastAPI Backend Server

```bash
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
The backend initializes the Vision Transformer ViT-B/16 backbone, loads classifier models, creates SQLite tables in `provenance_app.db`, and exposes interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

### Step 3: Start the React Frontend

Open a second terminal window:
```bash
cd frontend
npm install
npm run dev
```
Access the application in your browser at `http://localhost:5173`.

---

## 8. Running Automated Tests

Run the comprehensive end-to-end test suite:
```bash
.venv\Scripts\python.exe -m unittest tests/test_e2e_full.py
```
Expected output:
```text
Ran 6 tests in 9.227s
OK
```

---

## 9. REST API Reference

| Endpoint | Method | Parameters | Description |
|---|---|---|---|
| `/api/health` | `GET` | None | Returns live backend telemetry, CPU, RAM, and model readiness. |
| `/api/model-info` | `GET` | None | Returns technical details of ViT embedder, 115-D pipeline, and edit classes. |
| `/api/images` | `GET` | None | Lists available suspect and benchmark images. |
| `/api/upload` | `POST` | `multipart/form-data` | Uploads suspect images into a isolated investigation session directory. |
| `/api/analyze` | `POST` | JSON `{ analysis_id, similarity_threshold, confidence_filter }` | Reconstructs DAG, detects root image, orientations, and computes XAI. |
| `/api/predict-pair` | `POST` | JSON `{ image_a_path, image_b_path }` | Computes pairwise relationship probability, direction, and transformation. |
| `/api/predict-direction` | `POST` | JSON `{ image_a_path, image_b_path }` | Returns isolated directional hypothesis between two images. |
| `/api/predict-transformation` | `POST` | JSON `{ image_a_path, image_b_path }` | Returns predicted digital transformation type and confidence. |
| `/api/analysis/{id}/forest` | `GET` | `id: str` | Returns Kruskal Minimum Spanning Forest edges and weight semantics. |
| `/api/analysis/{id}/sensitivity` | `GET` | `id: str, source_id, target_id` | Returns feature-group sensitivity attribution across 115 dimensions. |
| `/api/analysis/{id}/explanation` | `GET` | `id: str` | Returns empirical noise residual heatmaps and forensic findings. |
| `/api/analysis/{id}/graph` | `GET` | `id: str` | Returns Cytoscape elements for interactive canvas visualization. |
| `/api/demo` | `POST` | `graph_id: str` (e.g. `graph_0001`) | Stages benchmark DAG and executes instant end-to-end inference. |
| `/api/download/{id}/html` | `GET` | `id: str` | Downloads the formal HTML investigation dossier for the case. |
| `/api/history` | `GET` | `limit: int, offset: int` | Returns historical provenance records from the SQLite database. |
| `/api/dataset` | `GET` | None | Returns SIPGD benchmark catalog with available DAG IDs and image counts. |

---

## 10. Research Citation

If you use this software or benchmark dataset in your research, please cite:

```bibtex
@inproceedings{rathika2026provenance,
  title={Explainable Image Provenance Analysis Via Graph Encoding With Vision Transformer In Forensics},
  author={Rathika, R. and Anandh D., Dhaya and Ahamed Y., Sharuk and Raj R., Mohan},
  booktitle={Department of Artificial Intelligence and Data Science},
  organization={Dr. Mahalingam College of Engineering and Technology},
  year={2026},
  address={Pollachi, Tamil Nadu, India}
}
```
