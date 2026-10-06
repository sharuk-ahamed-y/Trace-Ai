# Project Analysis: Image Provenance Analysis Research

This document provides a comprehensive audit of the existing codebase located in `image-provenance-using-graph-main` and outlines the exact architecture, gaps, and modification strategy required to construct the **Synthetic Image Provenance Graph Dataset (SIPGD)**.

---

## 1. Existing Pipeline

The current repository contains a initial prototype designed for image provenance graph reconstruction. Its workflow consists of:

```
Source Images (data/originals/)
       │
       ▼
Single-Step Child Generation (src/generate_dataset.py)
       │
       ▼
Pretrained ViT-B/16 Embeddings (src/feature_extractor.py)
       │
       ▼
Pairwise Feature Concatenation (src/pair_features.py)
       │
       ▼
Two-Branch MLPs for Direction & Transformation (src/train_classifiers.py)
       │
       ▼
MST Construction & Edge Orientation (src/graph_builder.py)
       │
       ▼
Graph Visualization (src/visualize_graph.py)
```

- **Execution Flow**:
  1. `train.py` calls `generate_dataset.py`, which applies 8 fixed transformations independently to each source image in `data/originals/`.
  2. Embeddings are extracted using torchvision's pretrained ViT-B/16 (CLS token, 768-dim) and cached in `models/embedding_cache.pkl`.
  3. `DirectionDataset` and `TransformationDataset` build pairwise feature vectors of dimension 2305 (`[emb_A, emb_B, |emb_A - emb_B|, cos_sim(A, B)]`).
  4. MLP models are trained to classify pair direction (`parent_to_child` vs `child_to_parent`) and transformation operation.
  5. `graph_builder.py` calculates pairwise cosine distances across images, computes an undirected Minimum Spanning Tree (MST), applies the MLPs to orient edges, and uses a cycle resolution heuristic.

---

## 2. Existing Dataset Format

- **Source Images**: Stored in `data/originals/`.
- **Generated Images**: Stored in `data/generated/` using single-step naming convention `{original_stem}__{transformation}.jpg`.
- **Labels File**: `data/labels/edges.csv` containing:
  - `parent_path`
  - `child_path`
  - `transformation`
  - `transformation_parameters` (JSON string)
  - `direction` (always `"parent_to_child"`)

---

## 3. Existing Transformations

Implemented in `src/transforms.py`:
1. `crop`: Crop 10–30% from one side (`top`, `bottom`, `left`, `right`) and resize back to original size.
2. `resize`: Downscale by factor 0.40–0.85 and upscale back to original resolution.
3. `jpeg`: Lossy JPEG re-encoding at quality 15–70.
4. `blur`: Gaussian blur with radius 1.0–5.0.
5. `brightness`: Brightness multiplier factor 0.50–1.60.
6. `contrast`: Contrast multiplier factor 0.50–1.60 (configured in `config.py`).
7. `rotation`: Rotation angle -30° to +30° (configured in `config.py`).
8. `text_insertion`: Text watermark overlay (configured in `config.py`).

*Note*: Only `crop`, `resize`, `jpeg`, `blur`, and `brightness` were functional in `src/transforms.py`; others were defined in `config.py` but lacked implementation handlers in `APPLY_FN`.

---

## 4. Existing Labels

- **Direction Labels**: 2 classes (`parent_to_child`: 0, `child_to_parent`: 1).
- **Transformation Labels**: 8 classes (`crop`, `resize`, `rotation`, `brightness`, `contrast`, `blur`, `text_insertion`, `jpeg`).
- **Graph Labels**: None. Ground truth trees, branching structures, root IDs, distractor pairs, and pairwise relationship matrices were missing.

---

## 5. Existing Model Architecture

- **Feature Extractor**: Pretrained Vision Transformer (`vit_b_16`, ImageNet-1k weights), frozen backbone extracting 768-dim L2-normalized CLS embeddings.
- **Pairwise Representation**:
  $$\text{Feature}(A, B) = [\mathbf{e}_A \,\|\, \mathbf{e}_B \,\|\, |\mathbf{e}_A - \mathbf{e}_B| \,\|\, \text{cos\_sim}(\mathbf{e}_A, \mathbf{e}_B)] \in \mathbb{R}^{2305}$$
- **Classifiers**:
  - `MLP` with 2 hidden layers: `Linear(2305, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 128) -> ReLU -> Dropout(0.3) -> Linear(128, n_classes)`.
  - Direction MLP: 2 output logits.
  - Transformation MLP: 8 output logits.
- **Graph Builder**: NetworkX `minimum_spanning_tree` over complete distance graph ($d(A, B) = 1 - \text{cos\_sim}(\mathbf{e}_A, \mathbf{e}_B)$), oriented edge-by-edge with cycle reversal.

---

## 6. What is Missing for a Proper Provenance Dataset

The current pipeline is a basic single-step pair generator rather than a benchmark provenance graph dataset. Specifically:

1. **No Multi-Generation / Multi-Level Tree Graphs**:
   - Current: `Original -> Child` (depth 1 only).
   - Missing: Multi-hop provenance trees ($Root \to Child_A \to Child_{A1} \to Child_{A1a}$) and branching trees ($Root \to Child_A$, $Root \to Child_B$).
2. **No Distractor / Unrelated Images**:
   - Current: Every image in the graph is derived from the same source image.
   - Missing: Distractor images from unrelated source identities. Without distractors, pairwise classifiers cannot evaluate `related = 0` vs `related = 1`.
3. **No Explicit Root / Original Identification Ground Truth**:
   - Missing explicit graph-level metadata denoting `root_image_id`.
4. **No Pairwise Benchmark Datasets (`pairs_train.csv`, `pairs_val.csv`, `pairs_test.csv`)**:
   - Missing balanced positive/negative pairwise datasets with explicit ground truth for relationship, direction (`A_TO_B`, `B_TO_A`, `NONE`), and transformation.
5. **No Forensic Feature Dataset**:
   - Missing physical, spectral, noise residual, histogram, and dimension features essential for Explainable AI (XAI) and hybrid forensic models.
6. **Limited Transformation Set & Hardcoded Ranges**:
   - Missing realistic operations like sharpening, color saturation, Gaussian noise, perspective/skew transformations, quality degradation combinations, and diverse parameter sampling.
7. **No Standardized Dataset Structure & Automated Validator**:
   - Missing standardized benchmark folders (`dataset/images/`, `dataset/metadata/`, `dataset/features/`), strict source-split validation to guarantee zero data leakage, and dataset integrity checks.
8. **No Automated Package & Documentation**:
   - Missing standard `DATASET_CARD.md`, `DATASET_ACCESS.md`, `requirements_dataset.txt`, and single `.zip` distribution archive.

---

## 7. What Files We Will Add / Change

### Added / Upgraded Files:

1. `PROJECT_ANALYSIS.md` — Detailed repo audit report (this document).
2. `DATASET_SOURCES.md` / `DATASET_ACCESS.md` — Strategic audit of public image sources & legal access terms (NIST MFC-Prov, COCO, Wikimedia, Unsplash).
3. `src/transforms.py` — Upgraded transformation suite (12 realistic operations with randomized parameter ranges).
4. `src/forensic_features.py` — Forensic feature extractor (dimensions, aspect ratio, JPEG quality estimates, RGB stats, histograms, edge statistics, high-pass noise residuals, DCT stats).
5. `src/generate_provenance_dataset.py` — Multi-generation provenance graph dataset generator producing multi-level DAGs, branching graphs, distractor images, pairwise data, graph JSONs, and strict identity-based splits.
6. `src/xai_explainer.py` — XAI explanation generator producing textual evidence and feature rationale for predicted provenance edges.
7. `visualize_graph.py` — Standalone script for visualizing provenance graphs with roots, transformations, and distractors.
8. `validate_dataset.py` — Dataset integrity validator (checks missing files, corruption, hashes, leakage, graph consistency, cycles).
9. `train_provenance.py` — Comprehensive multi-experiment benchmark script evaluating ViT similarity, ViT+MLP, ViT+Forensics, and MST reconstruction with confidence filtering.
10. `setup_provenance_dataset.py` — Master single-command orchestrator for end-to-end dataset creation, validation, packaging, and statistics reporting.
11. `DATASET_CARD.md` — Official dataset documentation card for SIPGD.
12. `requirements_dataset.txt` — Environment requirements for reproducing SIPGD.
13. `image_provenance_dataset.zip` — Single compressed distribution package containing the complete model-ready dataset.
