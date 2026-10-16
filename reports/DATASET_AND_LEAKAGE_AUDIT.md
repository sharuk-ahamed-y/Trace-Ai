# Dataset and Leakage Audit — TRACE AI Forensic Audit

**Audited Directories**:
- `dataset/images/train/` (102 images)
- `dataset/images/val/` (22 images)
- `dataset/images/test/` (30 images, Quarantined)
- `dataset/metadata/` (`graphs_*.json`, `edges_*.csv`, `pairs_*.csv`, `images.csv`)
**Source Code References**:
- [`scratch/audit_dataset_and_splits.py`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/scratch/audit_dataset_and_splits.py)
- [`dataset/README.md`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/dataset/README.md)

---

## 1. Executive Summary

A core vulnerability in machine learning for computer vision and digital forensics is **data leakage**: evaluating models on images that share identical background scenes, lighting, or camera sensors with training examples.

This audit verified the genealogical partitioning of the TRACE AI benchmark dataset, confirmed strict graph-disjoint separation across splits, audited feature scaler provenance, and validated the quarantine of the held-out test graphs.

---

## 2. Genealogical Partitioning Audit

The TRACE AI dataset contains **21 unique source scenes** organized into 21 independent provenance graphs (`graph_0001` through `graph_0021`):

| Split | Assigned Provenance Graphs | Image Count | Edge Count | Node Roles |
|---|---|---|---|---|
| **Train** | `graph_0002`, `0005`, `0006`, `0007`, `0010`, `0011`, `0012`, `0013`, `0014`, `0015`, `0016`, `0018`, `0019`, `0020` | 102 | 88 | 14 Roots, 88 Descendants |
| **Validation** | `graph_0003`, `0008`, `0017` | 22 | 19 | 3 Roots, 19 Descendants |
| **Test (Quarantined)** | `graph_0001`, `0004`, `0009`, `0021` | 30 | 26 | 4 Roots, 26 Descendants |
| **Total** | **21 Graphs** | **154 Images** | **133 Edges** | **21 Roots, 133 Descendants** |

### Graph-Disjoint Separation Rules Verified:
1. **Zero Source Overlap**:
   Every provenance graph originates from a distinct source graphic or photograph (e.g. checkerboards, color blocks, diagonal gradients, natural screenshots, noise fields). No source scene exists in more than one partition:
   $$\text{Sources}(\text{Train}) \cap \text{Sources}(\text{Val}) \cap \text{Sources}(\text{Test}) = \emptyset$$
2. **Descendant Containment**:
   All descendants (generations 1, 2, and 3) of a given source image reside strictly within the split assigned to that source graph. No descendant of a training root leaks into validation or test partitions.

---

## 3. Data Leakage & Scaler Isolation Verification

| Leakage Check | Audit Standard | Verification Status | Evidence / File Path |
|---|---|---|---|
| **Test Set Quarantine** | Test graphs must not be accessed during optimization or tuning. | **PASS (STRICT QUARANTINE)** | Graphs 0001, 0004, 0009, 0021 were never accessed during benchmark generation or parameter tuning. |
| **Feature Scaler Provenance** | Scalers must be fit strictly on training splits. | **PASS** | `StandardScaler` in `selected_relationship_classifier.pkl` was fitted exclusively on `pairs_train.csv`. |
| **Filename Independence** | Predictions must not use image filenames or IDs. | **PASS** | `ModelManager.predict_pair` takes raw file paths, extracts features from pixel buffers, and ignores filename tokens. |
| **Metadata Independence** | Prediction pipeline must not access ground-truth metadata. | **PASS** | `provenance_engine.py` computes all edge and root assignments from feature vectors and model logits alone. |
| **Threshold Independence** | Similarity and direction thresholds must not be tuned on test split. | **PASS** | Thresholds ($\tau_{\text{rel}} = 0.55$, $\tau_{\text{conf}} = 0.40$) were calibrated on validation graphs (`graph_0003`, `0008`, `0017`). |

---

## 4. Dataset Limitations & Future Benchmark Recommendations

While the current dataset satisfies strict graph-disjoint isolation, the audit identified important dataset-scale limitations that must be transparently reported:

1. **Source Diversity Limit (21 Sources)**:
   The dataset comprises 21 distinct source images. While this provides 154 images and hundreds of combinatorial pair evaluations, it is too small to cover the full diversity of real-world digital photography (e.g. extreme ISO noise, sensor PRNU variation across camera brands, deepfake/generative inpainting).
2. **Synthetic Transformation Generator**:
   Transformations were generated using deterministic PIL / OpenCV pipelines. Real-world wild social media images often undergo undocumented platform transcodings and lossy re-quantization tables.
3. **Recommendation**:
   Future work should expand the benchmark using large-scale forensic corpuses such as **NC2016**, **DSO-1**, or **IEEE IFS-TC Image Forensics Challenge** datasets while preserving the strict graph-disjoint partitioning validated here.
