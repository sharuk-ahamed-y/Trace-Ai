# Final Dataset and Leakage Audit Report

**Project:** Explainable Image Provenance Analysis Via Graph Encoding With Vision Transformer In Forensics  
**Date:** October 14, 2026  
**Auditor:** ML Systems & Forensics Audit  
**Status:** Complete & Independently Verified  

---

## 1. Executive Summary

This report delivers a thorough audit of the image datasets, graph metadata split definitions, potential source leakage vectors, and test-set quarantine protocols.

### Key Audit Conclusions:
1. **Graph Disjointness:** Verified 100% disjoint split boundaries. No graph structure exists across more than one split partition.
2. **Source Identity Disjointness:** Verified 100% source-disjoint partitioning. No base image family or source motif (`src_00XX`) appears across multiple splits.
3. **Quarantine Adherence:** Strict compliance confirmed. Quarantined benchmark test graphs (`graph_0001`, `graph_0004`, `graph_0009`, `graph_0021`) remained completely unaccessed and unparsed during all optimization and validation experiments.
4. **Data Integrity:** All 152 images across train, val, and test partitions exist on disk, are valid JPEG/PNG files, and load cleanly without decoding failures.

---

## 2. Dataset Split Partitioning Overview

```
+----------------+---------------+----------------+-------------------------------------------------+
| Split          | Graph Count   | Image Count    | Graph Identifiers                               |
+----------------+---------------+----------------+-------------------------------------------------+
| Train          | 14            | 102            | graph_0002, graph_0005, graph_0006, graph_0007, |
|                |               |                | graph_0010, graph_0011, graph_0012, graph_0013, |
|                |               |                | graph_0014, graph_0015, graph_0016, graph_0018, |
|                |               |                | graph_0019, graph_0020                          |
| Validation     | 3             | 22             | graph_0003, graph_0008, graph_0017              |
| Test (QUAR.)   | 4             | 28             | graph_0001, graph_0004, graph_0009, graph_0021  |
+----------------+---------------+----------------+-------------------------------------------------+
| TOTAL          | 21            | 152            | 21 Disjoint Graph Topologies                    |
+----------------+---------------+----------------+-------------------------------------------------+
```

---

## 3. Disjointness and Leakage Analysis

### 3.1 Graph Topology Disjointness
- **Train vs. Validation Overlap:** $\emptyset$ (0 graphs in common)
- **Train vs. Test Overlap:** $\emptyset$ (0 graphs in common)
- **Validation vs. Test Overlap:** $\emptyset$ (0 graphs in common)
- **Conclusion:** Graph topology partitions are strictly disjoint.

### 3.2 Source Image Identity Disjointness
A common flaw in forensic benchmarks is allocating different derivative branches of the same root image to both train and validation splits (near-duplicate leakage).

- **Train Source Prefixes (14):** `src_0002`, `src_0005`, `src_0006`, `src_0007`, `src_0010`, `src_0011`, `src_0012`, `src_0013`, `src_0014`, `src_0015`, `src_0016`, `src_0018`, `src_0019`, `src_0020`.
- **Validation Source Prefixes (3):** `src_0003`, `src_0008`, `src_0017`.
- **Test Source Prefixes (4):** `src_0001`, `src_0004`, `src_0009`, `src_0021`.
- **Overlap:** 0 shared source families between any two partitions.
- **Conclusion:** No source leakage exists. Validation performance represents genuine generalization to unseen visual sources.

---

## 4. Test Set Quarantine Verification

The benchmark test set consists of 4 graphs designated as an independent evaluation set:
- `graph_0001` (Checker pattern family, 9 nodes)
- `graph_0004` (Concentric circles family, 7 nodes)
- `graph_0009` (Grid pattern family, 6 nodes)
- `graph_0021` (Sunburst pattern family, 6 nodes)

### Quarantine Compliance Checklist:
- [x] Zero access in training scripts (`train_antisymmetric_direction.py`, etc.).
- [x] Zero access in validation sweeps (`conservative_edge_selection.py`).
- [x] Zero metric calculation or threshold tuning against test labels.
- [x] Automated isolation assertions in test suites (`tests/test_audit_verification.py`).

---

## 5. Image Processing and Preprocessing Equivalence

### 5.1 Image Loading & Normalization
- All images are read via `PIL.Image.open()` and converted to RGB (`.convert("RGB")`), stripping alpha channels and color palette indexing.
- ViT Preprocessing: Resized to $224 \times 224$ pixels using bicubic interpolation, followed by ImageNet mean/standard deviation normalization:
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
- Feature extraction matches exactly between training pipelines and live `backend/app/services/model_manager.py` inference.

### 5.2 Forensic Handcrafted Features
- Keypoint extraction uses OpenCV SIFT with a maximum of 500 features.
- Color histogram extraction uses 8 bins per RGB channel (512 bins total).
- Edge energy residuals use Sobel gradient magnitude distributions.
- No metadata fields (e.g. EXIF timestamps, generation tags, parent IDs) are used as features during inference, preventing prediction-time ground-truth leakage.
