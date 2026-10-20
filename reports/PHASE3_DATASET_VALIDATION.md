# TRACE AI — Phase 3 Dataset Generation & Integrity Validation Report

**Date:** October 18, 2026  
**Status:** VALIDATION PASSED (Ready for Phase 4 Model Training)  
**Target Output Directory:** `data/provenance_dataset_v2/`  
**Deterministic Seed:** `42`  
**Execution Platform:** Windows x64 (CPU Only, PyTorch 2.14.1+cpu)  
**Generation Runtime:** 70.46 seconds  
**Integrity Validation Runtime:** 40.12 seconds  

---

## 1. Executive Summary

Phase 3 generated a fully isolated, clean, deterministic, multi-generational provenance dataset under `data/provenance_dataset_v2/` adhering strictly to all preservation rules:
- **Legacy Protection:** The 795 legacy feature arrays in `image_provenance_dataset/features/individual_features/` and existing reports were 100% preserved and untouched.
- **Root Image Verification:** Exactly 100 procedural root source images were synthesized. All 100/100 SHA-256 and MD5 hashes matched `DATASET_SOURCES.csv` with zero drift.
- **DAG Multi-Generation Hierarchy:** 563 transformed descendants were synthesized across generations 1, 2, and 3, forming 100 directed acyclic provenance trees.
- **Pairwise Dataset Construction:** Exactly 2,252 directed image pairs were generated with a strictly balanced 50% positive (related) / 50% negative (unrelated) distribution and exact swap-antisymmetric labels (`A_TO_B`, `B_TO_A`, `NONE`).
- **Zero-Leakage Splitting:** Partitioned strictly at the **source root ID** level (70 Train / 15 Val / 15 Test). Verification confirmed **0.0% data leakage** between splits.
- **Multi-Label Coverage:** All 12 supported transformation classes are populated and verified against 12-dimensional binary ground-truth vectors $\mathbf{y} \in \{0, 1\}^{12}$.
- **Feature Descriptors:** 57-D forensic descriptors were pre-extracted for all 663 images and saved as individual `.npy` arrays and an indexed summary CSV.

---

## 2. Dataset Cardinality & Split Summary

| Partition | Source IDs | Image Count | Direct Edges | Directed Pairs | Pos Pairs (Related) | Neg Pairs (Unrelated) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | `source_0001` – `source_0070` (70) | 456 | 386 | 1,544 | 772 (50.0%) | 772 (50.0%) |
| **Validation** | `source_0071` – `source_0085` (15) | 111 | 96 | 384 | 192 (50.0%) | 192 (50.0%) |
| **Test** | `source_0086` – `source_0100` (15) | 96 | 81 | 324 | 162 (50.0%) | 162 (50.0%) |
| **Total** | **100 Sources** | **663 Images** | **563 Edges** | **2,252 Pairs** | **1,126 (50.0%)** | **1,126 (50.0%)** |

### Leakage Audit
- $\text{Train Sources} \cap \text{Val Sources} = \emptyset$ (0 overlap)
- $\text{Train Sources} \cap \text{Test Sources} = \emptyset$ (0 overlap)
- $\text{Val Sources} \cap \text{Test Sources} = \emptyset$ (0 overlap)
- **Data Leakage:** **0.00%** (Strict source-grouped isolation verified).

---

## 3. Direction & Symmetry Balance

Every positive parent-child transformation edge $A \xrightarrow{T} B$ has an exact mathematical counter-pair $B \xrightarrow{T^{-1}} A$ in the pairwise manifest to ensure swap-antisymmetry training:

| Direction Class | Pair Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| `A_TO_B` | 563 | 25.0% | Parent $\to$ Child (ground-truth forward edge) |
| `B_TO_A` | 563 | 25.0% | Child $\to$ Parent (swapped backward edge) |
| `NONE` | 1,126 | 50.0% | Unrelated distractor pairs (sampled strictly within split) |
| **Total** | **2,252** | **100.0%** | **Perfect 1:1 forward:reverse:distractor balance** |

---

## 4. Multi-Label Transformation Class Coverage

All 12 transformation classes supported by TRACE AI are represented in the 563 DAG edges:

| Index | Transformation Name | Edge Count | Frequency (%) | Sample Parameters Checked |
| :---: | :--- | :---: | :---: | :--- |
| 0 | `crop` | 38 | 6.7% | Box margins: 8%–15% |
| 1 | `resize` | 113 | 20.1% | Scale factors: 0.70x, 0.85x, 1.25x |
| 2 | `jpeg` | 83 | 14.7% | Quality: 30, 45, 60, 75 |
| 3 | `blur` | 42 | 7.5% | Gaussian $\sigma$: 1.5–2.5, ksize: 5–9 |
| 4 | `brightness` | 43 | 7.6% | Alpha factors: 0.70x, 1.30x |
| 5 | `contrast` | 55 | 9.8% | Contrast factors: 0.65x, 1.40x |
| 6 | `sharpen` | 50 | 8.9% | Kernel weights: 1.2–2.0 |
| 7 | `rotation` | 46 | 8.2% | Angles: 90°, 180°, 270° |
| 8 | `color` | 44 | 7.8% | Saturation shifts: 0.60x, 1.45x |
| 9 | `noise` | 88 | 15.6% | Gaussian std: 12.0–25.0 |
| 10 | `perspective` | 61 | 10.8% | Distortion magnitude: 0.08–0.14 |
| 11 | `quality_degradation` | 42 | 7.5% | Bilinear downscale 0.5x + upscale 2.0x |

*Note: In composite multi-step transformations, multiple bits in the 12-D ground-truth vector $\mathbf{y}$ are active simultaneously, providing realistic multi-label training signals.*

---

## 5. Automated Integrity Checks Executed

The validation script `src/validate_dataset_v2.py` executed 8 comprehensive tests:

| Check | Description | Target Standard | Result | Notes |
| :---: | :--- | :--- | :---: | :--- |
| **1** | Image Registration | All image records exist in `images.csv` | **PASS** | 663 unique images registered |
| **2** | Image Readability | Open & decode every image using OpenCV | **PASS** | 663/663 valid, 0 corrupted |
| **3** | DAG Hierarchy | No broken parent IDs, generations valid (0–3) | **PASS** | 100 trees, 0 circular refs |
| **4** | Split Non-Leakage | Source IDs mutually disjoint across splits | **PASS** | 0 overlap across Train/Val/Test |
| **5** | Multi-Label Vectors | Binary $\in \{0, 1\}$, exact length 12 | **PASS** | 563/563 edge vectors compliant |
| **6** | Pair Balance | Exactly 50% pos / 50% neg, exact direction labels | **PASS** | 1,126 Pos / 1,126 Neg / 563 A$\to$B / 563 B$\to$A |
| **7** | Forensic Features | 57-D descriptors valid, no NaNs/Infs | **PASS** | 663 `.npy` arrays (shape: (57,)) |
| **8** | Legacy Preservation | Verify existing files untouched | **PASS** | Exactly 795 legacy `.npy` preserved |

---

## 6. Directory Structure Created

```
data/provenance_dataset_v2/
├── dataset_summary.json
├── originals/                      # 100 verified procedural original root images
│   ├── original_0001.png
│   └── ... (100 files)
├── images/                         # 563 transformed descendants across gen 1-3
│   ├── img_0001_gen1_n01.png
│   └── ... (563 files)
├── metadata/                       # Split-partitioned manifests
│   ├── sources.csv                 # 100 roots with deterministic hashes & splits
│   ├── images.csv                  # 663 registered images with resolution & gen
│   ├── edges_train.csv             # 386 transformation edges
│   ├── edges_val.csv               # 96 transformation edges
│   ├── edges_test.csv              # 81 transformation edges
│   ├── pairs_train.csv             # 1,544 directed training pairs
│   ├── pairs_val.csv               # 384 directed validation pairs
│   ├── pairs_test.csv              # 324 directed test pairs
│   ├── graphs_train.json           # 70 ground truth DAG graph structures
│   ├── graphs_val.json             # 15 ground truth DAG graph structures
│   └── graphs_test.json            # 15 ground truth DAG graph structures
└── features/                       # Forensic descriptors
    ├── forensic_features.csv       # Consolidated 57-D CSV with image indices
    └── individual_features/        # 663 individual numpy arrays
        ├── original_0001_features.npy
        └── ... (663 files)
```

---

## 7. Resource Usage & Platform Performance

- **Peak RAM Consumption:** ~1.4 GB (well below 15.67 GB system ceiling)
- **Disk Usage:** ~98 MB total for `data/provenance_dataset_v2/`
- **CPU Generation Time:** 70.46s (704 ms per root DAG including feature extraction)
- **External Downloads:** Exactly 0 bytes downloaded (zero pretrained weights downloaded)
- **Target Compatibility:** Fully compatible with PyTorch CPU dataloaders (`torch.utils.data.Dataset`).

---

## 8. Conclusion & Readiness for Phase 4

Phase 3 is complete and thoroughly validated. The new dataset fulfills every requirement:
1. Zero overwrites of legacy code or data.
2. Exact hash reproduction of procedural sources.
3. Deterministic multi-generation provenance graphs.
4. 0.0% data leakage across Train/Val/Test splits.
5. Symmetrically balanced pair classes and antisymmetry labels.
6. Full 12-class transformation coverage.
7. Pre-extracted 57-D forensic vectors ready for instantaneous PyTorch training.

**Phase 4 (Model Training & Evaluation) is ready to begin upon user approval.**
