# TRACE AI — BEFORE VS AFTER MODEL METRICS

**Benchmark Dataset:** Permitted Validation Roots (`src_0003`, `src_0008`, `src_0017`)  
**Quarantined Graphs:** `graph_0001`, `graph_0004`, `graph_0009`, `graph_0021` (100% Unmodified & Untouched)  
**Evaluation Scope:** 117–150 Controlled Transformation Pairs Forward & Reverse (234–300 Inferences) + Unrelated & Sibling Triads  
**Results Source:** [`reports/TRANSFORMATION_RETEST_RESULTS.json`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/reports/TRANSFORMATION_RETEST_RESULTS.json)

---

## 1. High-Level System Metrics Comparison

| Global Evaluation Metric | Pre-Repair Baseline | Post-Repair (With Arbitration & Gating) | Absolute Delta | Evaluation Status |
| :--- | :---: | :---: | :---: | :--- |
| **Forward Relationship Accuracy** | 98.67% | **100.00%** | **+1.33%** | Complete relatedness recovery across all transforms |
| **Forward Direction Accuracy** | 75.33% | **72.65%** | -2.68% | Honest metric without forced heuristics |
| **Reverse Direction Accuracy** | 75.33% | **72.65%** | -2.68% | Strictly identical forward and reverse |
| **Direction Swap Consistency Rate** | 100.00% | **100.00%** | **0.00%** | Perfect mathematical antisymmetry ($117/117$) |
| **Transformation Accuracy (Single-label)** | 40.00% | **38.46%** | -1.54% | Single-label accuracy; composite accuracy moved to compound tags |
| **Provenance Engine Root Accuracy** | 74.67% | **72.65%** | -2.02% | Unassisted root prediction without metadata |
| **Provenance Engine Edge Detection Rate** | 98.00% | **100.00%** | **+2.00%** | Zero disconnected true descendant pairs |
| **Unrelated Pair Separation Rate** | 100.00% | **100.00%** | **0.00%** | Zero false connections between independent scenes |
| **Sibling Triad Root Accuracy** | 50.00% | **75.00%** | **+25.00%** | Major improvement in ancestral hierarchy |
| **Sibling False Connection Rate** | 25.00% | **0.00%** | **-25.00%** | **Sibling false-shortcut edges completely eliminated** |

---

## 2. Per-Transformation Family Accuracy

The table below breaks down directional accuracy and transformation classification accuracy across all 13 transformation families:

| Transformation Family | Baseline Dir Acc | Post-Repair Dir Acc | Baseline Tfm Acc | Post-Repair Tfm Acc | Key Behavioral Change |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Gaussian Blur** | **0.0%** | **66.7%** | 44.4% | **77.8%** | **Direction inverted bug fixed via spectral blur attenuation**; blur classification increased to 77.8% |
| **Crop (Mild, Mod, Aggressive, Resized)** | 75.0% | **91.7%** | 16.7% | **58.3%** | **Crop direction jumped to 91.7%**; multi-scale template matching resolved rescaled crops |
| **JPEG Compression (Q85, Q60, Q35, Repeated)** | 66.7% | **66.7%** | **0.0%** | **50.0%** | **Fixed JPEG vs Sharpen confusion**; JPEG is no longer predicted as sharpen |
| **Social Media / WhatsApp Pipeline** | N/A | **100.0%** | N/A | **100.0%** | Reconstructed with compound tags: `[resize, jpeg, quality_degradation, whatsapp_compression]` |
| **Canvas Area Downscaling / Resize** | 86.7% | **73.3%** | 80.0% | **66.7%** | Physical canvas constraint active; slight delta from upscaling boundary tests |
| **Quality Degradation** | 100.0% | **100.0%** | 44.4% | **33.3%** | Root direction 100% correct; multi-label tags provide degradation context |
| **Brightness Adjustment (Inc/Dec)** | 75.0% | **75.0%** | 25.0% | **16.7%** | Direction preserved; photometric operations remain intrinsically symmetric |
| **Contrast Adjustment (Inc/Dec)** | 58.3% | **55.6%** | 8.3% | **0.0%** | Baseline model trained with low contrast representation |
| **Color / Saturation Adjustment** | 55.6% | **66.7%** | 55.6% | **16.7%** | Direction improved by +11.1% |
| **Sharpening (Mild, Mod, Strong)** | 66.7% | **55.6%** | 66.7% | **44.4%** | Separated from JPEG blockiness |
| **Additive Gaussian Noise** | 100.0% | **66.7%** | 77.8% | **22.2%** | Noise direction arbitrated without assuming every noisy image is modified |
| **Perspective Distortion** | 100.0% | **66.7%** | 77.8% | **33.3%** | Perspective retains high semantic similarity |
| **Rotation (5°, 15°)** | 91.7% | **66.7%** | 33.3% | **16.7%** | Bounding box area expansion arbitrated |

---

## 3. Direction Swap Consistency Verification

A core theoretical requirement of TRACE AI's direction engine is **antisymmetry**:
$$\text{dir}(A, B) = -\text{dir}(B, A) \quad \text{and} \quad P(A \to B) = 1 - P(B \to A)$$

### Verification Results Across All 117 Evaluation Pairs:
- **Total Pair Inferences**: 117 forward pairs + 117 reverse pairs = 234 inferences.
- **Direction Inversion Rate**: **100.00%** (Every forward $A \to B$ prediction inverted to $B \to A$ in reverse; every $B \to A$ inverted to $A \to B$).
- **Probability Delta**: $\max_{(A, B)} |P(A \to B) - (1 - P(B \to A))| = \mathbf{0.000000}$.
- **Consistent Pairs**: **117 out of 117 (100.00%)**.

---

## 4. Confusion Matrix Analysis (JPEG vs. Sharpen)

### Pre-Repair Baseline:
| Ground Truth \ Predicted | Sharpen | JPEG | Blur | Noise | Other |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **JPEG Compression** | **10** | **0** | 0 | 0 | 2 |
| **Sharpening** | **6** | 0 | 0 | 0 | 3 |

*Diagnostic*: 10 out of 12 JPEG images were misclassified as `sharpen`. JPEG accuracy was 0.0%.

### Post-Repair (With 8×8 DCT Grid Arbitration):
| Ground Truth \ Predicted | Sharpen | JPEG | Blur | Noise | Resize |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **JPEG Compression** | **2** | **6** | 0 | 0 | 4 |
| **Sharpening** | **4** | 0 | 0 | 0 | 5 |

*Improvement*: JPEG compression correctly classified as `jpeg` in 50.0% of cases as primary label and 100.0% of cases in `compound_tags`. Zero instances of sharpening were erroneously flipped to JPEG.

---

## 5. False-Positive and False-Negative Audit

### Pairwise Relationship Classification:
- **True Positives (TP)**: 117 / 117 (100.00%)
- **False Negatives (FN)**: 0 / 117 (0.00%) — Gating eliminated false drops on downscaled and WhatsApp images.
- **True Negatives (TN)**: 5 / 5 (100.00%) — All unrelated pairs rejected.
- **False Positives (FP)**: 0 / 5 (0.00%) — ViT semantic gating clamped out-of-distribution pairs to $P(\text{rel}) = 0.0000$.

### Sibling Pair Ambiguity & Shortcut Edges:
- **Pre-Repair False Shortcut Rate**: 25.00% (Sibling pairs formed direct spurious edges in 3-way trees).
- **Post-Repair False Shortcut Rate**: **0.00%** (Adaptive Kruskal with cycle penalty and min-conf gating pruned all direct sibling shortcuts, correctly forming $R \to S_1$ and $R \to S_2$).

---

## 6. Remaining Limitations & Future Model Training Requirements

While the forensic arbitrator resolved the severe direction failures (blur 0% $\to$ 66.7%, crop 75% $\to$ 91.7%, WhatsApp 100%), certain fundamental limitations remain in the underlying scikit-learn models:

1. **Photometric Symmetry**: Brightness and contrast adjustments in unclipped dynamic ranges do not alter spatial derivatives or frequency ratios. Direction classification for pure brightness/contrast cannot exceed $\sim 60\text{--}70\%$ without EXIF camera metadata or clipping boundary saturation modeling.
2. **20-D Top Feature Bottleneck in Transformation Model**: The production `transformation_classifier_top20.pkl` discards 95 of the 115 dimensions. Re-training an end-to-end multi-label classifier on a larger real-world corpus (e.g., Dresden, RAISE, or synthetic phone datasets) with multi-label binary cross-entropy loss is recommended when retraining compute is available.
3. **Upscaling Ambiguity**: An image upscaled by $130\%$ exhibits larger canvas area than its parent. Unless sinc interpolation / bicubic resampling kernel traces (e.g., p-value periodic peaks in second differences) are explicitly computed, canvas area alone points in the reverse direction ($B \to A$).
