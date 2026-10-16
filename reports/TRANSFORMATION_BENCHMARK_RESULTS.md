# Transformation Benchmark Results — TRACE AI Forensic Audit

**Benchmark Suite**: Controlled Transformation Evaluation (`scratch/run_rigorous_transformation_audit.py`)  
**Data Sources**: Permitted Validation Source Images:
1. `src_0003_circles_black_gold_root.jpg` (Geometric abstract)
2. `src_0008_grad_v_red_blue_root.jpg` (Continuous gradient)
3. `src_0017_Screenshot 2026-09-05 214927_root.jpg` (Real UI desktop screenshot)
**Evaluation Scale**: 150 live forward/reverse inference tests, 5 unrelated graph pairs, 4 sibling 3-way graphs.  
**Test Quarantine Status**: Graphs 0001, 0004, 0009, 0021 strictly isolated and unaccessed.  

---

## 1. Overall Benchmark Summary

| Metric | Score | Sample Count | Analysis & Interpretation |
|--------|-------|--------------|---------------------------|
| **Parent-Child Lineage Recall (Fwd)** | **98.67%** | 148 / 150 | Exceptional lineage detection across almost all transformation intensities. |
| **Direction Accuracy (Forward)** | **75.33%** | 113 / 150 | High overall accuracy, penalized primarily by Gaussian blur inversion (0%). |
| **Direction Accuracy (Reverse)** | **75.33%** | 113 / 150 | Exactly matches forward accuracy due to anti-symmetric formulation. |
| **Direction Swap Consistency** | **100.00%** | 150 / 150 | $logit(B, A) \equiv -logit(A, B)$; maximum swap confidence delta is **0.000000**. |
| **Root Identification Accuracy** | **74.67%** | 112 / 150 | Component root correctly assigned to ground-truth parent in 74.67% of pairs. |
| **Transformation Classifier Top-1 Accuracy** | **40.00%** | 60 / 150 | Moderate top-1 accuracy; heavily impacted by JPEG->sharpen confusion. |
| **Unrelated Image Separation Rate** | **100.00%** | 5 / 5 | Zero false edges invented between unrelated images; completely disjoint. |
| **Sibling False Connection Rate** | **25.00%** | 1 / 4 | Kruskal MST creates sibling shortcut in 1 out of 4 triad tests. |

---

## 2. Per-Transformation Family Breakdown

| Transformation Family | Test Count | Rel Recall | Direction Acc | Root Acc | Top-1 Tfm Acc | Mean PSNR (dB) | Mean SSIM |
|-----------------------|------------|------------|---------------|----------|---------------|----------------|-----------|
| **crop** | 12 | 100.0% | 75.0% | 75.0% | 16.7% | 16.53 | 0.4238 |
| **resize** | 15 | 100.0% | 86.7% | 86.7% | 80.0% | 38.94 | 0.9752 |
| **jpeg** | 12 | 100.0% | 66.7% | 66.7% | 0.0% | 36.53 | 0.9945 |
| **brightness** | 12 | 91.7% | 75.0% | 75.0% | 25.0% | 21.89 | 0.9151 |
| **contrast** | 12 | 91.7% | 58.3% | 58.3% | 8.3% | 23.87 | 0.9386 |
| **color** | 9 | 100.0% | 55.6% | 44.4% | 55.6% | 22.18 | 0.8035 |
| **blur** | 9 | 100.0% | **0.0%** | **0.0%** | 44.4% | 34.07 | 0.9248 |
| **sharpen** | 9 | 100.0% | 66.7% | 66.7% | 66.7% | 41.97 | 0.9892 |
| **noise** | 9 | 100.0% | 100.0% | 100.0% | 77.8% | 26.48 | 0.9551 |
| **rotation** | 12 | 100.0% | 91.7% | 91.7% | 33.3% | 14.65 | 0.5087 |
| **perspective** | 9 | 100.0% | 100.0% | 100.0% | 77.8% | 12.20 | 0.4159 |
| **quality_degradation**| 9 | 100.0% | 100.0% | 100.0% | 44.4% | 26.43 | 0.9537 |
| **whatsapp_compression**| 3 | 100.0% | 66.7% | 66.7% | 0.0% | 34.74 | 0.9852 |
| **composite** | 18 | 100.0% | 88.9% | 88.9% | 27.8% | 22.21 | 0.6924 |

---

## 3. Direction Swap Invariance Analysis

Every pair $(A, B)$ was tested in both forward $(A, B)$ and reverse $(B, A)$ sequence through `ModelManager.predict_pair`.

- **Anti-symmetric Direction Guarantee**:
  $$\text{logit}(B, A) = -\text{logit}(A, B) \implies P(B \to A) = 1.0 - P(A \to B)$$
- **Observed empirical consistency**:
  - Out of 150 tested pairs: **150 / 150 (100.00%)** maintained exact swap consistency.
  - When input order is swapped, the model identifies the identical physical image as the source.
  - Mean confidence difference across swapped orders: **$0.000000$**.

---

## 4. Confusion Matrix for Transformation Classification

| Ground Truth \ Predicted | crop | resize | jpeg | blur | brightness | contrast | color | sharpen | noise | perspective | q_deg | rotation | none |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **crop (12)** | 2 | 4 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 3 | 0 |
| **resize (15)** | 0 | **12** | 0 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| **jpeg (12)** | 0 | 1 | **0** | 0 | 0 | 0 | 1 | **10** | 0 | 0 | 0 | 0 | 0 |
| **brightness (12)** | 0 | 0 | 0 | 0 | **3** | 1 | 1 | 0 | 0 | 6 | 0 | 0 | 1 |
| **contrast (12)** | 0 | 0 | 0 | 0 | 2 | **1** | 2 | 1 | 0 | 5 | 0 | 0 | 1 |
| **color (9)** | 0 | 0 | 1 | 0 | 0 | 0 | **5** | 3 | 0 | 0 | 0 | 0 | 0 |
| **blur (9)** | 0 | 2 | 0 | **4** | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 |
| **sharpen (9)** | 0 | 0 | 0 | 2 | 0 | 0 | 0 | **6** | 0 | 0 | 1 | 0 | 0 |
| **noise (9)** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **7** | 0 | 2 | 0 | 0 |
| **rotation (12)** | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 4 | 0 | **4** | 0 |
| **perspective (9)** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | **7** | 0 | 1 | 0 |
| **q_deg (9)** | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 1 | **4** | 0 | 0 |
| **whatsapp (3)** | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **composite (18)** | 5 | 2 | 0 | 0 | 1 | 0 | 0 | 4 | 0 | 2 | 2 | 2 | 0 |

---

## 5. Key Forensic Insights & Engineering Conclusions

1. **Noise, Perspective, and Quality Degradation Are Forensically Unambiguous**:
   - 100% directional accuracy and root accuracy across all test source images.
   - Spatial high-pass noise residuals (`noise_res_std`) and geometric alignment provide decisive physical evidence.

2. **The Gaussian Blur Causal Inversion Bug**:
   - Blur achieved **0.0% direction accuracy** (9 out of 9 pairs inverted).
   - The system systematically predicts that the blurred child is the parent of the original.
   - Root cause: The direction classifier learned a prior that images with lower edge intensity are earlier in the generational chain.

3. **JPEG Compression / Sharpening Artifact Coupling**:
   - 10 out of 12 pure JPEG compressed images were misclassified as `sharpen`.
   - 8x8 DCT block boundaries increase Laplacian edge intensity, fooling the top-20 classifier.
