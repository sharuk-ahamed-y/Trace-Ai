# Noise and Compression Evaluation — TRACE AI Forensic Audit

**Transformations Audited**:
- Gaussian Noise (`apply_noise`, $\sigma \in \{6.0, 15.0, 28.0\}$)
- JPEG Compression (`apply_jpeg`, $Q \in \{85, 60, 30\}$)
- Repeated JPEG Compression ($Q=70 \times 3$ cycles)
- WhatsApp-Style Processing (spatial downscaling + $Q=75$ compression + EXIF stripping)
**Source Code References**:
- [`src/transforms.py:151`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L151) (`apply_jpeg`)
- [`src/transforms.py:247`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L247) (`apply_noise`)
- [`src/forensic_features.py:29`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/forensic_features.py#L29) (`estimate_jpeg_quality`)

---

## 1. Executive Summary

Digital image compression and noise injection are primary markers in multimedia forensics. In digital investigations, sensor PRNU (photo-response non-uniformity), stochastic noise addition, and lossy discrete cosine transform (DCT) quantization provide distinct spatial and frequency signatures.

### Benchmark Highlights:
- **Gaussian Noise**:
  - Relationship Recall: **100.0%**
  - Direction Accuracy: **100.0%** (9/9 correct $A \to B$)
  - Root Identification Accuracy: **100.0%**
  - Transformation Classification Accuracy: **77.8%** (7/9 `noise`, 2/9 `quality_degradation`)
- **JPEG Compression**:
  - Relationship Recall: **100.0%**
  - Direction Accuracy: **66.7%** (8/12 correct $A \to B$)
  - Transformation Classification Accuracy: **0.0%** (**10/12 misclassified as `sharpen`**, 1 `color`, 1 `resize`)
- **WhatsApp-Style Processing**:
  - Relationship Recall: **100.0%**
  - Direction Accuracy: **66.7%**
  - Transformation Classification Accuracy: Classified as `resize` (100%)

---

## 2. In-Depth Analysis: Why Gaussian Noise Achieves 100% Direction Accuracy

The physical model for additive image noise is:
$$I_{\text{noisy}}(x, y) = I(x, y) + \eta(x, y), \quad \eta \sim \mathcal{N}(0, \sigma^2)$$

In [`backend/app/services/model_manager.py:306`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/backend/app/services/model_manager.py#L306), the 17-D forensic vector extracts `noise_res_std`:
$$\text{res}(x, y) = |I(x, y) - \text{median}_{3\times 3}(I(x, y))|$$
$$\sigma_{\text{res}} = \text{std}(\text{res}(x, y))$$

When noise is injected, $\sigma_{\text{res}}$ increases monotonically from approximately $1.5 - 4.5$ in clean images to $8.2 - 24.6$ in noisy images. The 172-D direction vector incorporates signed difference:
$$\Delta \sigma_{\text{res}} = \sigma_{\text{res}}^{(B)} - \sigma_{\text{res}}^{(A)} > 0$$
This provides an unambiguous physical gradient: entropy and residual variance increase along the arrow of time, allowing the `AntisymmetricDirectionMLP` to determine the true parent with **100% precision**.

---

## 3. Systematic Classification Defect: JPEG Compression Misidentified as Sharpening

In the transformation classifier ([`models/transformation_classifier_top20.pkl`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/models/transformation_classifier_top20.pkl)), pure JPEG compression achieved **0.0% top-1 accuracy**, with 10 out of 12 test pairs classified as **`sharpen`**.

### Root Cause Mechanism:
1. **8x8 DCT Block Discontinuities**: Standard JPEG divides an image into $8 \times 8$ pixel blocks, applies a 2D DCT, and quantizes high-frequency coefficients. At block boundaries (pixels $x \equiv 0 \pmod 8$), adjacent reconstructed blocks exhibit step discontinuities (blocking artifacts).
2. **Laplacian Edge Statistics**:
   The feature extractor computes:
   - Feature 12: `edge_mean` (mean of $3 \times 3$ Laplacian kernel response)
   - Feature 13: `edge_std`
   - Feature 14: `edge_max`
   The 8x8 block grid introduces artificial high-frequency edges across the entire image.
3. **Classifier Weighting**:
   In `transformation_classifier_top20.pkl`, the top features are:
   - Index 12: `abs_diff[edge_mean]` (Laplacian edge mean)
   - Index 13: `abs_diff[edge_std]`
   The model associates an increase in Laplacian edge gradients with the label `sharpen`. Because `estimate_jpeg_quality` was not selected among the top 20 features (it ranked 27th in feature importance), the classifier has no competing feature to veto the `sharpen` prediction!

---

## 4. WhatsApp-Style Compression Analysis

Social media platforms (e.g. WhatsApp, Facebook, Instagram) process user uploads through a pipeline that downscales images to a standard resolution (e.g. max dimension 1600 px or aspect fit), re-compresses using JPEG at quality 70–80, and strips all EXIF metadata.

In our benchmark:
- Lineage detection is **100.0% preserved**. The ViT-B/16 backbone retains semantic alignment despite moderate spatial downscaling and compression.
- Direction accuracy is **66.7%**. The downscaling step provides a negative area delta ($A_B < A_A$), which reinforces the physical direction prior.
- Transformation label: The model predicts `resize` (3/3), capturing the dominant spatial transformation.

---

## 5. Summary Table: Noise & Compression Test Results

| Source | Variant | Quality / Sigma | MSE | PSNR (dB) | SSIM | Direction | Pred Tfm | GT Tfm |
|---|---|---|---|---|---|---|---|---|
| `circles_root` | Noise Low | $\sigma=6.0$ | 35.8 | 32.59 | 0.9681 | **A_TO_B** | noise | noise |
| `circles_root` | Noise Med | $\sigma=15.0$ | 224.2 | 24.62 | 0.9412 | **A_TO_B** | noise | noise |
| `circles_root` | Noise High | $\sigma=28.0$ | 781.9 | 19.20 | 0.8920 | **A_TO_B** | noise | noise |
| `circles_root` | JPEG Mild | $Q=85$ | 4.2 | 41.90 | 0.9982 | **A_TO_B** | **sharpen** | jpeg |
| `circles_root` | JPEG Mod | $Q=60$ | 14.8 | 36.43 | 0.9950 | **A_TO_B** | **sharpen** | jpeg |
| `circles_root` | JPEG Low | $Q=30$ | 48.6 | 31.26 | 0.9890 | **A_TO_B** | **sharpen** | jpeg |
| `circles_root` | Repeated JPEG | $Q=70 \times 3$ | 16.2 | 36.03 | 0.9942 | **A_TO_B** | **sharpen** | jpeg |
| `screenshot_root` | WhatsApp Sim | $Q=75$, scale=0.75 | 22.4 | 34.62 | 0.9850 | **A_TO_B** | **resize** | quality_deg |

---

## 6. Recommendations for Engineering Improvements

1. **Include JPEG Blockiness Metric in Classifier**:
   Replace one of the redundant histogram bins in `transformation_classifier_top20.pkl` with Feature 5: `jpeg_quality_est` (which explicitly calculates the ratio of block-boundary differences to interior differences). This directly discriminates JPEG blocking from authentic unsharp masking.
2. **Dedicated WhatsApp Pipeline Classifier**:
   Expose compound labels in the API when both dimensional reduction and DCT block boundaries are detected simultaneously.
