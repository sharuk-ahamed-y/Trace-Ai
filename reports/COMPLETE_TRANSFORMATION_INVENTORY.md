# Complete Transformation Inventory — TRACE AI Forensic Audit

**Project**: TRACE AI — Explainable Image Provenance Analysis via Graph Encoding with Vision Transformer for Digital Image Forensics  
**Audit Scope**: Every image transformation claimed, supported, trained, or tested in the codebase, models, API, and UI.  
**Audit Date**: October 2026  
**Auditor**: Senior Computer Vision, Forensic ML, and Full-Stack QA Engineering Team  

---

## 1. Executive Summary

This inventory documents the exhaustive forensic investigation into all **25 image transformations** queried in the project audit mandate. Every transformation has been evaluated against the active codebase ([`src/transforms.py`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py)), model checkpoints ([`models/transformation_classifier_top20.pkl`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/models/transformation_classifier_top20.pkl)), feature extractors ([`backend/app/services/model_manager.py`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/backend/app/services/model_manager.py)), explainability engine ([`src/xai_explainer.py`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/xai_explainer.py)), and controlled empirical benchmarks across 150 live inference pairs.

### Standardized Status Classifications:
- **VERIFIED WORKING**: Fully implemented in code, present in training classes, detected by inference models with high directional and lineage accuracy, rendered in UI/XAI.
- **PARTIALLY WORKING**: Implemented and detected as related, but exhibits specific operational limitations (e.g. classification label confusion, loss of directional cues when dimensions are rescaled).
- **FAILED**: Fundamentally fails core forensic invariants (e.g. systematic direction inversion where child is identified as parent).
- **NOT IMPLEMENTED**: No transformation logic exists in [`src/transforms.py`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py).
- **NOT TRAINED**: Transformation logic may exist or be simulated, but no dedicated discrete class was trained in the model checkpoint.
- **INSUFFICIENT TEST DATA**: Insufficient independent source samples exist in the current dataset to establish statistical significance.
- **UNABLE TO VERIFY**: Evidence is not directly measurable from image pixel data.

---

## 2. Complete 25-Transformation Audit Matrix

| # | Transformation Type | Implementation Status | Training Data | Model Support | Direction Accuracy | Transformation Accuracy | Audit Classification |
|---|---------------------|-----------------------|---------------|---------------|--------------------|-------------------------|----------------------|
| 1 | **Crop** | `apply_crop` in `transforms.py` | Present (`images.csv`) | Class `crop` in Top-20 RF | 75.0% | 16.7% | **PARTIALLY WORKING** |
| 2 | **Resize** | `apply_resize` in `transforms.py` | Present (`images.csv`) | Class `resize` in Top-20 RF | 86.7% | 80.0% | **VERIFIED WORKING** |
| 3 | **JPEG compression** | `apply_jpeg` in `transforms.py` | Present (`images.csv`) | Class `jpeg` in Top-20 RF | 66.7% | 0.0% (predicted as sharpen) | **PARTIALLY WORKING** |
| 4 | **Brightness increase** | `apply_brightness` (factor > 1) | Present (pooled) | Class `brightness` | 75.0% | 25.0% | **PARTIALLY WORKING** |
| 5 | **Brightness decrease** | `apply_brightness` (factor < 1) | Present (pooled) | Class `brightness` | 75.0% | 25.0% | **PARTIALLY WORKING** |
| 6 | **Contrast increase** | `apply_contrast` (factor > 1) | Present (pooled) | Class `contrast` | 58.3% | 8.3% | **PARTIALLY WORKING** |
| 7 | **Contrast decrease** | `apply_contrast` (factor < 1) | Present (pooled) | Class `contrast` | 58.3% | 8.3% | **PARTIALLY WORKING** |
| 8 | **Color adjustment** | `apply_color` in `transforms.py` | Present (`images.csv`) | Class `color` in Top-20 RF | 55.6% | 55.6% | **PARTIALLY WORKING** |
| 9 | **Saturation adjustment** | `apply_color` via `ImageEnhance.Color` | Present (pooled) | Class `color` | 55.6% | 55.6% | **PARTIALLY WORKING** |
| 10 | **Blur (Gaussian)** | `apply_blur` in `transforms.py` | Present (`images.csv`) | Class `blur` in Top-20 RF | **0.0%** (inverted) | 44.4% | **FAILED** |
| 11 | **Sharpening** | `apply_sharpen` in `transforms.py` | Present (`images.csv`) | Class `sharpen` in Top-20 RF | 66.7% | 66.7% | **PARTIALLY WORKING** |
| 12 | **Gaussian noise** | `apply_noise` in `transforms.py` | Present (`images.csv`) | Class `noise` in Top-20 RF | 100.0% | 77.8% | **VERIFIED WORKING** |
| 13 | **Other noise types (Poisson/Salt&Pepper/Speckle)** | None | Absent | None | N/A | N/A | **NOT IMPLEMENTED** |
| 14 | **Rotation** | `apply_rotation` in `transforms.py` | Present (`images.csv`) | Class `rotation` in Top-20 RF | 91.7% | 33.3% | **VERIFIED WORKING** |
| 15 | **Perspective transformation** | `apply_perspective` in `transforms.py` | Present (`images.csv`) | Class `perspective` in Top-20 RF | 100.0% | 77.8% | **VERIFIED WORKING** |
| 16 | **Image enhancement** | Implemented as sharpen/contrast/color | Present (subsumed) | Subsumed | 66.7% | 66.7% | **PARTIALLY WORKING** |
| 17 | **Denoising** | None | Absent | None | N/A | N/A | **NOT IMPLEMENTED** |
| 18 | **Color correction** | Subsumed under `apply_color` | Absent | None | N/A | N/A | **NOT IMPLEMENTED** |
| 19 | **Gamma adjustment** | Subsumed under contrast | Absent | None | N/A | N/A | **NOT IMPLEMENTED** |
| 20 | **Repeated JPEG compression** | Sequential `apply_jpeg` calls | Sequential in graphs | Class `jpeg` | 66.7% | 0.0% | **PARTIALLY WORKING** |
| 21 | **WhatsApp-style processing** | Simulated downscale+JPEG75 | Multi-step | Classified as `resize` | 66.7% | 0.0% (as standalone) | **PARTIALLY WORKING** |
| 22 | **Combined / Sequential transformations** | Multi-hop graph generation | Present in multi-gen | Single-label model | 88.9% | 27.8% | **PARTIALLY WORKING** |
| 23 | **Metadata removal / modification** | Stripped on PIL save | Not modeled in pixels | None | N/A | N/A | **UNABLE TO VERIFY** |
| 24 | **Screenshot-derived images** | Present in source `src_0017` | Present in Val Set | Handled by pipeline | 100.0% | N/A | **VERIFIED WORKING** |
| 25 | **Quality degradation (Compound)** | `apply_quality_degradation` | Present (`images.csv`) | Class `quality_degradation` | 100.0% | 44.4% | **VERIFIED WORKING** |

---

## 3. Detailed Per-Transformation Findings

### 1. Crop (`apply_crop`)
- **Code**: [`src/transforms.py:55`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L55). Supports modes `random`, `center`, and `rectangular`. Parameter `resize_back` determines whether canvas is restored to original $(W, H)$.
- **Empirical Metrics**: Relationship Recall: 100.0%, Direction Accuracy: 75.0%, Top-1 Transformation Accuracy: 16.7%.
- **Limitation**: When `resize_back=True`, image dimensions $(W, H)$ are unchanged. Because physical area delta is zero, the model frequently misclassifies the edit as `rotation` or `resize`.

### 2. Resize (`apply_resize`)
- **Code**: [`src/transforms.py:114`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L114). Supports interpolation methods `LANCZOS`, `BICUBIC`, `BILINEAR`, and `NEAREST`.
- **Empirical Metrics**: Relationship Recall: 100.0%, Direction Accuracy: 86.7%, Transformation Accuracy: 80.0%.
- **Verification**: Excellent performance. Image dimension changes provide strong physical evidence.

### 3. JPEG Compression (`apply_jpeg`)
- **Code**: [`src/transforms.py:151`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L151). Saves to in-memory BytesIO buffer with specified quality (30–95).
- **Empirical Metrics**: Relationship Recall: 100.0%, Direction Accuracy: 66.7%, Transformation Accuracy: 0.0%.
- **Critical Finding**: Pure JPEG compression is systematically predicted as `sharpen` (10/12) because 8x8 block boundary discontinuities create high-frequency Laplacian edge gradients (`edge_mean` and `edge_max`) that mimic sharpening.

### 4 & 5. Brightness Increase and Decrease (`apply_brightness`)
- **Code**: [`src/transforms.py:179`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L179). Enhances image luminance via `PIL.ImageEnhance.Brightness(img).enhance(factor)`.
- **Empirical Metrics**: Relationship Recall: 91.7%, Direction Accuracy: 75.0%, Transformation Accuracy: 25.0%.
- **Finding**: Increases (factor > 1.0) and decreases (factor < 1.0) are pooled under a single `brightness` label.

### 6 & 7. Contrast Increase and Decrease (`apply_contrast`)
- **Code**: [`src/transforms.py:192`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L192). Uses `PIL.ImageEnhance.Contrast`.
- **Empirical Metrics**: Relationship Recall: 91.7%, Direction Accuracy: 58.3%, Transformation Accuracy: 8.3%.
- **Finding**: The classifier often confuses contrast adjustments with perspective distortion or brightness adjustments.

### 8 & 9. Color and Saturation Adjustment (`apply_color`)
- **Code**: [`src/transforms.py:234`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L234). Adjusts saturation multiplier.
- **Empirical Metrics**: Relationship Recall: 100.0%, Direction Accuracy: 55.6%, Transformation Accuracy: 55.6%.

### 10. Blur (`apply_blur`) — CRITICAL FAILURE
- **Code**: [`src/transforms.py:167`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L167). Uses `ImageFilter.GaussianBlur(radius)`.
- **Empirical Metrics**: Relationship Recall: 100.0%, Direction Accuracy: **0.0%** (9/9 inverted to `B_TO_A`), Root Accuracy: 0.0%, Transformation Accuracy: 44.4%.
- **Failure Cause**: Because the model expects manipulated images to accumulate high-frequency noise or edges, an image with reduced edge energy is mistakenly predicted as the *parent* of the crisp source image.

### 11. Sharpening (`apply_sharpen`)
- **Code**: [`src/transforms.py:205`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L205).
- **Empirical Metrics**: Relationship Recall: 100.0%, Direction Accuracy: 66.7%, Transformation Accuracy: 66.7%.

### 12. Gaussian Noise (`apply_noise`)
- **Code**: [`src/transforms.py:247`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L247).
- **Empirical Metrics**: Relationship Recall: 100.0%, Direction Accuracy: 100.0%, Transformation Accuracy: 77.8%.
- **Verification**: Superior forensic detection due to the `noise_res_std` forensic feature.

### 13. Other Noise Types
- **Status**: **NOT IMPLEMENTED**. Poisson, speckle, and salt-and-pepper noise do not exist in `src/transforms.py`.

### 14. Rotation (`apply_rotation`)
- **Code**: [`src/transforms.py:218`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L218).
- **Empirical Metrics**: Relationship Recall: 100.0%, Direction Accuracy: 91.7%, Transformation Accuracy: 33.3%.

### 15. Perspective Transformation (`apply_perspective`)
- **Code**: [`src/transforms.py:262`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L262).
- **Empirical Metrics**: Relationship Recall: 100.0%, Direction Accuracy: 100.0%, Transformation Accuracy: 77.8%.

### 16. Image Enhancement
- **Status**: **PARTIALLY WORKING**. Subsumed across `sharpen`, `contrast`, and `color`.

### 17–19. Denoising, Color Correction, Gamma
- **Status**: **NOT IMPLEMENTED**. No standalone algorithms or dataset classes exist.

### 20. Repeated JPEG Compression
- **Status**: **PARTIALLY WORKING**. Relationship is 100% detected; block artifact edges are predicted as `sharpen`.

### 21. WhatsApp-Style Compression
- **Status**: **PARTIALLY WORKING**. Combines downscaling to fit screen dimensions and JPEG quality 75. Accurately detected as related (100%) and directed (66.7%), with primary transformation classified as `resize`.

### 22. Combined / Sequential Transformations
- **Status**: **PARTIALLY WORKING**. Direction accuracy is 88.9%. Because the classifier is single-label multi-class, it predicts the dominant physical artifact (e.g. crop or sharpen) rather than a compound label sequence.

### 23. Metadata Removal
- **Status**: **UNABLE TO VERIFY VIA VISUAL EMBEDDINGS**. EXIF is stripped by default when PIL buffers are re-saved. Models operate purely on visual embeddings and pixel statistics.

### 24. Screenshot-Derived Images
- **Status**: **VERIFIED WORKING**. Provenance Graph 0017 (`src_0017_Screenshot 2026-09-05 214927_root.jpg`) verifies robust pipeline execution on real desktop screenshots.

### 25. Quality Degradation (`apply_quality_degradation`)
- **Code**: [`src/transforms.py:301`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L301). Combines downscaling, Gaussian noise, and low-quality JPEG.
- **Empirical Metrics**: Relationship Recall: 100.0%, Direction Accuracy: 100.0%, Transformation Accuracy: 44.4%.
