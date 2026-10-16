# Explainable AI (XAI) Evidence Audit — TRACE AI Forensic Audit

**Audited Modules**:
- Heatmap & Spatial Metric Generator: [`backend/app/services/xai_engine.py`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/backend/app/services/xai_engine.py)
- Empirical Forensic Rationale Generator: [`src/xai_explainer.py`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/xai_explainer.py)
- Feature Sensitivity Attribution Engine: [`backend/app/services/model_manager.py:550-579`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/backend/app/services/model_manager.py#L550-L579)

---

## 1. Executive Summary

A critical standard in legal and judicial image forensics is **explainability**: black-box neural network predictions are insufficient without verifiable, reproducible physical evidence.

TRACE AI implements a four-tiered explainability pipeline:
1. **Empirical Noise Residual Heatmaps**: 2D Jet overlays showing high-pass filtering differences.
2. **Deterministic Spatial Metrics**: Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM).
3. **Structured Forensic Rationale**: Technical bulleted summaries citing specific parent-to-child metric shifts.
4. **Feature Group Sensitivity**: Percentage contribution of 5 visual feature domains (ViT Latent, High-Pass Noise, Frequency Gradients, Color Space, Geometry).

---

## 2. Forensic Grounding & Integrity Verification

To prevent misleading evidence or fabricated claims in judicial contexts, the XAI engine was audited against strict evidentiary standards:

### Verification Checklist:
- **No Fabricated Sensors or Metadata**: The system does NOT invent camera serial numbers, GPS tags, or PRNU sensor fingerprint IDs.
- **Clear Evidentiary Separation**:
  - **Tier 1 (Directly Measured Physical Evidence)**: Dimensions ($W \times H$), file size (KB), MSE, PSNR, SSIM, empirical Laplacian gradient mean, and noise residual standard deviation.
  - **Tier 2 (Statistical Model Predictions)**: ViT CLS cosine similarity, relationship probability ($P(\text{rel})$), direction probability ($P(\text{dir})$), and top-1 transformation class.
  - **Tier 3 (Forensic Rationale)**: Deterministic explanation text explaining the physical significance of the observed deltas.
- **Handling of Ordinary Pixel Differences**: Pixel subtraction is never claimed as proof of provenance; it is framed as a spatial perceptual difference map.

---

## 3. High-Pass Noise Residual Heatmap Algorithm

The heatmap generation pipeline in [`backend/app/services/xai_engine.py:19-79`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/backend/app/services/xai_engine.py#L19-L79) operates as follows:
1. **Aspect Alignment**: Child image $I_B$ is rescaled to parent dimensions $(W_A, H_A)$ via area interpolation.
2. **Non-Linear Denoising (High-Pass Residual)**:
   $$R_A = |I_A - \text{median}_{3\times 3}(I_A)|, \quad R_B = |I_B - \text{median}_{3\times 3}(I_B)|$$
3. **Residual Difference Map**:
   $$\Delta R = |R_A - R_B|$$
4. **Spatial Regularization**:
   A $15 \times 15$ Gaussian smoothing filter attenuates single-pixel camera shot noise, highlighting contiguous regions of structural manipulation.
5. **Colormap Overlay**:
   Normalized to $[0, 255]$, color-mapped with `cv2.COLORMAP_JET`, and alpha-blended with the source image ($60\%$ original $+ 40\%$ heatmap).

---

## 4. XAI Evidence Across All 12 Transformation Categories

Following our code updates to [`src/xai_explainer.py`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/xai_explainer.py), every transformation category generates tailored, grounded forensic evidence:

| Transformation Category | Primary Evidence Metrics Reported | Forensic Grounding Rationale |
|---|---|---|
| **crop** | Bounding box reduction ($W_A \times H_A \to W_B \times H_B$) or canvas normalization note | Reports spatial framing loss vs full canvas rescaling. |
| **resize** | Spatial interpolation shift, pixel area delta | Cites exact pixel count change ($W_A H_A \to W_B H_B$). |
| **jpeg** | Estimated JPEG quality score ($Q_A \to Q_B$) | Cites 8x8 block boundary discontinuity shift. |
| **blur** | Laplacian edge gradient intensity drop ($\text{edge}_A \to \text{edge}_B$) | High-frequency spectral attenuation. |
| **sharpen** | Laplacian edge gradient accentuation ($\text{edge}_A \to \text{edge}_B$) | High-frequency edge gradient accentuation. |
| **noise** | High-pass spatial noise residual standard deviation ($\sigma_A \to \sigma_B$) | Additive stochastic noise residual increase. |
| **brightness** | Photometric mean luminance delta ($\Delta L$) | Reports whether luminance increased or attenuated. |
| **contrast** | Dynamic range standard deviation delta ($\Delta \sigma$) | Reports expansion or compression of dynamic range. |
| **color** | RGB channel variance and standard deviation shifts | Chromatic distribution shifts. |
| **rotation** | Angular realignment and coordinate plane boundary artifacts | Spatial interpolation artifacts and border shifts. |
| **perspective** | Planar homography warping across coordinate plane | Projective homography warping distortion. |
| **quality_degradation**| Compound multi-factor degradation (downscale + noise + JPEG) | Reports noise injection and JPEG quality simultaneously. |

---

## 5. Independent Pair XAI Support (Bug 2 Resolution Verification)

When an analysis evaluates two unrelated images or two disconnected nodes in a forest:
- Previously, the UI displayed `"No XAI Forensic Evidence Staged"`.
- Under the fix in [`backend/app/services/xai_engine.py:131-209`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/backend/app/services/xai_engine.py#L131-L209), `generate_xai_report_for_independent_pair` executes.
- The system generates a comparative noise residual difference map, measures MSE/PSNR/SSIM, and provides an explicit **Divergence Rationale**:
  ```text
  INDEPENDENT / UNRELATED IMAGES EXPLANATION
  Relationship Status: UNRELATED / INDEPENDENT
  Lineage Relatedness: 3.4% (Threshold: 55.0%)
  Evidence: Severe semantic divergence in ViT embedding space (CosSim: 0.082).
  Decision: Nodes are preserved as distinct genealogical trees.
  ```
This ensures complete forensic transparency for both derived edges and independent nodes.
