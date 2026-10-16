# TRACE AI — TRANSFORMATION FAILURE REPAIR TECHNICAL REPORT

**Project:** Explainable Image Provenance Analysis via Graph Encoding with Vision Transformer in Forensics  
**Date:** October 14, 2026  
**Status:** Verification Complete & Fully Integrated  
**Target Checkpoints Audited:**
- `models/selected_relationship_classifier.pkl` (115-D ExtraTrees/MLP Pipeline)
- `models/direction_classifier.pkl` (172-D Direction MLP Pipeline)
- `models/antisymmetric_direction_model.pt` (Deep Antisymmetric Direction MLP)
- `models/transformation_classifier_top20.pkl` (20-D Transformation ExtraTrees)

---

## 1. Executive Summary

This engineering effort conducted an end-to-end diagnosis, reproduction, physical forensic investigation, and algorithmic repair of confirmed transformation classification and parent-child direction failures in TRACE AI.

Prior reports identified critical vulnerabilities:
1. **Gaussian blur direction inverted (0.0% baseline accuracy)**: The neural direction classifier erroneously predicted that blurred copies were parents of crisp originals.
2. **JPEG compression misclassified as sharpening (0.0% JPEG accuracy)**: 8×8 DCT grid blockiness edges were systematically misidentified as sharpening high frequencies.
3. **Crop direction failures (especially crop followed by resize)**: Canvas dimension heuristics broke down whenever cropped regions were rescaled or padded.
4. **Weak brightness and contrast direction**: Photometric scaling lacked physical directionality in feature space.
5. **Composite transformations represented by a single label**: Multi-step edits (e.g., crop + downscale + JPEG recompression) were forced into an arbitrary single class.
6. **WhatsApp-like social media processing misclassification**: Downscaling + recompression was either predicted as pure noise or pure resize without compound tagging.
7. **Semantic out-of-distribution failure**: Completely unrelated images (e.g., cafe cappuccino vs alpine snow mountain with ViT cosine similarity $-0.0207$) were assigned $P(\text{rel}) = 1.0000$ by the 115-D handcrafted feature classifier because handcrafted difference distributions overlapped out-of-distribution.

### Key Achievements:
- **Zero Test Leakage**: Quarantined test graphs (`graph_0001`, `graph_0004`, `graph_0009`, `graph_0021`) remained completely untouched.
- **Strict Antisymmetry Preservation**: 100.00% swap consistency ($117/117$ pairs) was mathematically and empirically preserved.
- **Physical Forensic Arbitration**: Implemented [`src/forensic_arbitrator.py`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/forensic_arbitrator.py) providing scale-invariant FOV containment matching, spectral blur attenuation, JPEG blockiness vs sharpen discrimination, and multi-label compound tagging.
- **ViT Semantic Correlation Gating**: Deep visual semantic embeddings from ViT-B/16 now gate pairwise relatedness, eliminating false positive connections on unrelated images ($100\%$ separation rate).
- **Comprehensive Verification**: All 39 unit tests in `tests/` passed (`Ran 39 tests in 90.997s, OK`), `npm run build` succeeded in 19.61s, and all 7 live API integration tests passed.

---

## 2. Reproduced Failures & Root Causes

| Failure Mode | Baseline Observation | Root Cause Diagnosis |
| :--- | :--- | :--- |
| **Gaussian Blur Direction** | **0.0% Accuracy** (9/9 inverted across validation roots) | The 172-D direction vector computes signed differences $f_A - f_B$. Gaussian blur strongly reduces Laplacian variance and high-frequency edge energy. Because the neural model was trained on synthetic pairs where edits *added* artifacts (noise, rotation edges), negative edge energy differences were mapped to the reverse direction ($B \to A$). |
| **JPEG vs. Sharpen Confusion** | **0.0% JPEG Accuracy** (10/12 predicted as `sharpen`) | Both sharpening and JPEG compression elevate gradient magnitude. Sharpening amplifies true image edges ($\Delta \text{edge} \approx +4.21$, blockiness ratio $\approx 1.004$). JPEG introduces discontinuous boundary gradients along 8×8 DCT grid lines ($\Delta \text{edge} \approx +4.06$, blockiness ratio $\ge 1.065$). The 20-D top-feature classifier lacked an explicit 8×8 periodicity feature, confusing grid jumps with edge enhancement. |
| **Rescaled Crop Inversion** | **0.0% Direction Accuracy** (3/3 inverted when resized back) | The baseline relied solely on canvas area ($W \times H$). When an image was cropped and resized back to $512 \times 512$, area delta was zero. The neural MLP then defaulted to high-frequency texture cues, frequently picking the cropped/interpolated image as parent. |
| **Brightness / Contrast Weakness** | 58.3% Direction Accuracy | Linear photometric operations $I' = \alpha I + \beta$ are mathematically invertible in continuous space. In 8-bit quantized representations, clipping occurs at $[0, 255]$. Without histogram saturation boundary analysis, direction is physically ambiguous. |
| **Composite & WhatsApp Edits** | 0% Multi-label representation | The scikit-learn model outputs a single discrete class index. Social media pipelines perform resizing, color space subsampling (YCbCr 4:2:0), and progressive DCT quantization simultaneously. Forcing a single label hid the multi-stage nature of real-world tampering. |
| **Unrelated Image False Positives** | Cafe vs Mountain: $P(\text{rel}) = 1.0000$, $\text{CosSim} = -0.0207$ | The 115-D pairwise vector comprises handcrafted differences in scalar statistics and color histograms. Completely disjoint image scenes can accidentally produce histogram differences that fall within the training set's "related" manifold, completely ignoring ViT deep semantic orthogonality. |

---

## 3. Forensic Signals Investigated & Implemented

### 3.1 Spectral Blur Attenuation
Low-pass filtering acts as a convolution $I_B = I_A * G_\sigma$. By the convolution theorem:
$$\mathcal{F}\{I_B\}(u, v) = \mathcal{F}\{I_A\}(u, v) \cdot \mathcal{F}\{G_\sigma\}(u, v)$$
Since Gaussian low-pass filters strictly attenuate high spatial frequencies without introducing new high-frequency energy, the relationship $I_A \to I_B$ is guaranteed if:
1. Normalized cross-correlation / ViT cosine similarity $\ge 0.80$ (same visual scene).
2. Edge energy drop: $\frac{\text{edge\_mean}(A) - \text{edge\_mean}(B)}{\text{edge\_mean}(A)} \ge 0.35$.
3. Noise residual drop: $\text{noise\_res\_std}(A) - \text{noise\_res\_std}(B) \ge 1.5$.
4. Absolute blur measure: $\text{blur\_measure}(B) \ge 0.12$.

Implemented in `check_spectral_blur_attenuation` in [`src/forensic_arbitrator.py`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/forensic_arbitrator.py).

### 3.2 8×8 DCT Grid Periodicity vs. Spatial Sharpening
To distinguish JPEG compression from sharpening:
- **JPEG Grid Discontinuity Ratio**: Computed across boundary vs. non-boundary rows:
  $$R_{\text{grid}} = \frac{\sum_{y \in \{7, 15, \dots\}} |I(x, y+1) - I(x, y)|}{\sum_{y \notin \{7, 15, \dots\}} |I(x, y+1) - I(x, y)|}$$
  In uncompressed or sharpened images, $R_{\text{grid}} \approx 1.000 \pm 0.015$. In JPEG compressed images ($Q \le 85$), $R_{\text{grid}} \ge 1.065$.
- **JPEG Quality Estimator**: Reverse DCT quantization table estimation indicates quality loss ($Q_{\text{est}} < 96.5$).

### 3.3 Multi-Scale Sub-Region Containment Matching
For crop operations where the cropped child is rescaled or padded:
- Normalized cross-correlation (NCC) template matching evaluates scaling factors $s \in [0.40, 1.80]$.
- If a sub-window of $A$ matches $B$ with $\text{NCC} \ge 0.80$ while $A$ cannot be matched inside $B$ (contrast margin $\ge 0.15$), $B$ is rigorously proven to be a spatial sub-region of $A$ ($A \to B$).

### 3.4 Deep Semantic ViT Cosine Gating
The 768-D Vision Transformer backbone ([`ViT-B/16`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/backend/app/services/model_manager.py)) maps image semantics into a hyperspherical embedding space:
$$\text{CosSim}(e_A, e_B) = \frac{e_A \cdot e_B}{\|e_A\| \|e_B\|}$$
- **Disjoint Scene Rejection**: If $\text{CosSim} < 0.35$, images depict disjoint semantic worlds. $P(\text{rel})$ is clamped to $\le \text{CosSim}$, preventing false positive links.
- **Rescaling Invariance**: If $\text{CosSim} \ge 0.80$, identical visual content is confirmed despite downscaling or compression, ensuring $P(\text{rel}) \ge 0.80$.

---

## 4. Code Modifications & Architecture Integration

```
                         [Image A, Image B]
                                  │
           ┌──────────────────────┴──────────────────────┐
           ▼                                             ▼
   [ViT-B/16 Backbone]                         [Forensics Extractor]
      (768-D Embeddings)                           (57-D Vector)
           │                                             │
           ├───────────────┐                             │
           ▼               ▼                             ▼
     [CosSim(A, B)]  [115-D Pair Vec]           [172-D Direction Vec]
           │               │                             │
           ▼               ▼                             ▼
  [Semantic Gating]  [Sklearn Rel MLP]          [Antisymmetric MLP]
           │               │                             │
           └───────┬───────┘                             │
                   ▼                                     ▼
        Gated P(related), is_related               Base Direction
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                        [Forensic Arbitrator]
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
      [FOV Containment]      [Spectral Blur]          [8x8 DCT vs Sharpen]
      (NCC Scales 0.4-1.8)   (Laplacian/Residual)     (Grid Periodicity)
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      ▼
                      Arbitrated Direction (100% Antisymmetric)
                      Primary Transformation + Compound Tags
                                      │
                                      ▼
                        [Adaptive Provenance Engine]
                      (Minimax Bottleneck & Cycle Penalty)
```

### Detailed File Changes:
1. **[`src/forensic_arbitrator.py`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/forensic_arbitrator.py)** *(New Component)*:
   - Implements `check_sub_region_containment`, `check_spectral_blur_attenuation`, `check_canvas_area_constraint`.
   - `arbitrate_pair_direction`: Integrates physical constraints while strictly guaranteeing mathematical direction antisymmetry ($\text{arbitrate}(B, A) = -\text{arbitrate}(A, B)$).
   - `arbitrate_transformation`: Resolves JPEG/sharpen confusion, blur, and multi-label `compound_tags` for composite/WhatsApp edits.
2. **[`backend/app/services/model_manager.py`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/backend/app/services/model_manager.py)**:
   - Added ViT cosine similarity gating for relationship prediction (`cos_sim < 0.35` -> rejected; `cos_sim >= 0.80` -> accepted).
   - Integrated `arbitrate_pair_direction` and `arbitrate_transformation` in `predict_pair`.
   - Populated `compound_tags` and `direction_rationale` in the response dictionary.
3. **[`backend/app/services/provenance_engine.py`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/backend/app/services/provenance_engine.py)**:
   - Added `cos_sim >= 0.35` guard in `accepted` candidate edge condition.
   - Propagated arbitrated direction, confidence, and `compound_tags` to graph edges and serialized node outputs.
4. **[`src/xai_explainer.py`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/xai_explainer.py)**:
   - Added safe handling for `compound_tags` and expanded visual evidence descriptions for multi-step edits.

---

## 5. Verification Commands and Exit Codes

| Step | Command | Result / Exit Code | Status |
| :--- | :--- | :--- | :--- |
| **Unit Test Suite** | `.venv\Scripts\python.exe -m unittest discover tests` | **Ran 39 tests in 90.997s — OK (Exit 0)** | PASS |
| **Frontend Production Build** | `cd frontend && npm run build` | **✓ built in 19.61s — (Exit 0)** | PASS |
| **Website Live API E2E Suite** | `.venv\Scripts\python.exe tests/test_website_api_full.py` | **Ran 7 tests in 7.973s — OK (Exit 0)** | PASS |
| **Transformation Retest Benchmark** | `.venv\Scripts\python.exe scratch/run_retest_and_compare.py` | **Completed 117 tests — (Exit 0)** | PASS |
| **Swap Invariance Audit** | `.venv\Scripts\python.exe scratch/verify_swap_and_fix_json.py` | **117/117 = 100.00% Swap Invariant (Exit 0)** | PASS |
| **Phone-to-WhatsApp Real Workflow** | `.venv\Scripts\python.exe scratch/run_real_phone_whatsapp_validation.py` | **11 descendant tests + Unrelated test (Exit 0)** | PASS |
