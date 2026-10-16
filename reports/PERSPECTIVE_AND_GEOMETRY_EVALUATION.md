# Perspective and Geometry Evaluation — TRACE AI Forensic Audit

**Transformations Audited**:
- Rotation (`apply_rotation`, angle $\in \{3.0^\circ, 15.0^\circ, 90.0^\circ\}$, expand $\in \{\text{False}, \text{True}\}$)
- Perspective Transformation (`apply_perspective`, distortion $\in \{0.06, 0.12, 0.20\}$)
**Source Code References**:
- [`src/transforms.py:218`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L218) (`apply_rotation`)
- [`src/transforms.py:262`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L262) (`apply_perspective`)

---

## 1. Executive Summary

Geometric transformations alter the spatial grid of pixel coordinates through linear or projective homographies:
$$\mathbf{x}' = \mathbf{H} \mathbf{x}, \quad \mathbf{H} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & 1 \end{bmatrix}$$

### Benchmark Findings:
- **Perspective Transformation**:
  - Relationship Recall: **100.0%** (9/9 pairs detected as related)
  - Direction Accuracy: **100.0%** (9/9 correct $A \to B$)
  - Root Accuracy: **100.0%**
  - Transformation Classification Accuracy: **77.8%** (7/9 `perspective`, 1 `sharpen`, 1 `rotation`)
  - Mean PSNR: **12.20 dB**, Mean SSIM: **0.4159**
- **Rotation**:
  - Relationship Recall: **100.0%** (12/12 pairs detected as related)
  - Direction Accuracy: **91.7%** (11/12 correct $A \to B$)
  - Root Accuracy: **91.7%**
  - Transformation Classification Accuracy: **33.3%** (4/12 `rotation`, 4 `perspective`, 3 `sharpen`, 1 `resize`)
  - Mean PSNR: **14.65 dB**, Mean SSIM: **0.5087**

---

## 2. Robustness of Vision Transformer to Geometric Distortions

A major technical question in ViT-based forensics is whether patch tokenization ($16 \times 16$ non-overlapping patches in ViT-B/16) degrades under non-grid-aligned geometric warping.

### Findings:
1. **Semantic Invariance**:
   Despite low pixel-level SSIM ($0.41 - 0.51$) and low PSNR ($12 - 15 \text{ dB}$), ViT CLS cosine similarity remains consistently high:
   - Perspective moderate: $\text{CosSim} = 0.88 - 0.94$
   - Rotation $15^\circ$: $\text{CosSim} = 0.91 - 0.96$
   - 90-degree orthogonal rotation: $\text{CosSim} = 0.84 - 0.89$
   The multi-head self-attention mechanism successfully aggregates features across distorted patches, preventing false graph disconnects.
2. **Directional Signals in Projective Warping**:
   Projective warping introduces non-uniform spatial interpolation (stretching near vanishing points and compression at receding edges). This gradient of interpolation blur and pixel coordinate distortion provides strong directional features to the `AntisymmetricDirectionMLP`, resulting in **100% directional accuracy**.

---

## 3. Rotation vs. Perspective Classification Overlap

In the classifier evaluation, **33.3% of rotation examples were classified as `perspective`**.

### Mathematical Reason:
Planar rotation is a special case of affine transformation, which itself is a subgroup of projective transformations:
$$\mathbf{H}_{\text{rotation}} = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix} \subset \mathbf{H}_{\text{projective}}$$

When an image is rotated without expanding the canvas (`expand=False`), triangular black wedges or cropped corners appear at the four boundaries. These edge-boundary transitions produce spatial and frequency gradient changes that closely resemble mild perspective shearing.

---

## 4. Empirical Evaluation Table

| Source | Geometric Type | Parameter | P(rel) | Cos Sim | Pred Direction | Pred Tfm | GT Tfm |
|---|---|---|---|---|---|---|---|
| `circles_root` | Rotation Small | angle=$3.0^\circ$, exp=False | 0.8812 | 0.9754 | **A_TO_B** | rotation | rotation |
| `circles_root` | Rotation Mod | angle=$15.0^\circ$, exp=False | 0.8144 | 0.9525 | **A_TO_B** | rotation | rotation |
| `circles_root` | Rotation 90 | angle=$90.0^\circ$, exp=False | 0.7650 | 0.8920 | **A_TO_B** | **perspective** | rotation |
| `circles_root` | Rotation Expand | angle=$15.0^\circ$, exp=True | 0.7920 | 0.9310 | **A_TO_B** | **sharpen** | rotation |
| `circles_root` | Perspective Mild | distortion=0.06 | 0.8540 | 0.9620 | **A_TO_B** | perspective | perspective |
| `circles_root` | Perspective Mod | distortion=0.12 | 0.8120 | 0.9150 | **A_TO_B** | perspective | perspective |
| `circles_root` | Perspective Strong| distortion=0.20 | 0.7810 | 0.8740 | **A_TO_B** | perspective | perspective |
| `screenshot_root` | Perspective Mod | distortion=0.12 | 0.9420 | 0.9580 | **A_TO_B** | perspective | perspective |

---

## 5. Explainability & Visualization Verification

In [`src/xai_explainer.py`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/xai_explainer.py):
- For `rotation`, the explanation reports geometric rotation with border realignment artifacts.
- For `perspective`, the explanation reports projective planar homography warping distortion across the coordinate plane.
- In the frontend XAI heatmaps, the noise residual difference map displays characteristic peripheral radial gradients highlighting the corner and edge warping regions.
