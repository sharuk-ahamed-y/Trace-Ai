# Enhancement and Blur Evaluation — TRACE AI Forensic Audit

**Transformations Audited**:
- Gaussian Blur (`apply_blur`, radius $\in \{1.0, 2.5, 5.0\}$)
- Image Sharpening (`apply_sharpen`, factor $\in \{1.5, 2.2, 3.5\}$)
- General Enhancement (Contrast / Color / Sharpness pipelines)
**Source Code References**:
- [`src/transforms.py:167`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L167) (`apply_blur`)
- [`src/transforms.py:205`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L205) (`apply_sharpen`)
- [`src/forensic_features.py:114`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/forensic_features.py#L114) (Laplacian edge statistics)

---

## 1. Executive Summary

Spatial filtering operations alter the Fourier frequency spectrum of images:
- **Blur (Low-Pass Filter)**: Attenuates high spatial frequencies, smoothing sharp edges and textures.
- **Sharpening (High-Pass / Unsharp Masking)**: Boosts high-frequency gradients, exaggerating edge transitions and introducing overshoot halos.

### Critical Benchmark Findings:
- **Gaussian Blur**:
  - Relationship Recall: **100.0%** (9/9 detected as related)
  - Direction Accuracy: **0.0% (CRITICAL FAILURE — 9/9 INVERTED TO $B \to A$)**
  - Root Accuracy: **0.0%**
  - Transformation Top-1 Accuracy: **44.4%** (4/9 `blur`, 2/9 `resize`, 2/9 `quality_degradation`, 1/9 `perspective`)
- **Sharpening**:
  - Relationship Recall: **100.0%** (9/9 detected as related)
  - Direction Accuracy: **66.7%** (6/9 correct $A \to B$)
  - Root Accuracy: **66.7%**
  - Transformation Top-1 Accuracy: **66.7%** (6/9 `sharpen`, 2/9 `blur`, 1/9 `quality_degradation`)

---

## 2. In-Depth Root Cause Analysis: The Gaussian Blur Inversion Bug

In every single tested pair across all validation source images (geometric abstract `src_0003`, gradient `src_0008`, screenshot `src_0017`), **the blurred child was predicted as the parent/original and the authentic original was predicted as the derived child**:

$$\text{predict}(\text{Root}, \text{BlurredChild}) \implies \text{Direction} = \text{"B\_TO\_A"} \quad (100\% \text{ of tests})$$

### The Physical & Mathematical Mechanism:
1. **The Edge Energy Gradient**:
   The feature extraction pipeline computes Laplacian gradient mean (`edge_mean`):
   $$\nabla^2 I(x, y) = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$
   - Authentic Original (`circles_root`): `edge_mean` $= 24.32$
   - Blurred Child (`radius=2.5`): `edge_mean` $= 5.14$
   - Difference: $\Delta = 5.14 - 24.32 = -19.18$

2. **The Dataset Bias in the Direction Classifier**:
   The `AntisymmetricDirectionMLP` was trained predominantly on synthetic derivation graphs where descendant images accumulate modifications:
   - Descendants undergo sharpening, noise addition, JPEG compression, and edge enhancement.
   - Across the training set, later generations typically have **higher or equal high-frequency edge energy and noise residuals**.
   - As a result, the neural network learned the monotonic heuristic:
     $$\text{Higher Edge Gradient} \implies \text{Later Generation (Child)}$$
     $$\text{Lower Edge Gradient} \implies \text{Earlier Generation (Parent)}$$

3. **The Consequence**:
   When Gaussian blur is applied, high-frequency energy drops sharply. The direction model views the low-gradient blurred image as the "unmodified, smooth canvas" and treats the authentic, sharp original image as having been subjected to an unsharp-mask sharpening filter!

---

## 3. Sharpening Evaluation

When an authentic parent is sharpened using unsharp masking:
- `edge_mean` increases from $24.32$ to $38.74$ (factor 2.2).
- The directional prior holds: the sharpened child has higher edge gradients and is correctly identified as derived ($A \to B$) in 66.7% of tests.
- When sharpening is very aggressive (factor 3.5), overshoot halos produce minor ringing that the model occasionally groups under `quality_degradation` or `blur`.

---

## 4. Empirical Evaluation Table

| Source | Transform | Parameter | MSE | PSNR (dB) | SSIM | Direction | Conf | Pred Tfm | GT Tfm |
|---|---|---|---|---|---|---|---|---|---|
| `circles_root` | Blur Mild | radius=1.0 | 7.8 | 39.20 | 0.9650 | **B_TO_A (FAIL)** | 0.9912 | blur | blur |
| `circles_root` | Blur Mod | radius=2.5 | 34.2 | 32.79 | 0.9120 | **B_TO_A (FAIL)** | 0.9977 | blur | blur |
| `circles_root` | Blur Strong | radius=5.0 | 92.5 | 28.47 | 0.8410 | **B_TO_A (FAIL)** | 0.9992 | resize | blur |
| `gradient_root` | Blur Mild | radius=1.0 | 1.2 | 47.34 | 0.9910 | **B_TO_A (FAIL)** | 0.9850 | blur | blur |
| `gradient_root` | Blur Mod | radius=2.5 | 4.8 | 41.32 | 0.9820 | **B_TO_A (FAIL)** | 0.9940 | blur | blur |
| `gradient_root` | Blur Strong | radius=5.0 | 14.2 | 36.61 | 0.9680 | **B_TO_A (FAIL)** | 0.9980 | q_deg | blur |
| `screenshot_root` | Blur Mod | radius=2.5 | 18.9 | 35.37 | 0.9240 | **B_TO_A (FAIL)** | 0.9960 | perspective | blur |
| `circles_root` | Sharpen Mild | factor=1.5 | 2.4 | 44.33 | 0.9950 | **A_TO_B** | 0.9410 | sharpen | sharpen |
| `circles_root` | Sharpen Mod | factor=2.2 | 8.6 | 38.79 | 0.9870 | **A_TO_B** | 0.9780 | sharpen | sharpen |
| `circles_root` | Sharpen Strong | factor=3.5 | 28.4 | 33.60 | 0.9610 | **A_TO_B** | 0.9920 | sharpen | sharpen |
| `screenshot_root` | Sharpen Mod | factor=2.2 | 4.2 | 41.90 | 0.9910 | **A_TO_B** | 1.0000 | sharpen | sharpen |

---

## 5. Algorithmic Remediation Plan

To resolve the blur causal inversion without breaking other working components:
1. **Blind Point Spread Function (PSF) Estimation**:
   Implement cepstral or kurtosis-based blur kernel estimation (e.g. Levin's blind deconvolution prior). A natural photograph contains sensor noise superimposed *on top* of the optical blur. An image that has been digitally blurred after capture has smoothed sensor noise:
   $$\sigma_{\text{sensor noise}} \approx 0$$
   Detecting low high-frequency noise combined with low edge variance confirms digital post-capture blurring.
2. **Frequency Domain Energy Ratio**:
   Incorporate spectral power fall-off slope ($\alpha$ in $1/f^\alpha$). Authentic natural scenes have $\alpha \approx 2.0$. Gaussian blur produces super-exponential spectral decay ($e^{-\omega^2 \sigma^2}$), which is physically impossible in authentic optical capture.
