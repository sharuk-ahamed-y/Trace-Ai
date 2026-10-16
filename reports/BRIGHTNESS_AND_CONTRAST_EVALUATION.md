# Brightness and Contrast Evaluation — TRACE AI Forensic Audit

**Transformations Audited**:
- Brightness Increase (`apply_brightness`, factor: 1.15, 1.50)
- Brightness Decrease (`apply_brightness`, factor: 0.85, 0.60)
- Contrast Increase (`apply_contrast`, factor: 1.20, 1.60)
- Contrast Decrease (`apply_contrast`, factor: 0.80, 0.55)
**Source Code Reference**: [`src/transforms.py:179-202`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L179-L202)  

---

## 1. Executive Summary

Brightness and contrast adjustments are photometric, non-geometric transformations that preserve canvas dimensions $(W, H)$ while modifying the pixel intensity distribution:
$$I_{\text{brightness}}(x, y) = \text{clip}(\alpha \cdot I(x, y), 0, 255)$$
$$I_{\text{contrast}}(x, y) = \text{clip}(\mu + \beta \cdot (I(x, y) - \mu), 0, 255)$$

### Benchmark Findings:
- **Brightness**:
  - Relationship Recall: **91.7%** (11/12 pairs detected as related; 1 extreme darkening pair rejected)
  - Direction Accuracy: **75.0%** (9/12 correct)
  - Transformation Top-1 Accuracy: **25.0%** (3/12 correct; 6 predicted as `perspective`, 1 `color`, 1 `contrast`, 1 `none`)
- **Contrast**:
  - Relationship Recall: **91.7%** (11/12 pairs detected as related)
  - Direction Accuracy: **58.3%** (7/12 correct)
  - Transformation Top-1 Accuracy: **8.3%** (1/12 correct; 5 predicted as `perspective`, 2 `color`, 2 `brightness`, 1 `sharpen`, 1 `none`)

---

## 2. Inherent Causal Ambiguity in Photometric Provenance

Unlike cropping or downscaling—where high-frequency information or spatial extent is irreversibly destroyed—linear photometric adjustments are theoretically invertible over unclipped dynamic ranges:
$$I = \frac{1}{\alpha} I_{\text{brightness}}$$

A forensic image analyst cannot reliably determine whether a darker image was brightened or a brighter image was darkened unless:
1. **Highlight / Shadow Clipping Occurs**: Severe brightening pushes values to 255 (loss of texture in highlights); severe darkening crushes shadows to 0.
2. **Quantization Comb Artifacts**: Expanding contrast stretches discrete integer bins, leaving empty gaps ("combs") in the 256-bin histogram.

When mild adjustments are applied (e.g. factor 1.15 or 0.85):
- No clipping occurs.
- The direction model's confidence hovers near threshold ($0.52 - 0.65$), leading to occasional direction flips (e.g. on smooth gradients where entropy changes are minimal).

---

## 3. Transformation Classification Defect: The "Perspective" Confusion

A notable defect discovered in `transformation_classifier_top20.pkl` is that **50% of brightness tests (6/12) and 42% of contrast tests (5/12) were classified as `perspective`**.

### Technical Root Cause:
The 20 selected features in `transformation_classifier_top20.pkl` include 7 grayscale and color histogram bins:
- Feature 17: `abs_diff[gray_hist[0]]`
- Feature 23: `abs_diff[gray_hist[6]]`
- Feature 24: `abs_diff[gray_hist[7]]`
- Feature 28: `abs_diff[gray_hist[11]]`
- Feature 33: `abs_diff[r_hist[0]]`
- Feature 53: `abs_diff[b_hist[4]]`

When perspective warping is applied in `src/transforms.py`, black triangular borders are introduced if the warped quadrilateral does not fill the rectangular frame, shifting the low-intensity histogram bins (`gray_hist[0]`). Because the training set contained many perspective examples with border fill, the ExtraTrees classifier learned to associate large shifts in histogram bin 0 with `perspective` rather than `brightness` or `contrast`!

---

## 4. Empirical Evaluation Table

| Transform Type | Parameter | P(rel) | Cos Sim | Pred Direction | GT Direction | Pred Transformation | GT Transformation |
|---|---|---|---|---|---|---|---|
| Brightness Inc Mild | factor=1.15 | 0.9433 | 0.9912 | **B_TO_A (Invert)** | A_TO_B | perspective | brightness |
| Brightness Inc Strong | factor=1.50 | 0.8874 | 0.9431 | **A_TO_B** | A_TO_B | brightness | brightness |
| Brightness Dec Mild | factor=0.85 | 0.8518 | 0.9802 | **A_TO_B** | A_TO_B | perspective | brightness |
| Brightness Dec Strong | factor=0.60 | 0.7410 | 0.9125 | **A_TO_B** | A_TO_B | perspective | brightness |
| Contrast Inc Mild | factor=1.20 | 0.9512 | 0.9930 | **A_TO_B** | A_TO_B | perspective | contrast |
| Contrast Inc Strong | factor=1.60 | 0.8920 | 0.9450 | **A_TO_B** | A_TO_B | contrast | contrast |
| Contrast Dec Mild | factor=0.80 | 0.8845 | 0.9880 | **B_TO_A (Invert)** | A_TO_B | brightness | contrast |
| Contrast Dec Strong | factor=0.55 | 0.7620 | 0.9150 | **B_TO_A (Invert)** | A_TO_B | perspective | contrast |

---

## 5. Explainable AI Forensic Grounding

In response to this audit, [`src/xai_explainer.py`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/xai_explainer.py) has been upgraded:
- When `brightness` is predicted, the system reports the exact photometric luminance delta ($\Delta L = L_{\text{child}} - L_{\text{parent}}$).
- When `contrast` is predicted, the system reports standard deviation expansion/compression ($\Delta \sigma = \sigma_{\text{child}} - \sigma_{\text{parent}}$).
- Ordinary pixel differences are explicitly marked as local perceptual shifts rather than causal proofs.
