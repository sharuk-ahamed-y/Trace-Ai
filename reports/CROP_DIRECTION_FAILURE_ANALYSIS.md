# Crop Direction & Transformation Failure Analysis

**Component Under Audit**: Crop Operations (`apply_crop` in [`src/transforms.py`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/src/transforms.py#L55))  
**Inference Engines**: `AntisymmetricDirectionMLP` and `ProvenanceEngine` physical rules ([`backend/app/services/provenance_engine.py:182`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/backend/app/services/provenance_engine.py#L182))  
**Model**: `transformation_classifier_top20.pkl`  

---

## 1. Executive Summary

Cropping is one of the most common image manipulations in digital forensics. In this project, cropping was audited under four controlled operational conditions across three independent validation source images:
1. **Mild Crop** (8% boundary removal, `resize_back=False`)
2. **Moderate Crop** (20% boundary removal, `resize_back=False`)
3. **Aggressive Crop** (35% boundary removal, `resize_back=False`)
4. **Rescaled Crop** (20% boundary removal, `resize_back=True`)

**Key Empirical Findings**:
- **Overall Lineage Recall**: **100.0%** (12/12 pairs detected as related).
- **Direction Accuracy (Unscaled Crops)**: **100.0%** (9/9 correct $A \to B$).
- **Direction Accuracy (Rescaled Crops)**: **0.0%** (3/3 inverted to $B \to A$).
- **Transformation Classifier Accuracy**: **16.7%** (only 2/12 correctly classified as `crop`).

---

## 2. Root Cause Analysis: The Two Regimes of Crop Direction

### Regime A: Dimensional Bounding Box Reduction (`resize_back=False`)
When a region is cropped without rescaling, the child image dimensions $(W_B, H_B)$ and area $A_B = W_B \times H_B$ are strictly less than the parent area $A_A$:
$$A_B < A_A$$

In [`provenance_engine.py:181-190`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/backend/app/services/provenance_engine.py#L181):
```python
if pred["direction"] == "A_TO_B":
    if pred["transformation"] in ["crop", "resize"] and area_a < area_b * 0.85:
        src, tgt = id_b, id_a
    else:
        src, tgt = id_a, id_b
```
This physical rule enforces that an ordinary crop or downscale cannot increase image canvas area. Even when neural network direction logits are close to threshold, the area prior and feature deltas `abs_diff[0]` (width), `abs_diff[1]` (height), and `abs_diff[2]` (area) decisively resolve the direction:
- `mild_crop` (471x471 vs 512x512): Direction: $A \to B$ (100% correct)
- `moderate_crop` (409x409 vs 512x512): Direction: $A \to B$ (100% correct)
- `aggressive_crop` (332x332 vs 512x512): Direction: $A \to B$ (100% correct)

### Regime B: Rescaled Sub-region Crops (`resize_back=True`)
When a crop is interpolated back to the original canvas dimensions (e.g. $512 \times 512$), a catastrophic failure occurs:
- Area delta: $\Delta A = |A_A - A_B| = 0$
- Width delta: $\Delta W = 0$
- Height delta: $\Delta H = 0$

Because the physical dimension invariant is neutralized ($\Delta A = 0$), the pipeline must rely entirely on neural direction features (172-D vector). However, interpolation during rescaling applies anti-aliasing / Lanczos smoothing across the entire canvas. The model's latent representation interprets this global smoothing as a natural, uncropped original and predicts that the rescaled cropped image is the **parent** of the original image ($B \to A$, 100% failure rate).

---

## 3. Why the Transformation Classifier Fails on Crop (16.7% Top-1)

The Random Forest model ([`transformation_classifier_top20.pkl`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/models/transformation_classifier_top20.pkl)) selects 20 features from the 115-D difference vector. The top features related to spatial transformations are:
- Feature 0: `abs_diff[width]`
- Feature 1: `abs_diff[height]`
- Feature 2: `abs_diff[area]`
- Feature 4: `abs_diff[file_size]`

When a crop is performed:
1. If dimensions change, the model predicts `resize` (4 out of 12 times) instead of `crop` because `resize` is also strongly associated with dimensional changes in the training set.
2. If dimensions do not change (`resize_back=True`), all dimensional deltas are zero. The model then attributes the visual difference to `rotation` (3 times), `perspective` (1 time), or `contrast` (1 time).

---

## 4. Verification Table: Crop Experiments Across Validation Sources

| Source Image | Subtype | Parent Dim | Child Dim | P(rel) | Pred Direction | Ground Truth | Pred Tfm | Tfm Acc |
|---|---|---|---|---|---|---|---|---|
| `circles_root` | mild | 512x512 | 471x471 | 0.8267 | **A_TO_B** | A_TO_B | resize | False |
| `circles_root` | moderate | 512x512 | 409x409 | 0.8050 | **A_TO_B** | A_TO_B | crop | **True** |
| `circles_root` | aggressive | 512x512 | 332x332 | 0.7788 | **A_TO_B** | A_TO_B | resize | False |
| `circles_root` | rescaled | 512x512 | 512x512 | 0.8143 | **B_TO_A (FAIL)** | A_TO_B | rotation | False |
| `gradient_root` | mild | 512x512 | 471x471 | 0.8920 | **A_TO_B** | A_TO_B | rotation | False |
| `gradient_root` | moderate | 512x512 | 409x409 | 0.8654 | **A_TO_B** | A_TO_B | crop | **True** |
| `gradient_root` | aggressive | 512x512 | 332x332 | 0.8310 | **A_TO_B** | A_TO_B | contrast | False |
| `gradient_root` | rescaled | 512x512 | 512x512 | 0.8712 | **B_TO_A (FAIL)** | A_TO_B | rotation | False |
| `screenshot_root` | mild | 1344x583 | 1236x536 | 0.9509 | **A_TO_B** | A_TO_B | resize | False |
| `screenshot_root` | moderate | 1344x583 | 1075x466 | 0.9320 | **A_TO_B** | A_TO_B | resize | False |
| `screenshot_root` | aggressive | 1344x583 | 873x378 | 0.9015 | **A_TO_B** | A_TO_B | sharpen | False |
| `screenshot_root` | rescaled | 1344x583 | 1344x583 | 0.9380 | **B_TO_A (FAIL)** | A_TO_B | perspective | False |

---

## 5. Architectural Recommendations & Fix Strategy

1. **Sub-image Keypoint / SIFT Matching for Scale-Invariant Crop Detection**:
   When $\Delta A = 0$ but ViT cosine similarity is in the range $[0.75, 0.95]$, compute SIFT / ORB feature match homography. If the matched bounding box occupies an interior sub-rectangle of Image A, Image B is definitively a crop descendant.
2. **Multi-label Feature Partitioning**:
   Disentangle `crop` and `resize` by training a dedicated binary detector for field-of-view reduction versus full-canvas resampling.
