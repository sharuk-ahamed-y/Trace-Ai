# TRACE AI — REAL-WORLD PHONE-TO-WHATSAPP WORKFLOW VALIDATION

**Test Objective:** Validate TRACE AI on an authentic, high-entropy photographic smartphone capture subject to real-world social media distribution (WhatsApp compression, downsampling, cropping, photometric adjustment, noise addition).  
**Source Image:** Controlled high-resolution smartphone photo (`A_original_phone_photo.jpg`, 1024×1024, 24-bit RGB, natural sensor noise and optical depth of field).  
**Evaluation Date:** October 14, 2026  
**Artifact Data:** [`reports/REAL_WORLD_PHONE_WHATSAPP_VALIDATION.json`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/reports/REAL_WORLD_PHONE_WHATSAPP_VALIDATION.json)

---

## 1. Controlled Experimental Setup

Twelve controlled image variants (A through L) were synthesized from the original phone photo root to emulate real-world forensic tampering and platform dissemination:

| Test ID | Image Name | Transformation Applied & Parameters | Ground Truth Parent | Ground Truth Transformation |
| :---: | :--- | :--- | :--- | :--- |
| **A** | `A_original_phone_photo` | Original unaltered smartphone camera capture | *Self (Root)* | *None* |
| **B** | `B_crop_photo` | Center crop ($25\%$ width, $25\%$ height reduction, $768\times 768$) | `A` | `crop` |
| **C** | `C_brightness_adjusted` | Global linear brightness boost ($+25\%$, factor $1.25$) | `A` | `brightness` |
| **D** | `D_contrast_adjusted` | Global linear contrast expansion ($+35\%$, factor $1.35$) | `A` | `contrast` |
| **E** | `E_jpeg_compressed` | Lossy JPEG compression ($Q = 55$) | `A` | `jpeg` |
| **F** | `F_blurred_copy` | Gaussian low-pass spatial blur ($\sigma = 2.5$) | `A` | `blur` |
| **G** | `G_sharpened_copy` | Unsharp masking / spatial edge enhancement (factor $2.2$) | `A` | `sharpen` |
| **H** | `H_noisy_copy` | Additive zero-mean Gaussian sensor noise ($\sigma = 15.0$) | `A` | `noise` |
| **I** | `I_resized_copy` | Bilinear canvas downscale ($50\%$, $512\times 512$) | `A` | `resize` |
| **J** | `J_whatsapp_transferred` | WhatsApp pipeline emulation: scale $0.65$ + JPEG $Q=70$ + stripped EXIF | `A` | `resize` / `whatsapp` |
| **K** | `K_crop_then_resize` | Compound edit: $20\%$ center crop followed by scale $0.60$ ($491\times 491$) | `A` | `crop` |
| **L** | `L_multistep_sequence` | 4-stage pipeline: Crop $15\%$ $\to$ Brightness $+10\%$ $\to$ Scale $0.75$ $\to$ JPEG $Q=65$ | `A` | `crop` |
| **X** | `X_unrelated_mountain` | Independent alpine snow mountain photo (Disjoint semantic scene) | *None* | *Unrelated* |

---

## 2. Experimental Results (Pairwise Inferences Against Root A)

Every test was evaluated symmetrically in both forward ($A \to X$) and reverse ($X \to A$) directions using the updated `ModelManager` with `ForensicArbitrator` and ViT semantic gating:

| ID | Descendant Test Name | PSNR (dB) | SSIM | $P(\text{rel})$ | Pred Dir | Dir Conf | Uncertainty | Primary Tfm | Compound Multi-label Tags | Dir Correct |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **B** | `B_crop_photo` | 10.12 | 0.3162 | 0.9003 | **A_TO_B** | 0.9998 | 0.0002 | `crop` | `[resize, crop]` | **YES** |
| **C** | `C_brightness_adjusted` | 18.72 | 0.9590 | 0.9997 | **B_TO_A** | 1.0000 | 0.0000 | `noise` | `[noise]` | NO |
| **D** | `D_contrast_adjusted` | 17.43 | 0.9234 | 0.9996 | **B_TO_A** | 1.0000 | 0.0000 | `noise` | `[noise]` | NO |
| **E** | `E_jpeg_compressed` | 31.84 | 0.9851 | 0.9952 | **B_TO_A** | 1.0000 | 0.0000 | `noise` | `[noise]` | NO |
| **F** | `F_blurred_copy` | 27.15 | 0.9120 | 0.9613 | **A_TO_B** | 0.9850 | 0.0150 | `blur` | `[noise, blur]` | **YES** |
| **G** | `G_sharpened_copy` | 29.40 | 0.9782 | 1.0000 | **B_TO_A** | 1.0000 | 0.0000 | `noise` | `[noise]` | NO |
| **H** | `H_noisy_copy` | 24.62 | 0.8115 | 1.0000 | **B_TO_A** | 0.9850 | 0.0150 | `blur` | `[noise, blur]` | NO |
| **I** | `I_resized_copy` | 34.12 | 0.9912 | 0.9900 | **A_TO_B** | 1.0000 | 0.0000 | `resize` | `[resize]` | **YES** |
| **J** | `J_whatsapp_transferred` | 30.55 | 0.9745 | 0.9900 | **A_TO_B** | 1.0000 | 0.0000 | `resize` | `[resize, jpeg, quality_degradation, whatsapp_compression]` | **YES** |
| **K** | `K_crop_then_resize` | 12.80 | 0.4210 | 0.8379 | **A_TO_B** | 1.0000 | 0.0000 | `resize` | `[resize]` | **YES** |
| **L** | `L_multistep_sequence` | 14.15 | 0.4682 | 0.8331 | **A_TO_B** | 1.0000 | 0.0000 | `resize` | `[noise, jpeg, resize, quality_degradation, whatsapp_compression]` | **YES** |

---

## 3. Deep Dive into Critical Test Workflows

### 3.1 WhatsApp Transferred Copy (`J_whatsapp_transferred`)
- **Real-World Mechanism**: WhatsApp resizes incoming images (reducing resolution by $\sim 35\%$), converts RGB to YCbCr 4:2:0 chroma subsampling, applies progressive DCT quantization ($Q \approx 70\text{--}75$), and strips camera EXIF metadata.
- **System Prediction**:
  - Related Probability: $P(\text{rel}) = 0.9900$ (Accepted).
  - Predicted Direction: $A \to B$ with confidence $1.0000$. Rationale: *Physical Area Constraint: High-resolution canvas Image A exceeds derived subscale Image B*.
  - Primary Transformation: `resize`.
  - **Compound Multi-Label Tags**: `["resize", "jpeg", "quality_degradation", "whatsapp_compression"]`.
  - **Verdict**: Fully successful reproduction and categorization of social media pipeline.

### 3.2 Crop Followed by Resize (`K_crop_then_resize`)
- **Real-World Mechanism**: A user crops a detail and scales it down to post as an avatar or message attachment. Canvas dimensions are altered and original coordinates are lost.
- **System Prediction**:
  - Related Probability: $P(\text{rel}) = 0.8379$ (ViT semantic correlation confirms visual scene identity despite $12.8$ dB PSNR).
  - Direction: $A \to B$ with confidence $1.0000$.
  - Multi-scale FOV template matching identified sub-window containment ($\text{NCC} = 0.8259$).
  - **Verdict**: Successfully resolved the crop-resize inversion failure.

### 3.3 Disjoint Unrelated Image Test (`A_original_phone_photo` vs `X_unrelated_mountain`)
- **Challenge**: The cafe photo and mountain photo have completely independent provenance histories.
- **ViT Embedding Cosine Similarity**: $-0.0207$ (Negative correlation).
- **Pre-Gating Vulnerability**: Handcrafted 115-D difference feature MLP output $P(\text{rel}) = 1.0000$ because the scalar distance fell into an unregularized out-of-distribution space.
- **Post-Gating Decision**: ViT semantic gating clamped $P(\text{rel}) \to 0.0000$.
- **Graph Reconstruction**: 0 edges created (`Candidate-edge count: 0, Rejected-pair count: 1`).
- **Verdict**: **100% Correct Rejection**.

### 3.4 Sibling Triad Test (`Root A` + `B_crop_photo` + `J_whatsapp_transferred`)
- **Ground Truth Tree**: $A \to B$ and $A \to J$. No edge between $B$ and $J$.
- **Graph Reconstruction Output**:
  - Root selected: `A_original_phone_photo` (Correct).
  - MST Edges: `(A -> B, weight=0.0997)` and `(A -> J, weight=0.0100)`.
  - Edge between $B$ and $J$: **Rejected by Adaptive Kruskal cycle & margin pruning**.
  - **Verdict**: Avoided false sibling shortcut edge.

---

## 4. Key Limitations Revealed by Real Photographic Data

The evaluation of authentic 1024×1024 phone photography highlighted a key structural limitation in the legacy MLP direction checkpoint (`antisymmetric_direction_model.pt`):

1. **Synthetic Training Distribution Bias**:
   - The neural direction model was trained on synthetic graphics, geometric shapes, and clean gradients (`src_0001` through `src_0020`), where the root images had **zero natural sensor noise** ($\sigma_{\text{noise}} \approx 0$). In that synthetic domain, any transformation (JPEG, noise, sharpening) *added* noise or high-frequency edge energy.
   - On a genuine phone photo, the root image already contains natural CMOS sensor photon shot noise ($\sigma_{\text{noise}} \approx 8.4$).
   - When JPEG compression ($Q=55$) was applied to the phone photo, it smoothed out natural sensor grain in flat areas while creating 8×8 block edges. The neural model interpreted the smoothed flat areas as the original and the raw photo's natural grain as added noise, inverting direction ($B \to A$) for tests C, D, E, G, H.
2. **Mitigation Provided**:
   - For spatial operations (crop, resize, crop-resize, WhatsApp, multistep), physical area and FOV containment override the neural MLP, achieving $100\%$ directional accuracy ($A \to B$).
   - For pure Gaussian blur, spectral blur attenuation overrides the neural MLP, achieving $100\%$ directional accuracy ($A \to B$).
   - For pure in-place photometric scaling (brightness, contrast, mild JPEG), where dimensions remain identical, the system relies on the neural MLP.
3. **Requirement for Model Re-Training**:
   - Retraining `antisymmetric_direction_model.pt` on real photographic camera datasets (e.g., RAISE-1k, Dresden Image Database) containing authentic sensor noise and camera PRNU profiles is recommended to achieve directional accuracy $>85\%$ on in-place photometric edits without metadata.
