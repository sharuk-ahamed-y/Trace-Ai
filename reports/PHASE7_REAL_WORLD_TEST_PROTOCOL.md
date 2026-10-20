# PHASE 7: REAL-WORLD TEST & EVALUATION PROTOCOL

**TRACE AI: Image Provenance Using Graphs**  
**Protocol Version:** 1.0  
**Status:** **PREPARED BUT NOT EXECUTED** (Awaiting authorized real-world image contribution)  
**Security & Privacy:** High (Zero unapproved images, zero Git leaks)

---

## 1. Overview & Objective

This document defines the strict, reproducible experimental protocol for evaluating TRACE AI's image provenance pipeline on real-world imagery (smartphone cameras, messaging platforms, social media re-encodings, displays, and physical document prints).

Because Phase 4 and Phase 6 verified performance strictly on synthetic procedural imagery (`SIPGD_v2`), this protocol bridges the synthetic-to-real domain gap without compromising scientific integrity or leaking private user data.

---

## 2. Image Collection, Consent & Privacy Governance

1. **Explicit Permission & License:**
   - Only images with explicit written consent or open permissive licenses (e.g., CC0, CC-BY 4.0, or contributor authorization) may be added.
   - Images must not depict personally identifiable information (PII), faces of non-consenting individuals, sensitive documents, credentials, or private property without redaction.
2. **Git Exposure Prevention:**
   - Real-world images must reside exclusively in local, gitignored directories: `data/real_world_benchmark/`.
   - Ensure `.gitignore` explicitly excludes:
     ```gitignore
     data/real_world_benchmark/
     data/real_world_benchmark/**
     *.jpg
     *.jpeg
     *.png
     *.webp
     ```
   - Never commit raw photographs, uncompressed sensor captures, or personal device images to version control.

---

## 3. Source ID Assignment & Split Leakage Prevention

1. **Independent Source IDs:**
   - Every physical scene/original photograph must be assigned a unique immutable identifier: `real_src_0001`, `real_src_0002`, ..., `real_src_NNNN`.
2. **Derivative Grouping:**
   - Every derivative (crop, WhatsApp compression, screenshot, contrast tweak) must preserve its parent reference and root source ID: `real_src_0001_v01_whatsapp`, `real_src_0001_v02_crop`.
3. **Strict Grouped Partitioning:**
   - **All derivatives belonging to `real_src_XXXX` must reside in the exact same evaluation split.**
   - Under no circumstances may an original appear in Split A while its crop or compressed version appears in Split B.
   - An automated script must assert:
     $$\text{Sources}(\text{Split}_A) \cap \text{Sources}(\text{Split}_B) = \emptyset$$

---

## 4. Pair Construction Strategy

### 4.1 Positive (Related) Pairs
Positive pairs are constructed by pairing related versions originating from the same physical original:
1. **Camera Original $\to$ Messaging App (e.g., WhatsApp, Telegram, Signal):**
   - Transmitted through standard compression channels (WebP/JPEG, downsampling to 1600px, EXIF metadata stripped).
2. **Camera Original $\to$ Crop / Sub-region:**
   - Aspect ratio alterations, zoom-in, rule-of-thirds cuts.
3. **Camera Original $\to$ Photographed Screen / Physical Print:**
   - Moiré pattern, optical lens distortion, ambient illumination.
4. **Multi-generation / Composite Edits:**
   - Crop + WhatsApp compression + brightness adjustment (multi-hop lineage).

### 4.2 Realistic Negative Pairs
Negative pairs must represent challenging, realistic distractors rather than trivially distinguishable pairs:
1. **Semantic Similarity Negatives:**
   - Photographs of the same category (e.g., two distinct red sports cars, two distinct city skylines under sunset).
2. **Same Device / Sensor Negatives:**
   - Photographs captured on the same smartphone sensor under similar lighting, but depicting distinct physical objects (tests sensor noise PRNU resilience vs. content matching).
3. **Near-Duplicate Scenes:**
   - Two photos taken seconds apart from slightly different viewpoints (tests instance-level provenance vs. general scene classification).

---

## 5. Ground Truth Annotation (Independent of Model Outputs)

1. **No Circular Validation:**
   - The model's own predictions must **never** be used to label transformations or determine relationship ground truth.
2. **Metadata Manifest:**
   - Ground truth must be recorded at time of creation in a structured CSV (`real_pairs_manifest.csv`) with the following fields:
     - `pair_id`: UUID
     - `image_a`: Filename of image A
     - `image_b`: Filename of image B
     - `source_a`: Root source ID of A
     - `source_b`: Root source ID of B
     - `related`: Binary (1 if $source_A == source_B$, else 0)
     - `ground_truth_direction`: `A_TO_B`, `B_TO_A`, or `UNCERTAIN` (for identical copies)
     - `applied_transformations`: List of verified physical actions (e.g., `["crop", "whatsapp_compression"]`)
     - `generation_delta`: Exact generational difference ($0$ for siblings, $1$ for parent-child, $2$ for grandparent-grandchild)

---

## 6. Evaluation Metrics & Protocol

When real-world pairs become available, evaluation must be computed as follows:

1. **Pairwise Relationship Metrics:**
   - $\text{Precision} = \frac{TP}{TP + FP}$
   - $\text{Recall} = \frac{TP}{TP + FN}$
   - $\text{F1-Score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
   - $\text{False Positive Rate (FPR)} = \frac{FP}{FP + TN}$
   - Area under ROC curve ($\text{ROC-AUC}$) and PR curve ($\text{PR-AUC}$).
2. **Directional Accuracy & Abstention Coverage:**
   - Direction evaluated strictly on positive pairs ($y_{\text{rel}} = 1$).
   - **Abstention Coverage:** Fraction of pairs where $|P(A \to B) - 0.5| \ge \Delta_{\text{uncertain}}$ (where $\Delta = 0.04$).
   - **Direction Accuracy:** Accuracy on confident predictions.
   - **Order Swap Consistency:** $100\%$ check that swapping $(A, B) \to (B, A)$ strictly inverts predicted direction.
3. **Multi-Label Transformation Performance:**
   - Macro-F1 and Micro-F1 across the 12 supported transformation classes.
   - Confusion analysis for compression vs. sharpening artifacts.
4. **Confidence Calibration:**
   - Expected Calibration Error (ECE) across 10 confidence bins:
     $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$

---

## 7. Strict Separation of Benchmarks

- Real-world metrics must be reported in a dedicated section or distinct file (`reports/REAL_WORLD_EVALUATION.md`).
- Real-world scores must **never** be averaged into or merged with the synthetic `SIPGD_v2` test scores.
- When no real-world test set is present, all APIs and reports must declare real-world metrics as **`NOT VALIDATED`**.

---

## 8. Current Protocol Status

- **Protocol Specification:** **COMPLETE & VERIFIED**
- **Test Infrastructure Readiness:** **READY**
- **Image Execution Status:** **AWAITING CONTRIBUTED / AUTHORIZED DATASET**
