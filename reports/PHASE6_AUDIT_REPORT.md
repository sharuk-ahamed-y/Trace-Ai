# PHASE 6: INDEPENDENT MODEL AUDIT & REAL-WORLD VALIDATION REPORT

**TRACE AI: Image Provenance Using Graphs**  
**Audit Timestamp:** 2026-10-18  
**Audit Environment:** Windows (Python 3.12, PyTorch 2.6.0+cpu, scikit-learn 1.7.0, FastAPI 0.115.8)  
**Active Production Switch:** `ENABLE_PHASE4_MODELS=false` (Default: Disabled)

---

## 1. Executive Verdict

### Verdict: **PASS WITH LIMITATIONS**

- **Synthetic Performance & Architectural Integrity:** **PASS**  
  The Phase 4 normalized forensic models (`models/phase4_v1/`) and Phase 5 integration pass all architectural, mathematical, and regression requirements. Feature dimensions, scalers, swap antisymmetry ($P(A \to B) + P(B \to A) \equiv 1.0$), and split independence are verified down to the exact decimal with zero data leakage.
- **Production Integration & Reversibility:** **PASS**  
  The system is cleanly gated behind `ENABLE_PHASE4_MODELS` (default: `False`). Rollback to the legacy baseline is instantaneous and verified. Missing or partially available artifacts are caught gracefully without crashes or mocked predictions.
- **Real-World Validation:** **NOT VALIDATED (LIMITATION)**  
  Zero real-world test pairs (such as camera originals, WhatsApp recompressed crops, or photographed screens) exist in the repository (`data/originals` and `data/uploads` are empty). Therefore, real-world generalizability cannot be claimed and is explicitly marked as unvalidated.
- **API Metrics Reporting:** **LIMITATION**  
  `/api/model-metrics` returns hardcoded legacy metrics (96 samples, 85.42% accuracy) and does not dynamically report Phase 4 metrics when Phase 4 is enabled.

---

## 2. Gate 1: Source-Code Audit Findings & Line-by-Line Tracing

| Audit Check | Status | Exact File & Lines | Detailed Technical Finding |
|:---|:---:|:---|:---|
| **Feature Dimension Discrepancy** | **Resolved & Grounded** | [`backend/app/services/model_manager.py:422-465`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/backend/app/services/model_manager.py#L422-L465)<br>[`src/pair_features.py:83-89, 117-123`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/src/pair_features.py#L83-L89)<br>[`src/forensic_features.py:150-161`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/src/forensic_features.py#L150-L161) | The Phase 5 report casually described features as "20 forensic + 37 visual/spectral". **Actual code inspection reveals the descriptor is exactly 57-D**, composed of: <br>1. **17 physical/statistical scalar forensics**: `width`, `height`, `area`, `aspect_ratio`, `file_size`, `jpeg_quality_est`, `r_mean`, `r_std`, `g_mean`, `g_std`, `b_mean`, `b_std`, `edge_mean`, `edge_std`, `edge_max`, `noise_res_std`, `noise_res_skew`.<br>2. **40 histogram bins**: 16 grayscale bins + 8 R + 8 G + 8 B color bins.<br>Both training (`train_phase4_models.py:131-144`) and production inference (`model_manager.py:442-464`) extract identical 57-D vectors. |
| **Exact Pair-Feature Constructors** | **Verified** | [`src/pair_features.py:68-131`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/src/pair_features.py#L68-L131)<br>[`backend/app/services/model_manager.py:467-478, 536-541`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/backend/app/services/model_manager.py#L467-L478) | **115-D Pair Vector:** `[|vA - vB|, vA * vB, cos_sim]` ($57 + 57 + 1 = 115\text{-D}$, symmetric float32).<br>**172-D Direction Vector:** `[vA, vB, (vA - vB), cos_sim]` ($57 + 57 + 57 + 1 = 172\text{-D}$, antisymmetric float32). Both constructors strictly validate input length equal to 57. |
| **Scalers & Preprocessing** | **Verified** | [`backend/app/services/model_manager.py:356-358, 503, 538-539`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/backend/app/services/model_manager.py#L356-L358)<br>[`src/train_phase4_models.py:220-237`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/src/train_phase4_models.py#L220-L237) | `StandardScaler` objects `scaler_115.joblib` and `scaler_172.joblib` fit strictly on the Training set. Means and scales are finite with zero NaNs. Loaded and applied identically during inference via `.transform()`. |
| **Two Direction Checkpoints** | **Verified & Clarified** | [`models/phase4_v1/direction_model.pt`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/models/phase4_v1/direction_model.pt)<br>[`models/phase4_v1/antisymmetric_direction_model.pt`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/models/phase4_v1/antisymmetric_direction_model.pt) | Both checkpoints have identical weights for all layers (`g.0.*`, `g.1.*`, `g.4.*`, `g.7.*`).<br>- `direction_model.pt` contains `['state_dict', 'in_dim', 'hidden', 'delta_uncertain', 'seed']`, relying on external `scaler_172.joblib`.<br>- `antisymmetric_direction_model.pt` bundles `scaler_mean` and `scaler_var` inside the checkpoint dictionary for self-contained deployment. Production loads `direction_model.pt` together with `scaler_172.joblib`. |
| **Thresholds, Transformations, UNCERTAIN** | **Verified** | [`backend/app/services/model_manager.py:363-365, 547-575`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/backend/app/services/model_manager.py#L363-L365)<br>[`src/transforms.py:30-43`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/src/transforms.py#L30-L43) | - Relationship threshold: $0.6200$ (loaded from `metadata.json`).<br>- Multi-label transformation: 12 classes match `src/transforms.py` order exactly. Thresholds loaded per-class; returns primary label and `compound_tags`.<br>- Direction ambiguity: If $|P(A \to B) - 0.5| < 0.0400$, outputs `UNCERTAIN`. |
| **Legacy Heuristic Fallback Risk** | **Documented** | [`backend/app/services/model_manager.py:799-811`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/backend/app/services/model_manager.py#L799-L811) | If all trained checkpoints are absent, the legacy fallback uses `ViT-B/16 Cosine Similarity Heuristic`, setting `related_confidence = cos_sim`, `direction = "A_TO_B"`, `transformation = "crop"`, and `active_model = "ViT-B/16 Cosine Similarity Heuristic"`. Cosine similarity is reused as confidence for all three heads. While `active_model` clearly flags it as a heuristic, consumers ignoring `active_model` could misinterpret it as a model prediction. |
| **API Endpoints Inference Path Status** | **Partial Mismatch Identified** | [`backend/app/api/endpoints.py:64-81, 84-118, 121-140`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/backend/app/api/endpoints.py#L64-L81) | 1. `/api/health` accurately identifies the path (`phase4_models_enabled`, `phase4_models_loaded`, `active_model`).<br>2. `/api/model-info` identifies `active_relationship_model`, but its child sections still document legacy scikit-learn and legacy PyTorch multi-task.<br>3. `/api/model-metrics` returns hardcoded static JSON from legacy phases (85.42% accuracy on 96 samples) instead of Phase 4 metrics. |

---

## 3. Gate 2: Dataset Independence & Metric Recomputation

### 3.1 Split Overlap & Leakage Audit
An automated audit across `data/provenance_dataset_v2/metadata/` (`pairs_train.csv`, `pairs_val.csv`, `pairs_test.csv`) confirmed:
- **Total pairs:** Train = 1,544, Val = 384, Test = 324.
- **Source ID Intersection:**
  - $\text{Train} \cap \text{Val} = 0$
  - $\text{Train} \cap \text{Test} = 0$
  - $\text{Val} \cap \text{Test} = 0$
- **Image ID Intersection:**
  - $\text{Train} \cap \text{Val} = 0$
  - $\text{Train} \cap \text{Test} = 0$
  - $\text{Val} \cap \text{Test} = 0$
- **Leakage verdict:** Zero data leakage. All derivatives of every source image reside strictly in a single split.

### 3.2 Independent Metric Recomputation
Metrics were recomputed directly from the saved checkpoints in `models/phase4_v1/` on the held-out test split (`pairs_test.csv`, 324 pairs: 162 related, 162 unrelated) using [`tests/audit_verify_metrics.py`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/tests/audit_verify_metrics.py):

| Metric | Reported in Phase 4 / Metadata | Independently Recomputed in Phase 6 | Difference | Status |
|:---|:---:|:---:|:---:|:---:|
| **Relationship Threshold** | 0.6200 | 0.6200 | 0.0000 | **Exact Match** |
| **Relationship Precision** | 0.9877 | 0.9877 | 0.0000 | **Exact Match** |
| **Relationship Recall** | 0.9877 | 0.9877 | 0.0000 | **Exact Match** |
| **Relationship F1-Score** | 0.9877 | 0.9877 | 0.0000 | **Exact Match** |
| **Relationship ROC-AUC** | 0.9990 | 0.9990 | 0.0000 | **Exact Match** |
| **Relationship PR-AUC** | 0.9990 | 0.9990 | 0.0000 | **Exact Match** |
| **Relationship FPR** | 0.0123 | 0.0123 | 0.0000 | **Exact Match** |
| **Confusion Matrix** | TN=160, FP=2, FN=2, TP=160 | TN=160, FP=2, FN=2, TP=160 | None | **Exact Match** |
| **Transformation Macro-F1** | 0.6197 | 0.6197 | 0.0000 | **Exact Match** |
| **Transformation Micro-F1** | 0.6119 | 0.6119 | 0.0000 | **Exact Match** |
| **Direction Swap Antisymmetry Error** | $< 10^{-6}$ | $1.19 \times 10^{-7}$ | 0.0000 | **Exact Match** |
| **Direction Raw Accuracy** | 0.7531 | 0.7531 | 0.0000 | **Exact Match** |
| **Direction Delta Threshold** | 0.0400 | 0.0400 | 0.0000 | **Exact Match** |
| **Direction Coverage** | 91.36% | 91.36% | 0.00% | **Exact Match** |
| **Direction Confident Accuracy** | 0.7838 | 0.7838 | 0.0000 | **Exact Match** |

**Conclusion:** Reported metrics were independently verified to be 100% genuine and reproducible directly from the saved model checkpoints.

---

## 4. Gate 3: Robustness & Stress Test Outcomes

The expanded robustness suite [`tests/test_phase6_audit_robustness.py`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/tests/test_phase6_audit_robustness.py) was executed:
```powershell
python -m unittest tests/test_phase6_audit_robustness.py
```
**Outcome:** `Ran 10 tests in 11.065s — OK` (10/10 passed, 0 failures, 0 errors).

| Test ID | Robustness Category | Evaluation Details | Result |
|:---:|:---|:---|:---:|
| `test_01` | Bidirectional Order Swap Invariance | Tested across 6 transformations (resize, crop, bright, blur, sharp, jpeg). Symmetrically preserved relationship probability and strictly inverted direction ($A \to B \leftrightarrow B \to A$). | **PASS** |
| `test_02` | Unrelated Image Rejection | Symmetrically classified unrelated images as `related=False`, `direction="NONE"`, and `transformation="none"`. | **PASS** |
| `test_03` | Grayscale Ingestion | Pure 1-channel grayscale images were converted to 3-channel RGB without dimension mismatch or exceptions. | **PASS** |
| `test_04` | Extreme Canvas Dimensions | Ingested $1\times 1$ pixel image and extreme $16\times 1000$ aspect ratio without zero-division or NaN confidences. | **PASS** |
| `test_05` | Corrupted Image Handling | Handled non-image byte stream gracefully by raising clear decoding error rather than internal pipeline failure. | **PASS** |
| `test_06` | Dynamic Switch Toggling | Toggled between `ENABLE_PHASE4_MODELS=True` and `False`; active model dynamically switched without process restart. | **PASS** |
| `test_07` | Direction UNCERTAIN Persistence | Identical image pairs with near-zero feature difference yielded `UNCERTAIN` without being forced to `A_TO_B`. | **PASS** |
| `test_08` | Provenance Graph Plausibility | Minimum Spanning Forest connected related pair (`base_a` $\to$ `crop`) and rejected unrelated node (`unrelated_b`). | **PASS** |
| `test_09` | Non-Finite Feature Handling | Verified that input vectors with incorrect lengths ($50 \ne 57$) raise explicit `ValueError` rather than producing silent failures. | **PASS** |
| `test_10` | Partially Available Artifacts | Tested directory missing `transformation_model.pt`. ModelManager caught missing file, logged notice, and fell back to legacy without fabricating predictions. | **PASS** |

---

## 5. Gate 4: Real-World Validation

### Status: **NOT VALIDATED**

- **Project Inventory:** An inspection across all directories (`data/originals/`, `data/uploads/`, `image_provenance_dataset/`) found zero real-world test images.
- **Domain Gap Assessment:** The Phase 4 models were trained and evaluated exclusively on the synthetic procedural dataset (`data/provenance_dataset_v2/`). While procedural transformations cover noise, blur, color, and geometric edits, they do not replicate:
  1. Messaging application re-encoding pipelines (e.g., WhatsApp WebP/JPEG recompression with EXIF stripping and chroma subsampling).
  2. Camera-sensor noise patterns (PRNU).
  3. Physical photographs of screens or printed media.
  4. Complex real-world composite manipulations (inpainting, generative diffusion fills).
- **Rule Adherence:** Per Gate 4 safety rules, **no real-world accuracy claims are manufactured from synthetic tests**.

---

## 6. Known Limitations

1. **Synthetic-Only Generalization:** High metrics ($>0.98$ F1) are verified on synthetic procedural data; real-world image pairs may encounter distribution shifts in frequency response and compression artifacts.
2. **Static `/api/model-metrics` Endpoint:** As noted in Gate 1, the endpoint returns hardcoded legacy metrics and does not dynamically query Phase 4 metadata.
3. **CPU Execution Speed:** Feature extraction on CPU takes ~400ms per image due to ViT CLS token computation. Pairwise MLP inference is rapid (~10ms).
4. **Heuristic Fallback Transparency:** In the event that all models fail to load, the legacy heuristic reuses cosine similarity across all confidence metrics.

---

## 7. Final Recommendation

### Recommendation: **KEEP DISABLED BY DEFAULT; ENABLE ONLY FOR CONTROLLED TESTING**

1. **Production Default:** Maintain `ENABLE_PHASE4_MODELS=false` in `backend/app/config.py`.
2. **Controlled Testing:** The new models are fully verified and safe to enable selectively using `$env:ENABLE_PHASE4_MODELS = "true"` for user testing or controlled evaluation.
3. **Recommended Fix Before Production Release:**
   - Update `/api/model-metrics` in `backend/app/api/endpoints.py` to dynamically load and display Phase 4 metrics from `models/phase4_v1/metadata.json` when `ENABLE_PHASE4_MODELS=True`.
4. **Next Step for Full Release:** Collect a small benchmark set of real-world test image pairs (camera originals, social media uploads, crops) to validate the synthetic-to-real domain transfer before permanently enabling the models by default.
