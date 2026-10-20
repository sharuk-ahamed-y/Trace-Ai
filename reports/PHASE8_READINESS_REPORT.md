# PHASE 8: CONTROLLED REAL-WORLD EVALUATION READINESS REPORT

**TRACE AI: Image Provenance Using Graphs**  
**Execution Timestamp:** 2026-10-18  
**Environment:** Windows (Python 3.12, PyTorch 2.6.0+cpu, scikit-learn 1.7.0, FastAPI 0.115.8)  
**Safety Status:** `ENABLE_PHASE4_MODELS=false` (Default: Disabled)  
**Real-World Benchmark Status:** **PREPARED BUT NOT EXECUTED**

---

## 1. Executive Summary

Phase 8 performed an independent, controlled-evaluation readiness review across TRACE AI's model architecture, metadata reporting, probability calibration semantics, directional metrics demarcation, checkpoint compatibility, and evaluation procedures.

### Key Conclusions:
1. **Benchmark Execution Status:** Declared strictly as **PREPARED BUT NOT EXECUTED**. No authorized real-world image pairs exist in the repository (`data/originals/` and `data/uploads/` contain 0 benchmark images). No real-world accuracy is claimed, hallucinated, or fabricated.
2. **Probability Calibration Semantics Clarified:** Prior labeling of model confidence as `"calibrated_mlp"` was technically inaccurate because no probability distribution calibration (e.g., Platt scaling, isotonic regression, temperature scaling) was fitted or evaluated. Model confidence outputs are now technically justified and explicitly reported as `"uncalibrated_sigmoid"` with `"decision_threshold_status": "validation_tuned"`.
3. **Direction Metrics Demarcated:** Direction Raw Accuracy ($75.31\%$), Confident-Prediction Accuracy ($78.38\%$), Abstention Coverage ($91.36\%$), and Order Swap Anti-Symmetry Error ($1.19 \times 10^{-7}$) are distinctly reported in API schemas, metadata, and evaluation code without conflation.
4. **Checkpoint-Metadata Compatibility Safeguard:** Hardened `_init_phase4_models()` in `backend/app/services/model_manager.py` to assert parameter agreement (version, feature dimensions, optimal threshold, uncertainty delta) between `.pt` checkpoint weights and `metadata.json`. Incompatible or stale metadata is actively rejected, triggering safe fallback with `status: "METRICS_UNAVAILABLE"`.
5. **Evaluation Workflow Isolation:** Implemented `src/evaluate_real_world.py`, providing a leak-free, source-grouped evaluation pipeline that outputs strictly to `reports/REAL_WORLD_EVALUATION.md`, preserving complete separation from synthetic held-out metrics (`SIPGD_v2`).
6. **Regression Verification:** Full 36-test suite across Phases 5, 6, 7, and 8 passed ($36/36$ passed, $0$ failed).

---

## 2. Strict Constraints Compliance Audit

| Constraint | Implementation / Verification | Status |
|:---|:---|:---:|
| **Keep `ENABLE_PHASE4_MODELS=false` by default** | Verified in `backend/app/config.py` line 56: `ENABLE_PHASE4_MODELS = os.getenv("ENABLE_PHASE4_MODELS", "false").lower() in ("true", "1", "yes")`. Asserted in `test_phase8_readiness.py`. | **PASS** |
| **Do not retrain models** | Checkpoints in `models/phase4_v1/` remain untouched. Timestamps and sha256 checksums preserved. | **PASS** |
| **Do not alter thresholds** | Relationship threshold ($\tau = 0.62$), direction delta ($\Delta = 0.04$), and per-class transformation thresholds unchanged. | **PASS** |
| **Do not alter the frontend** | Zero modifications to `frontend/` directory. REST API response schemas remain 100% backward-compatible. | **PASS** |
| **Do not deploy, commit, or push** | Local inspection and test execution only. No git commit or push commands issued. | **PASS** |
| **Do not use personal/camera images** | No personal or unapproved camera images added or processed. | **PASS** |
| **Do not claim real-world accuracy** | Real-world status remains **NOT VALIDATED** (0 samples evaluated). | **PASS** |

---

## 3. Technical Justification of Probability & Calibration Labels

### 3.1 Problem Identified in Phase 7
In Phase 7, `predict_pair()` returned:
```json
"confidence_calibration": "calibrated_mlp"
```
While decision thresholds ($\tau = 0.62$ for relationship and $\Delta = 0.04$ for direction uncertainty) were tuned on the validation set, **no probability calibration procedure** (such as Platt scaling / logistic calibration, isotonic regression, or temperature scaling) was fitted to the model logits. Neural networks trained with binary cross-entropy and evaluated through raw sigmoids typically produce uncalibrated probability estimates. Calling the probability output `"calibrated_mlp"` conflated decision threshold selection with probability calibration.

### 3.2 Implemented Fix & Truthful Representation
1. In `backend/app/services/model_manager.py`:
   - Updated Phase 4 inference return:
     ```python
     "confidence_calibration": "uncalibrated_sigmoid",
     "decision_threshold_status": "validation_tuned"
     ```
   - Legacy scikit-learn remains: `"confidence_calibration": "scikit_learn_predict_proba"`
   - Heuristic fallback remains: `"confidence_calibration": "uncalibrated_heuristic"`
2. In `backend/app/api/endpoints.py`:
   - Added explicit disclosure to `/api/model-metrics`:
     ```json
     "probability_calibration": {
       "status": "UNCALIBRATED_SIGMOID",
       "procedure_fitted": null,
       "disclosure": "Model outputs are uncalibrated sigmoid probabilities. No Platt scaling, isotonic regression, or temperature scaling was fitted."
     }
     ```
   - Clarified purpose string: `"Decision threshold tuned strictly on validation split (probability outputs remain uncalibrated sigmoid)"`.

---

## 4. Direction Metrics Demarcation

Direction classifier performance involves four distinct concepts that must never be collapsed into a single "accuracy" metric:

| Metric Concept | Mathematical Definition | Verified Value on `SIPGD_v2` Test Split (162 pairs) | Reporting Field |
|:---|:---|:---:|:---|
| **Raw Direction Accuracy** | $\frac{\text{Correct Predictions across ALL related pairs}}{N_{\text{related}}}$ | **75.31%** (122 / 162) | `test_raw_accuracy` |
| **Abstention Coverage** | $\frac{\text{Pairs where } \|P(A \to B) - 0.5\| \ge \Delta}{N_{\text{related}}}$ with $\Delta = 0.04$ | **91.36%** (148 / 162) | `abstention_coverage` |
| **Confident-Prediction Accuracy** | $\frac{\text{Correct Predictions on confident subset}}{N_{\text{confident}}}$ | **78.38%** (116 / 148) | `test_confident_accuracy` |
| **Order Swap Consistency Error** | $\max \|P(A \to B) + P(B \to A) - 1.0\|$ | **$1.19 \times 10^{-7}$** (0.0% error) | `swap_antisymmetry_max_error` / `swap_consistency_rate: 1.0` |

### Verification:
All four metrics are independently exposed in `/api/model-metrics` under `direction_classifier_metrics` and verified via automated test `test_03_direction_metrics_demarcation` in `tests/test_phase8_readiness.py`.

---

## 5. Checkpoint & Metadata Compatibility Architecture

To prevent stale, malformed, or mismatched metadata from being reported as current model performance, `_init_phase4_models()` in `backend/app/services/model_manager.py` now enforces strict structural and parameter validation:

```python
# 1. Metadata Version Check
if meta.get("version") != "phase4_v1":
    raise ValueError(f"Incompatible metadata version: '{meta.get('version')}' (expected 'phase4_v1')")

# 2. Checkpoint Feature Dimension Validation
if rel_ckpt.get("in_dim", 115) != 115:
    raise ValueError("Relationship checkpoint dimension mismatch")
if dir_ckpt.get("in_dim", 172) != 172:
    raise ValueError("Direction checkpoint dimension mismatch")

# 3. Checkpoint Threshold Agreement with Metadata
if abs(float(rel_ckpt.get("threshold")) - float(meta["relationship_metrics"]["optimal_threshold"])) > 1e-4:
    raise ValueError("Checkpoint threshold does not match metadata threshold")
if abs(float(dir_ckpt.get("delta_uncertain")) - float(meta["direction_metrics"]["calibrated_delta_uncertain"])) > 1e-4:
    raise ValueError("Checkpoint delta does not match metadata delta")
```

### Safety Result:
If metadata is corrupted or stale, `ModelManager` refuses to set `phase4_loaded = True`, records the detailed exception in `phase4_fallback_reason`, falls back cleanly to legacy inference, and returns `status: "METRICS_UNAVAILABLE"` via `/api/model-metrics`.

---

## 6. Real-World Evaluation Protocol Audit

The protocol defined in `reports/PHASE7_REAL_WORLD_TEST_PROTOCOL.md` was audited against scientific and privacy standards:

1. **Source-Grouped Splits:**
   - Enforces unique immutable source IDs (`real_src_0001`, `real_src_0002`).
   - All physical derivatives (crops, resaves, prints, platform re-encodings) must be grouped with their original parent:
     $$\text{Sources}(\text{Split}_A) \cap \text{Sources}(\text{Split}_B) = \emptyset$$
2. **Independent Ground Truth:**
   - Ground truth recorded at acquisition time in `real_pairs_manifest.csv`.
   - Circular validation is explicitly prohibited: model predictions must **never** be used to label ground truth.
3. **Challenging Negative Pairs:**
   - Distractors include semantic negatives (same object category, different instance), sensor/device negatives (same smartphone sensor, distinct scene), and viewpoint negatives (near-duplicate timestamps).
4. **Privacy & Git Protection:**
   - Evaluated images reside exclusively in local directory `data/real_world_benchmark/`.
   - Root `.gitignore` explicitly blocks all image extensions (`*.jpg`, `*.jpeg`, `*.png`, `*.webp`, `*.bmp`, `*.tiff`).
   - Personal images, non-consenting faces, and PII are prohibited.

---

## 7. Safe Real-World Evaluation Runner (`src/evaluate_real_world.py`)

A dedicated evaluation script was created at `src/evaluate_real_world.py`:

### Key Capabilities & Safeguards:
- **Default Check:** If `data/real_world_benchmark/real_pairs_manifest.csv` does not exist or contains 0 rows, exits with status code 0 and reports:
  ```
  Status: PREPARED BUT NOT EXECUTED
  Safeguard Enforced: Zero results fabricated or assumed.
  Synthetic held-out metrics (SIPGD_v2) remain strictly isolated.
  ```
- **Source Leakage Detection:** If splits are provided, automatically verifies disjointness of source IDs before executing inference.
- **Calibration Computation:** Evaluates Expected Calibration Error (ECE) across 10 confidence bins and Brier score.
- **Dedicated Output Isolation:** Outputs results strictly to `reports/REAL_WORLD_EVALUATION.md`, preventing any modification to `reports/PHASE4_MODEL_EVALUATION.md`.

---

## 8. Test Execution Commands & Verified Results

### 8.1 Phase 8 Readiness Test Suite
Executed command:
```powershell
python -m unittest tests/test_phase8_readiness.py
```
**Result:** `Ran 6 tests in 7.670s — OK` (6/6 Passed, 0 Failures, 0 Errors)

1. `test_01_default_configuration_safety`: **PASS** (`ENABLE_PHASE4_MODELS=false` verified in config and `/api/health`).
2. `test_02_calibration_technical_justification`: **PASS** (`confidence_calibration == "uncalibrated_sigmoid"` and calibration disclosure verified).
3. `test_03_direction_metrics_demarcation`: **PASS** (`test_raw_accuracy`, `abstention_coverage`, and `test_confident_accuracy` verified distinct).
4. `test_04_checkpoint_metadata_incompatibility_rejection`: **PASS** (Incompatible metadata version actively rejected; `/api/model-metrics` reports `METRICS_UNAVAILABLE`).
5. `test_05_real_world_metrics_unexecuted_safeguard`: **PASS** (Real-world metrics report `NOT_VALIDATED` with 0 samples across all modes).
6. `test_06_real_world_script_safe_dry_run`: **PASS** (`evaluate_real_world.py` safely exits with `PREPARED BUT NOT EXECUTED`).

### 8.2 Full Cumulative Regression Suite
Executed command:
```powershell
python -m unittest tests/test_phase5_integration.py tests/test_phase6_audit_robustness.py tests/test_phase7_metrics_and_inference.py tests/test_phase8_readiness.py
```
**Result:** `Ran 36 tests in 20.730s — OK` (36/36 Passed, 0 Failures, 0 Errors)

- **Phase 5 Integration (12 tests):** 100% Pass.
- **Phase 6 Audit & Robustness (10 tests):** 100% Pass.
- **Phase 7 Metrics & Inference (8 tests):** 100% Pass.
- **Phase 8 Readiness & Safeguards (6 tests):** 100% Pass.
- **Total:** **36/36 tests passed.**

---

## 9. Current Unresolved Issues & Limitations

1. **No Authorized Real-World Dataset:** No camera or smartphone images have been contributed or authorized. Field performance on WhatsApp/social media re-compression, physical prints, and screen photos remains **NOT VALIDATED**.
2. **CPU Inference Latency:** Feature extraction via ViT takes ~400ms per image on CPU; pairwise classification takes ~10ms.
3. **Probability Calibration Not Fitted:** If well-calibrated confidence probabilities are required (e.g. for cost-sensitive forensic decision thresholds), Platt scaling or isotonic regression should be fitted on a real-world validation split.

---

## 10. Next Safe Action

1. **Do Not Enable Phase 4 by Default:** Preserve `ENABLE_PHASE4_MODELS=false` in `backend/app/config.py`.
2. **Contribute Authorized Real-World Images:** To evaluate real-world performance:
   - Provide an authorized set of at least 30 real-world image pairs in `data/real_world_benchmark/images/`.
   - Provide ground-truth metadata in `data/real_world_benchmark/real_pairs_manifest.csv`.
3. **Execute Benchmark:** Run `python src/evaluate_real_world.py`. Results will be generated in `reports/REAL_WORLD_EVALUATION.md` without contaminating synthetic baseline metrics.
