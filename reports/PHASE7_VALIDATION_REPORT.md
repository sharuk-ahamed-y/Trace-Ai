# PHASE 7: METRICS INTEGRITY, INFERENCE TRANSPARENCY & CONTROLLED-TEST READINESS REPORT

**TRACE AI: Image Provenance Using Graphs**  
**Execution Timestamp:** 2026-10-18  
**Environment:** Windows (Python 3.12, PyTorch 2.6.0+cpu, scikit-learn 1.7.0, FastAPI 0.115.8)  
**Safety Status:** `ENABLE_PHASE4_MODELS=false` (Default: Disabled)  
**Controlled-Test Readiness:** **READY FOR CONTROLLED EVALUATION**

---

## 1. Files Changed & Technical Rationale

| File Path | Nature of Change | Technical Rationale |
|:---|:---:|:---|
| `backend/app/services/model_manager.py` | Modified | 1. Added `phase4_metadata` and `phase4_fallback_reason` attributes to `ModelManager`.<br>2. Updated `predict_pair()` across all branches (Phase 4, legacy scikit-learn, legacy PyTorch multi-task, and heuristic fallback) to return explicit metadata: `inference_mode` (`"trained_model"` vs `"heuristic_fallback"`), `confidence_calibration` (`"calibrated_mlp"` vs `"uncalibrated_heuristic"`), and `fallback_reason`.<br>3. Prevented heuristic fallback from forcing an ungrounded `A_TO_B` direction; now returns `UNCERTAIN` when related.<br>4. Aligned model identity string to `"ViT Cosine Heuristic"` across `predict_pair()`, `get_status()`, and API endpoints.<br>5. Ensured `set_phase4_enabled()` calls `initialize()` to prevent uninitialized attribute lookups. |
| `backend/app/api/endpoints.py` | Modified | 1. Rewrote `/api/model-metrics` to dynamically serve verified metrics from `models/phase4_v1/metadata.json` when Phase 4 is enabled, label legacy baseline metrics when disabled, and report `METRICS_UNAVAILABLE` without crashing or fabricating if metadata is missing.<br>2. Explicitly separated synthetic held-out test metrics (`SIPGD_v2`), validation threshold-tuning metrics, and real-world metrics (`NOT_VALIDATED`).<br>3. Updated `/api/model-info` to document Phase 4 Normalized Forensic MLP architecture components.<br>4. Exposed `phase4_fallback_reason` in `/api/health`. |
| `tests/test_phase7_metrics_and_inference.py` | Created | Added 8 automated tests covering metrics integrity, inference transparency, safe fallbacks, partial checkpoint handling, API schema compatibility, model identity agreement, and toggle idempotency. |
| `reports/PHASE7_REAL_WORLD_TEST_PROTOCOL.md` | Created | Comprehensive experimental protocol for collecting, annotating, evaluating, and securing real-world imagery without data leakage or version control exposure. |
| `reports/PHASE7_VALIDATION_REPORT.md` | Created | This validation report. |

---

## 2. Before-and-After Behavior of Metrics Reporting

### Before Phase 7
- `/api/model-metrics` returned static, hardcoded JSON (96 evaluation samples, 85.42% accuracy, 1.0 precision) originating from early legacy scikit-learn experiments, regardless of whether Phase 4 models were enabled or disabled.
- No distinction was made between synthetic data and real-world performance.
- If Phase 4 models were enabled, `/api/model-metrics` gave no indication that a 115-D/172-D normalized MLP pipeline was active or that test performance was verified on 324 samples.

### After Phase 7
- **When Phase 4 is Active (`ENABLE_PHASE4_MODELS=true`):**
  - Dynamically loads and returns verified evaluation metrics from `models/phase4_v1/metadata.json`.
  - Reports sample counts accurately (324 held-out test pairs, 384 validation pairs).
  - Explicitly tags dataset provenance: `SYNTHETIC_HELD_OUT_TEST_SIPGD_V2`.
  - Distinguishes validation threshold selection ($0.62$ relationship, $\Delta = 0.04$ direction) from test evaluation.
  - Serves multi-label transformation macro-F1 (0.6197) and per-class stats.
  - Serves direction classifier metrics (swap consistency 1.0, confident accuracy 0.7838 at 91.36% coverage).
  - Explicitly reports real-world metrics as:
    ```json
    "real_world_metrics": {
      "evaluation_samples": 0,
      "status": "NOT_VALIDATED",
      "disclosure": "Real-world dataset not present in repository. Real-world accuracy is unmeasured."
    }
    ```
- **When Phase 4 is Disabled (`ENABLE_PHASE4_MODELS=false`):**
  - Reports legacy baseline metrics, explicitly labeled as `Legacy Baseline: Scikit-learn Pipeline / ViT Heuristic` and `LEGACY_BASELINE_METRICS`.
- **When Metadata is Missing or Corrupted:**
  - Reports `status: "METRICS_UNAVAILABLE"` and `test_split_metrics: None` without fabricating fallback numbers or throwing 500 errors.

---

## 3. Active-Model Identification & Fallback Transparency

| Pipeline State | `active_model` Reported | `inference_mode` | `confidence_calibration` | `fallback_reason` | Direction Handling |
|:---|:---|:---:|:---:|:---|:---:|
| **Phase 4 Enabled & Valid** | `Phase 4 Normalized Forensic MLP Pipeline (v1)` | `trained_model` | `calibrated_mlp` | `None` | `A_TO_B`, `B_TO_A`, or `UNCERTAIN` |
| **Phase 4 Enabled but Missing Checkpoint** | Legacy Model (`Scikit-learn` or `ViT Cosine Heuristic`) | As determined by legacy | As determined by legacy | `"Missing required Phase 4 artifact: ..."` | Reverts safely |
| **Phase 4 Disabled by Config** | `Scikit-learn (...)` or `ViT Cosine Heuristic` | `trained_model` or `heuristic_fallback` | `predict_proba` or `uncalibrated_heuristic` | `"Phase 4 models disabled by configuration"` | Supported legacy behavior |
| **Total Model Absence (ViT Heuristic)** | `ViT Cosine Heuristic` | `heuristic_fallback` | `uncalibrated_heuristic` | Explicit explanation of heuristic fallback | **`UNCERTAIN`** when related (never forced to `A_TO_B`) |

---

## 4. Test Execution & Results

### 4.1 Phase 7 Metrics & Inference Suite
Executed command:
```powershell
python -m unittest tests/test_phase7_metrics_and_inference.py
```
**Result:** `Ran 8 tests in 6.659s — OK` (8/8 Passed, 0 Failures, 0 Errors)

1. `test_01_phase4_disabled_legacy_identity`: **PASS** (Confirmed legacy identity, status tags, and fallback indicators).
2. `test_02_phase4_enabled_model_and_metrics`: **PASS** (Confirmed dynamic Phase 4 metrics, sample count 324, and precision 0.9877).
3. `test_03_missing_metadata_marked_unavailable`: **PASS** (Confirmed `METRICS_UNAVAILABLE` when metadata path is invalid).
4. `test_04_missing_checkpoint_safe_fallback`: **PASS** (Confirmed safe fallback without crashes or mocked predictions).
5. `test_05_partial_checkpoints_prevention_of_version_mixture`: **PASS** (Confirmed missing 1 file aborts Phase 4 completely; 0 model version mixing).
6. `test_06_api_schema_compatibility`: **PASS** (Confirmed `/api/health`, `/api/model-info`, and `/api/model-metrics` schemas match contract).
7. `test_07_model_identity_agreement_across_endpoints`: **PASS** (Confirmed `predict_pair()`, `/api/health`, and `/api/model-info` return identical model strings).
8. `test_08_no_stale_identity_on_repeated_toggles`: **PASS** (Confirmed repeated toggling does not cache stale identity).

---

## 5. Comprehensive Regression Test Summary

All existing test suites were executed back-to-back:
```powershell
python -m unittest tests/test_phase5_integration.py tests/test_phase6_audit_robustness.py
```
**Result:** `Ran 22 tests in 16.537s — OK` (22/22 Passed, 0 Failures, 0 Errors)

- **Phase 5 Integration (12 tests):** 100% Pass.
- **Phase 6 Robustness & Adversarial Checks (10 tests):** 100% Pass.
- **Phase 7 Metrics & Inference (8 tests):** 100% Pass.
- **Total Combined Tests Executed:** **30 tests — 30 passed, 0 failed.**

---

## 6. Runtime Configuration vs. Startup Switching Disclosures

1. **Environment Variable (`ENABLE_PHASE4_MODELS`):**
   - Evaluated by `backend/app/config.py` at module import time.
   - For an external FastAPI process running under Uvicorn, changing `$env:ENABLE_PHASE4_MODELS` in a shell requires restarting the backend process to take effect.
2. **In-Memory Programmatic Switching:**
   - Within Python tests or scripts, `ModelManager().set_phase4_enabled(True/False)` supports dynamic switching at runtime without restarts.
3. **Disclosure:** Runtime switching is **not** claimed for deployed HTTP processes; restart is the documented, supported operational procedure.

---

## 7. Remaining Limitations

1. **Domain Gap to Real-World Imagery:** The models have been verified on synthetic procedural operations (`SIPGD_v2`). Real-world performance on smartphone cameras, messaging app compression, and physical photographs of screens remains **NOT VALIDATED**.
2. **CPU Inference Latency:** On CPU, ViT feature extraction takes ~400ms per image; pairwise MLP classification takes ~10ms.
3. **No Dynamic Runtime Switch via REST API:** Toggling the model switch in production requires setting the environment variable and restarting the backend.

---

## 8. Controlled-Test Readiness & Pre-Activation Requirements

### Readiness Status: **READY FOR CONTROLLED TESTING**
TRACE AI is fully prepared for controlled evaluation:
- Default state remains safely **`ENABLE_PHASE4_MODELS=false`**.
- Enabling Phase 4 via `$env:ENABLE_PHASE4_MODELS = "true"` cleanly activates the normalized forensic models, serves verified metrics, and preserves API contracts.

### Required Actions Before Full Production Activation:
1. **Real-World Pilot Evaluation:** Execute `reports/PHASE7_REAL_WORLD_TEST_PROTOCOL.md` on an authorized sample of at least 30 real-world image pairs.
2. **User Approval:** Obtain explicit user authorization to change the default in `backend/app/config.py` from `False` to `True`.
3. **No Automatic Commits or Deployments:** Changes remain strictly local and uncommitted awaiting user review.
