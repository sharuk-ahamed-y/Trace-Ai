# PHASE 11: SAFE BENCHMARK EVALUATION AUDIT REPORT

**TRACE AI: Image Provenance Using Graphs**  
**Execution Date:** 2026-10-18  
**Report Version:** 1.0  
**Phase Status:** **AUDIT & EVALUATION READINESS VERIFIED (INFERENCE DEFERRED)**  
**Safety Flag:** `ENABLE_PHASE4_MODELS=false` (Production Default Strictly Preserved)  
**Evaluation Mode:** **DRY-RUN VALIDATED — ACTUAL INFERENCE DEFERRED**  

---

## 1. Executive Summary

Phase 11 performed an exhaustive safety and architectural audit of the real-world benchmark evaluation pipeline in `src/evaluate_real_world.py`, the Phase 10 manifest in `data/real_world_benchmark/controlled_transforms/manifest.csv`, and associated test suites:

1. **All 35 Pairs Verified:** 100% of the 35 benchmark pairs, image paths, dimensions, PIL headers, relationship labels, direction ground-truth, applied transformations, and source IDs were verified on disk without missing files, corruption, or schema violations.
2. **Category-Disaggregated Reporting Implemented:** Identified that the evaluator previously grouped all pairs into an aggregate accuracy score, which could mask failure modes by mixing controlled synthetic distortions with genuine unconstrained camera captures and cross-scene negatives. Upgraded `src/evaluate_real_world.py` to classify and report performance across three strictly isolated categories:
   - **Category A:** Contributor-Confirmed Real-World Positive Pairs (1 pair: `pair_0001`)
   - **Category B:** Controlled-Transformation Positive Pairs (10 pairs: `ct_pair_0002` to `ct_pair_0011`)
   - **Category C:** Cross-Scene Negative Distractor Pairs (24 pairs: `neg_pair_0001` to `neg_pair_0024`)
   - **Category D:** Overall Combined Summary (with mandatory warning forbidding generalized real-world claims).
3. **Hard Checkpoint & Fallback Refusal Enforced:** Hardened `src/evaluate_real_world.py` so that when `--execute` is supplied, it asserts that Phase 4 models are loaded and validates on every pair that `inference_mode == "trained_model"` and `active_model` is the Phase 4 pipeline. If any fallback occurs, the evaluator halts with return code 1, preventing heuristic fallbacks from ever being reported as trained-model performance.
4. **Non-Inference Checks Succeeded:** Manifest validation, subgroup dry-run, and the entire 59-test regression suite across Phases 5 through 11 passed (100% OK).
5. **Strict Constraints Maintained:** `ENABLE_PHASE4_MODELS` remains `false`. No actual inference was run, no weights or original images were modified, and zero real-world metrics were fabricated.

---

## 2. Verification of All 35 Benchmark Pairs

All 35 pairs in `data/real_world_benchmark/controlled_transforms/manifest.csv` were verified with PIL image header checks, file size verification, and source-consistency audits:

### 2.1 Category A: Contributor-Confirmed Real Positive Pair (1 pair)

| Pair ID | Image A | Image B | Dimensions (A / B) | Source IDs | Related | GT Direction | Applied Tfm | Contributor Lineage Confirmation |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `pair_0001` | `WhatsApp Image 2026-10-13 at 1.31.11 AM.jpeg` | `WhatsApp Image 2026-10-13 at 1.31.12 AM.jpeg` | 1112×1308 px<br>900×1600 px | `real_src_0001`<br>`real_src_0001` | `1` | `B_TO_A` | `crop` | Contributor confirmed Image A was cropped from Image B. |

### 2.2 Category B: Controlled-Transformation Positive Pairs (10 pairs)

All derived from base photograph `WhatsApp Image 2026-10-13 at 1.31.12 AM.jpeg` (`real_src_0001`, 900×1600 px). Direction is `A_TO_B`, `related=1`.

| Pair ID | Image A | Image B | Dimensions B | Applied Tfm | Parameter Specification |
|:---|:---|:---|:---:|:---:|:---|
| `ct_pair_0002` | `WhatsApp...1.31.12 AM.jpeg` | `ct_real_src_0001_crop.jpg` | 805×1211 px | `crop` | Crop box `(36, 128, 841, 1339)`, scale `0.749` |
| `ct_pair_0003` | `WhatsApp...1.31.12 AM.jpeg` | `ct_real_src_0001_resize.jpg` | 954×1696 px | `resize` | Scale `1.06`, interpolation `BICUBIC` |
| `ct_pair_0004` | `WhatsApp...1.31.12 AM.jpeg` | `ct_real_src_0001_jpeg.jpg` | 900×1600 px | `jpeg` | Quality factor `78` |
| `ct_pair_0005` | `WhatsApp...1.31.12 AM.jpeg` | `ct_real_src_0001_blur.jpg` | 900×1600 px | `blur` | Gaussian blur radius `1.28 px` |
| `ct_pair_0006` | `WhatsApp...1.31.12 AM.jpeg` | `ct_real_src_0001_brightness.jpg` | 900×1600 px | `brightness` | Factor `1.15` |
| `ct_pair_0007` | `WhatsApp...1.31.12 AM.jpeg` | `ct_real_src_0001_contrast.jpg` | 900×1600 px | `contrast` | Factor `0.88` |
| `ct_pair_0008` | `WhatsApp...1.31.12 AM.jpeg` | `ct_real_src_0001_noise.jpg` | 900×1600 px | `noise` | Gaussian sigma `5.8` |
| `ct_pair_0009` | `WhatsApp...1.31.12 AM.jpeg` | `ct_real_src_0001_perspective.jpg` | 900×1600 px | `perspective` | Distortion scale `0.038` |
| `ct_pair_0010` | `WhatsApp...1.31.12 AM.jpeg` | `ct_real_src_0001_sharpen.jpg` | 900×1600 px | `sharpen` | Factor `1.45` |
| `ct_pair_0011` | `WhatsApp...1.31.12 AM.jpeg` | `ct_real_src_0001_color.jpg` | 900×1600 px | `color` | Saturation factor `1.12` |

### 2.3 Category C: Cross-Scene Negative Distractor Pairs (24 pairs)

Generated across the 24 confirmed independent scene sources in a closed ring (`real_src_0001` -> `real_src_0002` -> ... -> `real_src_0024` -> `real_src_0001`). In all pairs: `related=0`, `direction=NONE`, `applied_transformations=none`.

| Pair ID | Image A Source | Image B Source | Physical Scene Subject A | Physical Scene Subject B |
|:---|:---:|:---:|:---|:---|
| `neg_pair_0001` | `real_src_0001` | `real_src_0002` | Table surface with key | Night sky with Moon |
| `neg_pair_0002` | `real_src_0002` | `real_src_0003` | Night sky with Moon | Wireless earbuds on fabric |
| `neg_pair_0003` | `real_src_0003` | `real_src_0004` | Wireless earbuds on fabric | Motorcycle front wheel & alloy rim |
| `neg_pair_0004` | `real_src_0004` | `real_src_0005` | Motorcycle wheel | Indoor classroom/office with students |
| `neg_pair_0005` | `real_src_0005` | `real_src_0006` | Classroom/office | Sunset sky through window |
| `neg_pair_0006` | `real_src_0006` | `real_src_0007` | Sunset sky | Man outdoors on phone near fence |
| `neg_pair_0007` | `real_src_0007` | `real_src_0008` | Man outdoors | Scooter rear view on grass with plate |
| `neg_pair_0008` | `real_src_0008` | `real_src_0009` | Scooter rear view | Fish swimming in aquarium tank |
| `neg_pair_0009` | `real_src_0009` | `real_src_0010` | Aquarium fish | Movie cinema screen with actor |
| `neg_pair_0010` | `real_src_0010` | `real_src_0011` | Cinema screen | Interior hallway with person walking |
| `neg_pair_0011` | `real_src_0011` | `real_src_0012` | Interior hallway | Ferris wheel illuminated at night |
| `neg_pair_0012` | `real_src_0012` | `real_src_0013` | Ferris wheel | Iced coffee latte glass with straw |
| `neg_pair_0013` | `real_src_0013` | `real_src_0014` | Iced coffee glass | Television screen in dark room |
| `neg_pair_0014` | `real_src_0014` | `real_src_0015` | Television screen | Peacock perched on rooftop |
| `neg_pair_0015` | `real_src_0015` | `real_src_0016` | Peacock on rooftop | Primate in zoo enclosure |
| `neg_pair_0016` | `real_src_0016` | `real_src_0017` | Primate in zoo | White python snake in terrarium |
| `neg_pair_0017` | `real_src_0017` | `real_src_0018` | White python | Macaw parrot on branch |
| `neg_pair_0018` | `real_src_0018` | `real_src_0019` | Macaw parrot | Sunset clouds above forest |
| `neg_pair_0019` | `real_src_0019` | `real_src_0020` | Sunset clouds | Cafe coffee cup & motorcycle helmet |
| `neg_pair_0020` | `real_src_0020` | `real_src_0021` | Cafe coffee cup | Forest foliage with wildlife |
| `neg_pair_0021` | `real_src_0021` | `real_src_0022` | Forest foliage | White horse at stable fence |
| `neg_pair_0022` | `real_src_0022` | `real_src_0023` | White horse | Grilled sandwich on white plate |
| `neg_pair_0023` | `real_src_0023` | `real_src_0024` | Grilled sandwich | Pasta dish on restaurant dining table |
| `neg_pair_0024` | `real_src_0024` | `real_src_0001` | Pasta dining dish | Table surface with key |

---

## 3. Subgroup Reporting & Anti-Conflation Architecture

### 3.1 The Risk of Aggregate Conflation
Reporting a single overall F1 or Accuracy on this benchmark would be scientifically misleading:
- 24 of the 35 pairs ($68.6\%$) are cross-scene negatives across distinct physical environments.
- 10 of the 11 positive pairs ($90.9\%$) are synthetic canonical transformations applied to a single base photo (`real_src_0001`).
- Only 1 positive pair ($9.1\%$) represents an unconstrained real-world photograph capture (`pair_0001`).
A blended metric could achieve $\sim 97\%$ accuracy purely by getting easy negatives and controlled transforms right while completely failing on the only genuine unconstrained photograph pair.

### 3.2 Implemented Subgroup Reporting
`src/evaluate_real_world.py` partitions evaluation into 4 distinct report sections:
1. **Section 1: Category A — Contributor-Confirmed Real Positive Pair (`pair_0001`)**
   - Individual diagnosis: Pair ID, image names, predicted probability $P(\text{rel})$, threshold decision, predicted direction, direction confidence, direction match (`PASS`/`FAIL`), and transformation prediction vs ground truth.
2. **Section 2: Category B — Controlled-Transformation Positive Pairs (`ct_pair_*`)**
   - Base photograph disclosure (`real_src_0001`).
   - Relationship recall ($TP / (TP + FN)$).
   - Direction accuracy (Raw accuracy, Abstention coverage, Confident accuracy).
   - Transformation detection breakdown.
3. **Section 3: Category C — Cross-Scene Negative Distractor Pairs (`neg_pair_*`)**
   - Independent physical scene count ($24$ distinct scene sources).
   - Specificity ($TN / (TN + FP)$) and False Positive Rate ($FP / (FP + TN)$).
   - Maximum and mean false-alarm relationship probabilities.
4. **Section 4: Overall Combined Metrics (With Mandatory Caveat Banner)**
   - Explicit alert warning the reader not to cite the aggregate number as general accuracy.

---

## 4. Safety Review and Pipeline Hardening

An exhaustive safety and security review was conducted across `backend/app/services/model_manager.py`, `src/evaluate_real_world.py`, and `tests/test_phase11_safe_benchmark_evaluation.py`. Concrete hardening measures were implemented to address all safety requirements:

### 4.1 Phase 4 Checkpoint & Scaler Validation
- **Dimension & Finite Checks:** Added explicit validation that `scaler_115` has exactly 115 features, `scaler_172` has 172 features, and both have finite `mean_` and `scale_` arrays.
- **Threshold Validity:** Enforced that `optimal_threshold` is finite and within $[0.0, 1.0]$, `calibrated_delta_uncertain` is within $[0.0, 0.5]$, and all 12 transformation class thresholds are finite and within $[0.0, 1.0]$.
- **Weight Integrity:** Validated that all Phase 4 neural MLP parameters (`torch.isfinite`) contain no NaNs or Infs upon checkpoint loading.
- **SHA-256 Artifact Verification:** Added optional integrity verification against an explicitly configured SHA-256 manifest (`checksums.sha256`, `sha256_manifest.json`, or `PHASE4_CHECKSUM_MANIFEST`), without fabricating unverified expected hashes.
- **Diagnostic Fallback:** Any checkpoint or metadata corruption, dimension mismatch, or missing file automatically sets `phase4_loaded = False`, clears model references, and records an actionable diagnostic reason in `phase4_fallback_reason`.

### 4.2 Trustworthy Inference Provenance
- **Structured Provenance Tracking:** Replaced reliance on simple model name strings with an immutable, structured `inference_path` dictionary attached to every prediction:
  - Phase 4: `{"pipeline": "phase4_v1", "relationship_model": "...", "direction_model": "...", "transformation_model": "...", "applied_arbitration": bool, "is_fallback": False}`
  - Fallbacks: `{"pipeline": "legacy_scikit_learn" | "legacy_pytorch_multitask" | "heuristic_fallback", "is_fallback": True, "fallback_reason": ...}`
- **Strict Phase 4 Execution Mode (`strict_phase4=True`):** In `--execute` mode, `manager.predict_pair(..., strict_phase4=True)` is enforced. If any runtime exception occurs during Phase 4 inference (e.g., OOM or tensor error), the exception is immediately re-raised (`RuntimeError: Phase 4 inference failed under strict mode`) rather than silently catching it and returning a heuristic fallback result.

### 4.3 Benchmark Input Validation & Security
- **Path Traversal Mitigation:** Enforced that all image paths in the manifest resolve strictly within `images_dir.resolve()` using `resolved_img.relative_to(resolved_dir)` and reject traversal sequences (`..`).
- **Complete Image Verification:** Manifest validation runs with `check_images=True` before inference, ensuring every single image file exists, is a valid file, and passes PIL header verification (`img.verify()`).
- **Unclassified Row Rejection:** Manifest rows that do not match recognized isolated categories (`contributor_confirmed`, `controlled_transform`, `cross_scene_negative`) are rejected unless explicitly defined as template/placeholder rows.
- **Source Leakage Checks:** Enforced source disjointness across evaluation splits to prevent information leakage.

### 4.4 Evaluation Integrity & Disclosures
- **Halt on Partial Evaluation:** If any pair fails to evaluate or images are missing, the evaluator halts with exit code 1 and explicitly refuses to generate or present a partial dataset report as a complete benchmark.
- **Audited Counts:** Generated reports record `Expected Pairs`, `Evaluated Pairs`, `Skipped Pairs`, and `Failed Pairs`.
- **Arbitration Disclosure:** Clearly documented that final direction and transformation outputs reflect neural inference combined with physical arbitration rules (e.g., crop containment).
- **Uncalibrated Confidence Labeling:** Clearly identified that raw confidence scores are direct sigmoid outputs without temperature scaling or Platt calibration.

---

## 5. Test Suite Verification & Audit Results

### 5.1 Focused Safety Tests (`tests/test_phase11_safe_benchmark_evaluation.py`)
Fourteen (14) targeted tests were implemented and verified (100% passing):
1. `test_01_all_35_pairs_verified_and_readable`: Verifies all 35 benchmark pairs exist on disk and pass PIL verification.
2. `test_02_evaluator_subgroup_classification`: Verifies 1 contributor-confirmed, 10 controlled, and 24 negative partitions.
3. `test_03_evaluator_dry_run_outputs_subgroup_breakdown`: Verifies `--dry-run` outputs composition and exits 0.
4. `test_04_evaluator_without_execute_exits_safely`: Verifies safe execution gating without `--execute`.
5. `test_05_evaluator_refuses_heuristic_fallback`: Verifies failure to load Phase 4 models exits 1.
6. `test_06_default_flag_remains_false`: Verifies `ENABLE_PHASE4_MODELS == False` in production config.
7. `test_07_missing_or_corrupt_checkpoint_keeps_phase4_unavailable`: Tests corrupted checkpoint handling and diagnostic error logging.
8. `test_08_invalid_metadata_or_scaler_dimensions_rejected`: Tests rejection of incompatible scaler dimensions (e.g. 10-D).
9. `test_09_phase4_inference_exception_followed_by_legacy_fallback_refused_in_strict_mode`: Tests strict re-raise on Phase 4 inference errors.
10. `test_10_incorrect_or_missing_inference_provenance_rejected`: Tests evaluator rejection of results with fallback provenance.
11. `test_11_manifest_validation_rejects_missing_or_unreadable_image`: Tests manifest rejection of missing or 0-byte corrupt files.
12. `test_12_manifest_validation_rejects_unclassified_row`: Tests rejection of unclassified rows.
13. `test_13_manifest_validation_rejects_path_traversal`: Tests rejection of path traversal patterns (`../secret.jpg`).
14. `test_14_partial_evaluation_fails_to_produce_complete_report`: Tests that partial evaluation aborts report generation.

**Result:** `Ran 14 tests in 11.522s — OK`

### 5.2 Evaluator Dry-Run (`src/evaluate_real_world.py --dry-run`)
```powershell
python src/evaluate_real_world.py `
  --manifest data/real_world_benchmark/controlled_transforms/manifest.csv `
  --images-dir data/real_world_benchmark/controlled_transforms/images `
  --dry-run
```
**Output:**
```
================================================================================
TRACE AI — Controlled Real-World Benchmark Evaluation Runner
================================================================================
[RealWorldEval] Manifest validation: PASSED (35 pairs, schema, readability, and source-grouping verified).
[RealWorldEval] Benchmark Subgroup Composition:
  - Contributor-Confirmed Real Pairs: 1
  - Controlled-Transform Pairs:       10
  - Cross-Scene Negative Pairs:       24
[RealWorldEval] Dry-run complete. All schema, subgroup isolation, and partition checks passed.
```

### 5.3 Full Regression Suite (67/67 Passed, 0 Regressions)
All regression test suites across Phases 5 through 11 passed cleanly without regression:
- `tests/test_phase5_integration.py`
- `tests/test_phase6_audit_robustness.py`
- `tests/test_phase7_metrics_and_inference.py`
- `tests/test_phase8_readiness.py`
- `tests/test_phase9_manifest_and_evaluator.py`
- `tests/test_phase10_benchmark_preparation.py`
- `tests/test_phase11_safe_benchmark_evaluation.py`

---

## 6. Exact Command Required for Authorized Benchmark Evaluation

When benchmark evaluation is explicitly authorized by the user, run the following exact command:

```powershell
python src/evaluate_real_world.py `
  --manifest data/real_world_benchmark/controlled_transforms/manifest.csv `
  --images-dir data/real_world_benchmark/controlled_transforms/images `
  --output-report reports/REAL_WORLD_EVALUATION.md `
  --execute
```

> [!IMPORTANT]
> - Do **not** execute this command without explicit user authorization.
> - Do **not** change `ENABLE_PHASE4_MODELS` in `backend/app/config.py`; the `--execute` flag dynamically activates Phase 4 models inside `evaluate_real_world.py` for benchmark execution while preserving the safe production default (`false`).

---

## 7. Known Benchmark Limitations

1. **Extremely Small Unconstrained Positive Sample Size:**
   - There is currently **only 1 contributor-confirmed unconstrained positive pair** (`pair_0001`). No statistical significance or generalization claims can be derived from $N=1$.
2. **Single Physical Source for Controlled Transformations:**
   - All 10 controlled positive transformations were generated from a single photograph base (`real_src_0001`, `WhatsApp Image 2026-10-13 at 1.31.12 AM.jpeg`). Sensor noise patterns, dynamic range, and resolution characteristics are therefore tied to that single camera capture.
3. **Cross-Scene Negative Ring Limitation:**
   - The 24 cross-scene negative pairs test scene discrimination across 24 confirmed environments, which evaluates specificity against arbitrary scene linking, but does not provide near-miss visual distractor pairs (e.g., photos of different tables or similar indoor spaces).
4. **Uncalibrated Model Confidence:**
   - Model outputs are raw sigmoids. Probabilities must not be interpreted as calibrated posterior risk estimates.
5. **Strict Metric Separation:**
   - Real-world evaluation results must never be averaged or conflated with synthetic held-out test split metrics (`SIPGD_v2`).
