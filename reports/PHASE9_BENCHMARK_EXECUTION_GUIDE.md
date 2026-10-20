# PHASE 9: REAL-WORLD BENCHMARK EXECUTION GUIDE

**TRACE AI: Image Provenance Using Graphs**  
**Guide Version:** 1.0  
**Safety Status:** `ENABLE_PHASE4_MODELS=false` (Default: Disabled)  
**Execution Policy:** **NEVER INVENT METRICS — EXECUTE ONLY ON AUTHORIZED DATA**

---

## 1. Overview & Objectives

This guide provides the complete, authoritative operational instructions for executing the first controlled real-world benchmark of TRACE AI's image provenance pipeline without modifying model checkpoints, compromising privacy, or contaminating synthetic held-out evaluation baselines (`SIPGD_v2`).

All real-world evaluations run strictly through `src/evaluate_real_world.py`, which enforces source-grouped split independence, independent ground truth, input integrity, and isolated report generation.

---

## 2. Directory Structure

The real-world benchmark requires the following local directory structure under `data/`:

```
Trace-Ai-main/
├── data/
│   └── real_world_benchmark/
│       ├── real_pairs_manifest.template.csv  <- Provided reference template
│       ├── real_pairs_manifest.csv           <- Authorized CSV manifest (user-provided)
│       └── images/                           <- Authorized image files (user-provided)
│           ├── real_src_0001_orig.jpg
│           ├── real_src_0001_whatsapp.jpg
│           ├── real_src_0001_crop.png
│           ├── real_src_0002_orig.jpg
│           └── ...
└── reports/
    └── REAL_WORLD_EVALUATION.md              <- Generated output report
```

> [!IMPORTANT]
> All images inside `data/real_world_benchmark/images/` and any `*.jpg`, `*.jpeg`, `*.png`, `*.webp` files are ignored by git in `.gitignore`. **Never force-add or commit real-world images into Git version control.**

---

## 3. Manifest Schema & Conventions

The evaluator expects a comma-separated values (CSV) manifest located at `data/real_world_benchmark/real_pairs_manifest.csv` containing the following schema:

| Column Header | Data Type | Required | Valid Values / Constraints | Description |
|:---|:---:|:---:|:---|:---|
| `pair_id` | String | **Yes** | Unique identifier (e.g., `real_pair_0001`) | No duplicate IDs permitted. |
| `image_a` | String | **Yes** | Relative filename in `images/` | Must exist in `images/` and use supported extension (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`). |
| `image_b` | String | **Yes** | Relative filename in `images/` | Same constraints as `image_a`. Order-agnostic duplicate pairs are blocked. |
| `source_a` | String | **Yes** | Immutable scene ID (e.g., `real_src_0001`) | Root capture/source entity for Image A. |
| `source_b` | String | **Yes** | Immutable scene ID (e.g., `real_src_0001`) | Root capture/source entity for Image B. |
| `related` | Integer | **Yes** | `1` or `0` | **Ground truth relationship**. <br>Rule: `related == 1` $\iff$ `source_a == source_b`. |
| `ground_truth_direction` | String | **Yes** | `A_TO_B`, `B_TO_A`, `UNCERTAIN`, `NONE` | **Ground truth lineage direction**.<br>If `related == 0`, **must** be `NONE`.<br>If `related == 1`, must be `A_TO_B`, `B_TO_A`, or `UNCERTAIN` (for identical copies). |
| `applied_transformations` | String | No | Semicolon or comma-delimited list | Physical transformations applied (e.g., `jpeg;resize` or `crop`). Supported classes: `crop`, `resize`, `jpeg`, `blur`, `brightness`, `contrast`, `sharpen`, `rotation`, `color`, `noise`, `perspective`, `quality_degradation`. |
| `split` | String | No | e.g., `benchmark_val`, `benchmark_test` | Evaluation split. **Strict source-grouping enforced**: Sources in split A must never appear in split B. |
| `notes` | String | No | Free text | Description of capture device, transmission platform, or distractor category. |

---

## 4. Example Manifest (Placeholder Rows Only)

A starter template is saved at `data/real_world_benchmark/real_pairs_manifest.template.csv`:

```csv
pair_id,image_a,image_b,source_a,source_b,related,ground_truth_direction,applied_transformations,split,notes
placeholder_pair_0001,placeholder_source_0001.jpg,placeholder_source_0001_v01_whatsapp.jpg,real_src_0001,real_src_0001,1,A_TO_B,jpeg;resize,benchmark_test,Camera original to messaging app compression
placeholder_pair_0002,placeholder_source_0001.jpg,placeholder_source_0001_v02_crop.png,real_src_0001,real_src_0001,1,A_TO_B,crop,benchmark_test,Camera original to subregion crop
placeholder_pair_0003,placeholder_source_0001.jpg,placeholder_source_0002.jpg,real_src_0001,real_src_0002,0,NONE,none,benchmark_test,Semantic negative distractor (distinct scenes)
placeholder_pair_0004,placeholder_source_0003_a.jpg,placeholder_source_0004_b.jpg,real_src_0003,real_src_0004,0,NONE,none,benchmark_test,Same device sensor negative distractor
```

---

## 5. Input Validation & Error Handling Safeguards

The evaluation runner actively validates the manifest and environment before performing inference:

1. **Non-Existent Benchmark:** If `real_pairs_manifest.csv` is missing or empty, exits with code `0` and announces:
   `Benchmark Status: PREPARED BUT NOT EXECUTED`.
2. **Missing Required Columns:** Fails with `ERROR: Manifest missing required columns: ...`.
3. **Duplicate Pairs:** Rejects duplicate `pair_id` entries or duplicate `(image_a, image_b)` image sets.
4. **Semantic Inconsistencies:** Rejects cases where:
   - `related: 1` but `source_a != source_b`.
   - `related: 0` but `source_a == source_b`.
   - `related: 0` but direction is not `NONE`.
   - `related: 1` but direction is `NONE`.
5. **Source Leakage Across Splits:** Rejects manifests where any source ID appears in more than one split partition ($\text{Sources}(S_1) \cap \text{Sources}(S_2) \ne \emptyset$).
6. **Path Traversal & Unsupported Files:** Blocks directory traversal attempts (`..`, absolute paths) and unsupported extensions.

---

## 6. Execution Commands

### Step 0: Controlled-Transformation Benchmark Preparation (Phase 10)
To augment authorized real-world originals with controlled transformations (preserving confirmed pairs without modifying source images):
```powershell
python src/prepare_real_world_benchmark.py `
  --input-images-dir data/real_world_benchmark/images `
  --existing-manifest data/real_world_benchmark/real_pairs_manifest.csv `
  --output-dir data/real_world_benchmark/controlled_transforms `
  --seed 42
```
This generates:
- Isolated derivative images in `data/real_world_benchmark/controlled_transforms/images/`
- Validated manifest at `data/real_world_benchmark/controlled_transforms/manifest.csv`
- Parameter audit log at `data/real_world_benchmark/controlled_transforms/transform_params.json`
- Strict labeling: derivatives marked as `controlled_transform`, never independent captures.

### Step 1: Dry-Run Manifest Validation
Before running inference, validate the manifest schema, source grouping, and file references without loading neural networks:
```powershell
# For original single pair:
python src/evaluate_real_world.py --dry-run

# For prepared controlled-transforms dataset:
python src/evaluate_real_world.py `
  --manifest data/real_world_benchmark/controlled_transforms/manifest.csv `
  --images-dir data/real_world_benchmark/controlled_transforms/images `
  --dry-run
```
Expected output:
```
[RealWorldEval] Manifest validation: PASSED (N pairs, schema and source-grouping verified).
[RealWorldEval] Dry-run complete. All schema, consistency, and partition checks passed.
```

### Step 2: Full Benchmark Execution (When Approved)
Execution requires the explicit `--execute` flag. Without `--execute`, the evaluator operates in safe dry-run/readiness mode:
```powershell
python src/evaluate_real_world.py `
  --manifest data/real_world_benchmark/controlled_transforms/manifest.csv `
  --images-dir data/real_world_benchmark/controlled_transforms/images `
  --output-report reports/REAL_WORLD_EVALUATION.md `
  --execute
```

---

## 7. Expected Output Report Structure

The evaluator writes exclusively to `reports/REAL_WORLD_EVALUATION.md`:

```markdown
# REAL-WORLD BENCHMARK EVALUATION REPORT
**Status:** EXECUTED ON AUTHORIZED REAL-WORLD DATASET
**Total Pairs Evaluated:** N (Positive: P, Negative: N)
**Decision Threshold (tau):** 0.62 (Validation-tuned)
**Direction Uncertainty Delta (Delta):** 0.04

## 1. Pairwise Relationship Metrics
- Precision, Recall, F1-Score, False Positive Rate (FPR), ROC-AUC
- Brier Calibration Score
- Expected Calibration Error (ECE across 10 confidence bins)
- Confusion Matrix (TN, FP, FN, TP)

## 2. Direction Classification Metrics
- Raw Direction Accuracy (across all evaluated positive pairs)
- Abstention Coverage (|P(A -> B) - 0.5| >= 0.04)
- Confident-Prediction Accuracy (accuracy on non-abstaining pairs)

## 3. Transformation Multi-Label Performance (where supported)
- Macro-F1 across supported classes with support > 0
- Per-class Precision, Recall, F1, and Real-World Support

## 4. Benchmark Isolation & Limitations
- Strict isolation from synthetic held-out test metrics (SIPGD_v2)
- Domain specificity disclosures
- Uncalibrated sigmoid probability disclosure
```

---

## 8. Privacy, Legal & Security Precautions

1. **Contributor Consent:** Every photograph must have documented written authorization or permissive CC0/CC-BY licensing.
2. **PII and Face Redaction:** Do not include identifiable human faces, license plates, sensitive documents, credentials, or private property interiors without explicit written release.
3. **Exclusion from Version Control:**
   - Verify `.gitignore` contains `data/real_world_benchmark/` and `*.jpg`, `*.png`, etc.
   - Run `git status` to verify no raw images appear in untracked files before committing other code.
4. **No Synthetic Metric Pollution:** Real-world metrics must never be averaged with or substituted for synthetic test metrics (`SIPGD_v2`) in `reports/PHASE4_MODEL_EVALUATION.md`.

---

## 9. Verification & Automated Test Status

All safeguards, preparation rules, and manifest validation are covered by dedicated automated tests:
```powershell
python -m unittest tests/test_phase9_manifest_and_evaluator.py
python -m unittest tests/test_phase10_benchmark_preparation.py
```
**Phase 9 & 10 Test Results:** `17/17 Passed, 0 Failures, 0 Errors`.

**Cumulative Regression Suite (Phases 5–10):**
```powershell
python -m unittest tests/test_phase5_integration.py tests/test_phase6_audit_robustness.py tests/test_phase7_metrics_and_inference.py tests/test_phase8_readiness.py tests/test_phase9_manifest_and_evaluator.py tests/test_phase10_benchmark_preparation.py
```
**Cumulative Results:** `53/53 Passed, 0 Failures, 0 Errors` in 25.7s.
