# PHASE 10: REAL-WORLD BENCHMARK DATASET PREPARATION REPORT

**TRACE AI: Image Provenance Using Graphs**  
**Execution Date:** 2026-10-18  
**Report Version:** 2.0 (Updated with Contributor-Confirmed Multi-Scene Benchmark)  
**Phase Status:** **COMPLETE — DATASET PREPARED & VALIDATED (EVALUATION PENDING AUDIT REVIEW)**  
**Safety Flag:** `ENABLE_PHASE4_MODELS=false` (Production Default Preserved)  
**Evaluation Execution Policy:** **DRY-RUN VALIDATED — ACTUAL INFERENCE DEFERRED**  

---

## 1. Executive Summary

Phase 10 prepared and updated a controlled real-world benchmark dataset incorporating contributor-confirmed independent physical scenes alongside authorized photographs, strictly adhering to all architectural, provenance, legal, and privacy constraints:

1. **Original Image & Pair Preservation:** The existing contributor-confirmed real-world pair (`pair_0001` with direction `B_TO_A` and transformation `crop`) and all 25 authorized original photographs in `data/real_world_benchmark/images/` were preserved intact without renaming, moving, or modifying.
2. **Reuse of Existing Transformations:** Reused `src/transforms.py` (`apply_transform`, `sample_transform_params`, and canonical transformation classes) instead of duplicating image manipulation logic.
3. **Dedicated Output Isolation:** Generated derivative images and manifest are stored in an isolated directory (`data/real_world_benchmark/controlled_transforms/`), completely separate from synthetic training data (`data/provenance_dataset_v2/`) and original benchmark captures (`data/real_world_benchmark/images/`).
4. **Transparent Labeling Policy:** Derivative pairs are strictly recorded as `controlled_transform: <transform_name>`, explicitly preventing misrepresentation as independent real-world captures.
5. **Rigorous Negative Pair Grounding:** With the contributor confirming that all 23 newly added photographs depict different, independent physical scenes, the system mapped them to unique source IDs `real_src_0002` through `real_src_0024` (alongside `real_src_0001`). Exactly **24 cross-scene negative pairs** were generated across distinct confirmed physical scenes (`related=0`, `ground_truth_direction=NONE`, `applied_transformations=none`).
6. **Zero Source Leakage:** All source identifiers and derivatives reside strictly within a single partition (`benchmark_test`), verified by source-disjointness checks.
7. **End-to-End Integrity:** 100% of images were verified with PIL image headers, and manifest schema validation passed with 0 errors across all 35 pairs. Evaluator `--dry-run` passed cleanly.
8. **Regression & Safety:** 53/53 cumulative automated tests passed across Phases 5 through 10. `ENABLE_PHASE4_MODELS=false` remains in place.

---

## 2. Directory & Artifact Structure

The Phase 10 pipeline establishes the following isolated directory hierarchy:

```
Trace-Ai-main/
├── data/
│   ├── real_world_benchmark/
│   │   ├── images/                                     <- Authorized original photos (25 files preserved)
│   │   │   ├── WhatsApp Image 2026-10-13 at 1.31.11 AM.jpeg  (real_src_0001, cropped)
│   │   │   ├── WhatsApp Image 2026-10-13 at 1.31.12 AM.jpeg  (real_src_0001, root base)
│   │   │   ├── WhatsApp Image 2026-10-18 at 2.20.04 PM (1).jpeg (real_src_0002, Moon)
│   │   │   ├── WhatsApp Image 2026-10-18 at 2.20.04 PM (2).jpeg (real_src_0003, Earbuds)
│   │   │   ├── ... (21 additional authorized photos)
│   │   ├── real_pairs_manifest.csv                     <- Contributor-confirmed baseline (pair_0001)
│   │   ├── real_pairs_manifest.template.csv            <- Benchmark reference template
│   │   │
│   │   └── controlled_transforms/                      <- Dedicated Phase 10 Output Directory
│   │       ├── manifest.csv                            <- Full benchmark manifest (35 pairs)
│   │       ├── transform_params.json                   <- Parameter audit log
│   │       └── images/                                 <- Self-contained image directory (35 files)
│   │           ├── (25 original image copies)
│   │           ├── ct_real_src_0001_crop.jpg
│   │           ├── ct_real_src_0001_resize.jpg
│   │           ├── ct_real_src_0001_jpeg.jpg
│   │           ├── ct_real_src_0001_blur.jpg
│   │           ├── ct_real_src_0001_brightness.jpg
│   │           ├── ct_real_src_0001_contrast.jpg
│   │           ├── ct_real_src_0001_noise.jpg
│   │           ├── ct_real_src_0001_perspective.jpg
│   │           ├── ct_real_src_0001_sharpen.jpg
│   │           └── ct_real_src_0001_color.jpg
├── src/
│   ├── prepare_real_world_benchmark.py                 <- Preparation script (multi-scene & negatives)
│   └── evaluate_real_world.py                          <- Evaluation runner with --execute safety gate
└── tests/
    └── test_phase10_benchmark_preparation.py           <- Automated test suite (10 tests)
```

---

## 3. Confirmed Scene Mapping (`real_src_0001` through `real_src_0024`)

| Source ID | Image Filename | Dimensions | Visual Scene Subject |
|:---|:---|:---:|:---|
| `real_src_0001` | `WhatsApp Image 2026-10-13 at 1.31.12 AM.jpeg` (Root Base) <br>`WhatsApp Image 2026-10-13 at 1.31.11 AM.jpeg` (Crop) | 900×1600 px<br>1112×1308 px | Table surface with key / cropped table |
| `real_src_0002` | `WhatsApp Image 2026-10-18 at 2.20.04 PM (1).jpeg` | 900×1600 px | Night sky with Moon |
| `real_src_0003` | `WhatsApp Image 2026-10-18 at 2.20.04 PM (2).jpeg` | 900×1600 px | Wireless earbuds on patterned fabric |
| `real_src_0004` | `WhatsApp Image 2026-10-18 at 2.20.04 PM.jpeg` | 1600×900 px | Motorcycle front wheel & alloy rim |
| `real_src_0005` | `WhatsApp Image 2026-10-18 at 2.20.05 PM (1).jpeg` | 900×1600 px | Indoor classroom/office with students |
| `real_src_0006` | `WhatsApp Image 2026-10-18 at 2.20.05 PM.jpeg` | 900×1600 px | Sunset sky through window |
| `real_src_0007` | `WhatsApp Image 2026-10-18 at 2.20.06 PM (1).jpeg` | 900×1600 px | Man outdoors on phone near fence |
| `real_src_0008` | `WhatsApp Image 2026-10-18 at 2.20.06 PM.jpeg` | 1600×900 px | Scooter rear view on grass with license plate |
| `real_src_0009` | `WhatsApp Image 2026-10-18 at 2.20.07 PM (1).jpeg` | 1600×721 px | Fish swimming in aquarium tank |
| `real_src_0010` | `WhatsApp Image 2026-10-18 at 2.20.07 PM (2).jpeg` | 1600×721 px | Movie cinema screen with actor speaking |
| `real_src_0011` | `WhatsApp Image 2026-10-18 at 2.20.07 PM.jpeg` | 900×1600 px | Interior hallway with person walking |
| `real_src_0012` | `WhatsApp Image 2026-10-18 at 2.20.08 PM (1).jpeg` | 722×1600 px | Illuminated Ferris wheel at night |
| `real_src_0013` | `WhatsApp Image 2026-10-18 at 2.20.08 PM (2).jpeg` | 722×1600 px | Iced coffee latte glass with drinking straw |
| `real_src_0014` | `WhatsApp Image 2026-10-18 at 2.20.08 PM (3).jpeg` | 720×1600 px | Television screen in dark room |
| `real_src_0015` | `WhatsApp Image 2026-10-18 at 2.20.08 PM.jpeg` | 722×1600 px | Peacock perched on rooftop against sky |
| `real_src_0016` | `WhatsApp Image 2026-10-18 at 2.20.09 PM (1).jpeg` | 722×1600 px | Primate / monkey in zoo enclosure |
| `real_src_0017` | `WhatsApp Image 2026-10-18 at 2.20.09 PM (2).jpeg` | 722×1600 px | White python snake in terrarium |
| `real_src_0018` | `WhatsApp Image 2026-10-18 at 2.20.09 PM (3).jpeg` | 722×1600 px | Blue-and-yellow macaw parrot on branch |
| `real_src_0019` | `WhatsApp Image 2026-10-18 at 2.20.09 PM.jpeg` | 722×1600 px | Sunset clouds above forest tree line |
| `real_src_0020` | `WhatsApp Image 2026-10-18 at 2.20.10 PM (1).jpeg` | 1600×721 px | Cafe coffee cup ("SWISS CAFE") & helmet |
| `real_src_0021` | `WhatsApp Image 2026-10-18 at 2.20.10 PM (2).jpeg` | 950×1598 px | Forest foliage with distant wildlife |
| `real_src_0022` | `WhatsApp Image 2026-10-18 at 2.20.10 PM.jpeg` | 722×1600 px | White horse at stable fence |
| `real_src_0023` | `WhatsApp Image 2026-10-18 at 2.20.11 PM (1).jpeg` | 1600×721 px | Grilled sandwich on white plate |
| `real_src_0024` | `WhatsApp Image 2026-10-18 at 2.20.11 PM.jpeg` | 722×1600 px | Pasta dish served on restaurant dining table |

---

## 4. Manifest Inventory (`data/real_world_benchmark/controlled_transforms/manifest.csv`)

The manifest contains **35 verified pairs**:
- **1 Contributor-Confirmed Real Positive Pair (`pair_0001`):** `real_src_0001`, `B_TO_A`, `crop`.
- **10 Controlled-Transform Positive Pairs (`ct_pair_0002` through `ct_pair_0011`):** `real_src_0001`, `A_TO_B`, 10 canonical transformations.
- **24 Authorized Cross-Scene Negative Pairs (`neg_pair_0001` through `neg_pair_0024`):** Distinct confirmed physical sources, `related=0`, `direction=NONE`, `applied_transformations=none`.

```
Summary Counts:
├── Confirmed Independent Physical Scenes: 24
├── Authorized Original Images:           25
├── Controlled-Transform Images:          10
├── Total Images in Benchmark:            35
├── Total Pairs in Manifest:              35
│   ├── Positive Pairs:                   11 (1 contributor-confirmed + 10 controlled_transforms)
│   └── Negative (Unrelated) Pairs:       24 (Cross-scene ring pairs across distinct sources)
└── Partition:                            benchmark_test (zero leakage)
```

---

## 5. Verification & Dry-Run Results

### A. Phase 10 Unit Tests (`tests/test_phase10_benchmark_preparation.py`)
Executed via:
```powershell
python -m unittest tests/test_phase10_benchmark_preparation.py
```
**Results:** `10/10 Passed, 0 Failures, 0 Errors` in 2.17s.
- `test_01_controlled_transforms_artifacts_exist`: **PASS**
- `test_02_preservation_of_contributor_pair`: **PASS**
- `test_03_controlled_transform_labels_and_notes`: **PASS**
- `test_04_cross_scene_negative_pairs_validation`: **PASS**
- `test_05_manifest_validation_clean`: **PASS**
- `test_06_image_files_exist_and_readable`: **PASS**
- `test_07_evaluator_dry_run_succeeds`: **PASS**
- `test_08_transform_params_logged`: **PASS**
- `test_09_single_source_safeguard_zero_negatives`: **PASS**
- `test_10_independent_physical_sources_count`: **PASS**

### B. Evaluator Dry-Run (`src/evaluate_real_world.py`)
Executed on the 35-pair benchmark manifest:
```powershell
python src/evaluate_real_world.py `
  --manifest data/real_world_benchmark/controlled_transforms/manifest.csv `
  --images-dir data/real_world_benchmark/controlled_transforms/images `
  --dry-run
```
**Terminal Output:**
```
================================================================================
TRACE AI — Controlled Real-World Benchmark Evaluation Runner
================================================================================
[RealWorldEval] Manifest validation: PASSED (35 pairs, schema and source-grouping verified).
[RealWorldEval] Dry-run complete. All schema, consistency, and partition checks passed.
```

### C. Cumulative Regression Suite (Phases 5–10)
Executed via:
```powershell
python -m unittest tests/test_phase5_integration.py tests/test_phase6_audit_robustness.py tests/test_phase7_metrics_and_inference.py tests/test_phase8_readiness.py tests/test_phase9_manifest_and_evaluator.py tests/test_phase10_benchmark_preparation.py
```
**Results:** `53/53 Passed, 0 Failures, 0 Errors` in 25.70s.

---

## 6. Exact Operational Commands

### 1. Re-run Benchmark Dataset Preparation:
```powershell
python src/prepare_real_world_benchmark.py `
  --input-images-dir data/real_world_benchmark/images `
  --existing-manifest data/real_world_benchmark/real_pairs_manifest.csv `
  --output-dir data/real_world_benchmark/controlled_transforms `
  --seed 42 `
  --negative-strategy ring
```

### 2. Validate Benchmark Manifest & Integrity (Dry-Run):
```powershell
python src/evaluate_real_world.py `
  --manifest data/real_world_benchmark/controlled_transforms/manifest.csv `
  --images-dir data/real_world_benchmark/controlled_transforms/images `
  --dry-run
```

### 3. Run Automated Unit & Regression Tests:
```powershell
# Phase 10 tests
python -m unittest tests/test_phase10_benchmark_preparation.py

# Full regression suite (Phases 5–10)
python -m unittest tests/test_phase5_integration.py tests/test_phase6_audit_robustness.py tests/test_phase7_metrics_and_inference.py tests/test_phase8_readiness.py tests/test_phase9_manifest_and_evaluator.py tests/test_phase10_benchmark_preparation.py
```

### 4. Execute Full Benchmark (Only When Approved):
```powershell
python src/evaluate_real_world.py `
  --manifest data/real_world_benchmark/controlled_transforms/manifest.csv `
  --images-dir data/real_world_benchmark/controlled_transforms/images `
  --output-report reports/REAL_WORLD_EVALUATION.md `
  --execute
```

---

## 7. Known Limitations & Remaining Safeguards

1. **Controlled vs Independent Positive Pairs:** While the benchmark now contains 24 independent scenes and 24 legitimate negative distractor pairs, the positive pairs currently consist of 1 contributor-confirmed real-world pair (`pair_0001`, `crop`) and 10 controlled transformations on `real_src_0001`. Evaluating independent real-world positive transformations (e.g. independently captured camera edits) across other scenes will require further contributor-confirmed paired captures.
2. **Evaluation Pending Final Review:** In accordance with instructions, actual model inference (`--execute`) was **not** run. The benchmark remains in dry-run validated readiness.
3. **No Claim of General Real-World Accuracy:** While the dataset now provides diverse cross-scene distractors, model performance has not yet been benchmarked against these real-world images. Production metrics continue to report `real_world_metrics.status: NOT_VALIDATED`.
4. **Production Isolation:** `ENABLE_PHASE4_MODELS=false` remains strictly enforced in `backend/app/config.py`.
