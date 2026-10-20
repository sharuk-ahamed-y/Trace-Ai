# TRACE AI — PHASE 12: DATASET EXPANSION PROTOCOL & QA SPECIFICATION

**TRACE AI: Image Provenance Using Graphs**  
**Version:** 1.0  
**Phase Status:** **DATASET EXPANSION STRATEGY & QA TOOLING VERIFIED (NO INFERENCE)**  
**Safety Flag:** `ENABLE_PHASE4_MODELS=false` (Production Default Strictly Preserved)  
**Evaluation Mode:** **INFERENCE STRICTLY DEFERRED**

---

## 1. Executive Summary

Phase 11 established rigorous safety gates, parameter integrity checks, and fallback refusal guarantees on the real-world evaluation runner. However, the existing benchmark dataset ($N=35$ pairs) remains a **limited controlled evidence set** and is fundamentally insufficient to claim general real-world forensic accuracy.

Phase 12 defines a reproducible, leakage-resistant expansion protocol and provides standalone dataset quality assurance (QA) tooling (`src/qa_benchmark_dataset.py`) and tests (`tests/test_phase12_dataset_qa.py`).

### Key Deficiencies of the Current Benchmark ($N=35$)
1. **Single Positive Physical Source ($N=1$):**
   - The entire positive evidence pool is concentrated in a single physical scene (`real_src_0001`): 1 contributor-confirmed unconstrained pair (`pair_0001`) and 10 controlled transformations of that same photograph (`ct_pair_0002` to `ct_pair_0011`).
   - If evaluated today, positive detection and direction recall reflect behavior on one specific smartphone camera sensor and lighting environment.
2. **Absence of Hard Semantic Negatives:**
   - The 24 negative pairs are distinct physical scenes arranged in a simple closed ring. None share similar objects, viewpoints, or repeated textures, meaning near-miss visual distractors are currently unmeasured.
3. **Impossibility of Train/Val/Test Partitioning:**
   - Because all positive pairs derive from `real_src_0001`, splitting the dataset by source-group leaves all non-test splits with zero positive real-world samples.

---

## 2. Dataset Expansion Strategy (Three Isolated Groups)

To build a statistically meaningful benchmark, all candidate pairs must be collected, tracked, and evaluated in three strictly separated categories:

```
                                 ┌─────────────────────────────────────────────────────────────┐
                                 │         TRACE AI Expanded Benchmark Dataset (Target)        │
                                 └──────────────────────────────┬──────────────────────────────┘
                                                                │
                 ┌──────────────────────────────────────────────┼──────────────────────────────────────────────┐
                 │                                              │                                              │
                 ▼                                              ▼                                              ▼
   ┌───────────────────────────┐                  ┌───────────────────────────┐                  ┌───────────────────────────┐
   │        Category A         │                  │        Category B         │                  │        Category C         │
   │   Contributor-Confirmed   │                  │       Hard Negative       │                  │        Controlled         │
   │      Positive Pairs       │                  │       Visual Pairs        │                  │     Transformations       │
   ├───────────────────────────┤                  ├───────────────────────────┤                  ├───────────────────────────┤
   │ • Multiple real sensors   │                  │ • Visually similar scenes │                  │ • Multi-source originals  │
   │ • Verified camera edits   │                  │ • Multi-camera viewpoints │                  │ • Deterministic params    │
   │ • Messaging compression   │                  │ • Same subject, diff time │                  │ • Preserved source IDs    │
   │ • Verified lineage        │                  │ • Confirmed unrelated     │                  │ • Canonical distortions   │
   └───────────────────────────┘                  └───────────────────────────┘                  └───────────────────────────┘
```

### 2.1 Category A: Contributor-Confirmed Positive Pairs
- **Target Size:** Minimum 30 independent positive source groups (aiming for $\ge 60$ pairs across diverse devices).
- **Physical Provenance Requirement:**
  - Must represent genuine camera captures where the contributor physically executed or directly witnessed the lineage (e.g. camera capture $\to$ photo editing app $\to$ messaging export $\to$ web upload).
  - Must never be labeled `related=1` based on visual similarity alone.
- **Transformation Diversity:**
  - **Cropping & Aspect Ratio Adjustments:** Subregion crops ($50\% - 90\%$ area retention).
  - **Recompression & Format Transcoding:** Messaging apps (WhatsApp, Telegram, Signal, iMessage) and JPEG quality degradation ($Q=65 - 85$).
  - **Color, Brightness, & Tone Curve Adjustments:** Mobile camera filters, contrast/saturation tweaks.
  - **Downscaling & Resampling:** Bilinear, bicubic, and Lanczos downsampling.
  - **Screenshots:** Screenshots of original photos on mobile displays (testing border and UI artifacts).

### 2.2 Category B: Hard Negative Pairs
- **Target Size:** Minimum 60 pairs across distinct physical sources.
- **Visual Distractor Categories:**
  - **Visual Match / Near-Miss Scenes:** Distinct physical locations sharing similar architectural themes, wallpaper, foliage, or furniture.
  - **Multi-Angle Distinct Captures:** Two separate camera shots of the same static room or object taken from different angles/times (distinct shutter events $\implies$ unrelated lineage, `related=0`, `direction=NONE`).
  - **Repeated Subject Distractors:** Different animals of the same breed, different vehicles of the same model, or similar desktop computer setups.
- **Uncertainty Rule:** If two images look similar but independent provenance cannot be confirmed with certainty, mark `related=unknown` and quarantine in `expansion_workspace/`. Never force uncertain pairs into positive or negative labels.

### 2.3 Category C: Multi-Source Controlled Transformations
- **Target Size:** 10 canonical transformations applied across $\ge 10$ distinct base photographs ($\ge 100$ controlled pairs).
- **Multi-Source Requirement:**
  - Base photographs must span indoor, outdoor, portrait, document, and low-light scenes.
  - Every derivative must inherit the base photograph's `real_src_XXXX` identifier so that all derivatives remain tied to that source during partitioning.

---

## 3. Provenance Schema Specification

All candidate and active benchmark manifests use the unified schema below. Extra columns are optional and backward-compatible with the Phase 11 evaluator:

| Column Name | Type | Required? | Valid Values / Constraints | Description |
|:---|:---:|:---:|:---|:---|
| `pair_id` | String | **Yes** | Unique, e.g. `cand_pair_0001`, `pair_0001` | Primary key for evaluation |
| `image_a` | String | **Yes** | Relative path, whitelist ext (`.jpg`, `.png`, etc.) | Path relative to benchmark images directory |
| `image_b` | String | **Yes** | Relative path, whitelist ext | Path relative to benchmark images directory |
| `source_a` | String | **Yes** | e.g. `real_src_0001` | Physical scene source identifier for Image A |
| `source_b` | String | **Yes** | e.g. `real_src_0001` | Physical scene source identifier for Image B |
| `related` | Int/Str | **Yes** | `1`, `0`, `unknown` | Ground-truth relatedness |
| `ground_truth_direction` | String | **Yes** | `A_TO_B`, `B_TO_A`, `NONE`, `UNCERTAIN` | Direction of derivation |
| `applied_transformations` | String | Optional | Semicolon or comma-separated tags | e.g. `crop;jpeg;resize` or `none` |
| `split` | String | Optional | `benchmark_test`, `benchmark_val`, etc. | Partition assignment |
| `provenance_status` | String | Optional | `contributor_confirmed`, `controlled_synthetic`, `hard_negative_visual_match`, `candidate_unverified` | Audit classification |
| `collection_method` | String | Optional | `camera_manual_edit`, `messaging_export`, `web_crawl`, `synthetic_script` | Ingestion technique |
| `verification_status` | String | Optional | `verified`, `pending_review`, `unverified` | Quality gate status |
| `uncertainty_flag` | Int | Optional | `0` (certain), `1` (uncertain / quarantined) | Flag for manual inspection |
| `notes` | String | Optional | Free text | Lineage descriptions, camera models, etc. |

---

## 4. Leakage-Resistant Source-Group Splitting Rules

```
  Source real_src_0001 ──┬── Image A (Original)
                         ├── Image B (Crop)          ─── ALL DERIVATIVES ASSIGNED TO:
                         └── Image C (Resize)             [ Split: benchmark_test ]

  Source real_src_0002 ──┬── Image D (Original)      ─── ALL DERIVATIVES ASSIGNED TO:
                         └── Image E (Blur)               [ Split: benchmark_val ]
```

### Splitting Invariants:
1. **Source-Atomic Partitions:**
   - Partitioning is performed strictly on `{source_id}` sets, never by randomly shuffling pair rows.
   - If `real_src_0001` is placed in `benchmark_test`, all derivatives, crops, and pairs derived from `real_src_0001` must reside exclusively in `benchmark_test`.
2. **Negative Distractor Source Containment:**
   - For negative pairs linking two distinct sources (`real_src_0001` vs `real_src_0002`), both sources must belong to the same split. A negative pair must never cross split boundaries.
3. **SHA-256 Content Deduplication:**
   - Exact duplicate files (identical SHA-256 hashes under different filenames) must be detected and consolidated.
4. **Leakage Audit Verification:**
   - Automated via `src/qa_benchmark_dataset.py`, which computes:
     $$\text{Overlap}(S_i, S_j) = \text{Sources}(S_i) \cap \text{Sources}(S_j)$$
     and raises a critical validation error if $\text{Overlap} \ne \emptyset$.

---

## 5. Contributor Annotation & Ingestion Workflow

To add new photographs to the TRACE AI expansion pool, follow this 5-step checklist:

1. **Place Raw Images in Expansion Workspace:**
   - Copy new images to `data/real_world_benchmark/expansion_workspace/candidate_images/`.
   - Never place unverified images directly into production `controlled_transforms/images/`.
2. **Assign Unique Source Groups:**
   - Each distinct physical capture must receive a new source ID (e.g. `real_src_0025`, `real_src_0026`).
   - If two photos depict the same scene from different angles, assign distinct source IDs and label them as hard negative candidates (`related=0`, `notes="hard_negative_visual_match"`).
3. **Document Physical Lineage in Candidate Manifest:**
   - Use `data/real_world_benchmark/expansion_workspace/candidates_manifest.template.csv`.
   - Record the exact editing tool, camera model, and applied transformations in `notes`.
4. **Run Pre-Flight QA Audit:**
   ```powershell
   python src/qa_benchmark_dataset.py `
     --manifest data/real_world_benchmark/expansion_workspace/candidates_manifest.template.csv `
     --images-dir data/real_world_benchmark/expansion_workspace/candidate_images
   ```
5. **Peer Review & Promotion:**
   - A candidate pair may only be promoted to the active benchmark after:
     a) Physical lineage is contributor-verified.
     b) File headers pass PIL verify.
     c) Zero source-group leakage is verified against existing benchmark splits.

---

## 6. Recommended Minimum Evidence Thresholds Before Evaluation

Before any general accuracy claims can be made:

| Benchmark Dimension | Current Benchmark ($N=35$) | Minimum Threshold for Evaluation | Generalization Requirement |
|:---|:---:|:---:|:---|
| **Unique Physical Scenes** | 24 | $\ge 50$ | Multi-device, diverse indoor/outdoor environments |
| **Independent Real Positive Groups** | **1** | $\ge 25$ | Diverse sensor resolutions, optics, and compression |
| **Controlled Multi-Source Bases** | **1** | $\ge 10$ | Robustness across different image frequencies |
| **Hard Negative Visual Pairs** | **0** | $\ge 30$ | Specificity against visually similar scenes |
| **Abstention & Coverage Evaluation** | Disaggregated | Disaggregated | Separate reporting of confident accuracy vs abstention rate |
