# TRACE AI — REAL-WORLD DATASET QA & INTEGRITY REPORT

**Target Manifest:** `data\real_world_benchmark\controlled_transforms\manifest.csv`  
**Target Images Directory:** `data\real_world_benchmark\controlled_transforms\images`  
**Audit Status:** PASSED  
**Total Pairs Audited:** 35  
**Total Unique Images:** 35  
**Unique Independent Physical Scenes:** 24  

> [!NOTE]
> **Dataset QA PASSED:** Schema, relative path containment, image decode verify, label semantics, and split independence are 100% verified.

## 1. Disaggregated Category Breakdown

| Subgroup Category | Pair Count | Proportion | Evaluation Purpose |
|:---|:---:|:---:|:---|
| **Contributor-Confirmed Real Pairs** | 1 | 2.9% | Genuine unconstrained photograph captures with verified lineage |
| **Controlled Transformations** | 10 | 28.6% | Controlled synthetic canonical transformations from authorized bases |
| **Cross-Scene Negative Distractors** | 24 | 68.6% | Unrelated photographs from confirmed distinct physical scenes |
| **Candidate Unverified / Quarantined** | 0 | 0.0% | Under review; quarantined from benchmark test scoring |
| **Unclassified Rows** | 0 | 0.0% | Error: Rows failing recognized category assignment |

## 2. Source Group & Leakage Audit

- **Total Independent Sources:** 24
- **Sources with Positive Pairs:** 1
- **Sources with Negative Pairs:** 24
- **Sources Appearing in Both Positive & Negative:** 1 (['real_src_0001'])

### Evaluation Split Composition

| Split Name | Pair Count | Source Count | Source IDs |
|:---|:---:|:---:|:---|
| `benchmark_test` | 35 | 24 | real_src_0001, real_src_0002, real_src_0003, real_src_0004, real_src_0005... (+19 more) |

> [!NOTE]
> **Split Independence Verified:** Zero source-group leakage across evaluation splits.

## 3. Duplicate Image Hash Analysis (SHA-256)

Zero identical image content duplicates detected across all referenced image paths.

## 4. Applied Transformation Inventory

| Transformation Tag | Frequency |
|:---|:---:|
| `crop` | 2 |
| `resize` | 1 |
| `jpeg` | 1 |
| `blur` | 1 |
| `brightness` | 1 |
| `contrast` | 1 |
| `noise` | 1 |
| `perspective` | 1 |
| `sharpen` | 1 |
| `color` | 1 |

## 5. Audit Errors & Warnings

No critical errors encountered.

### Warnings & Review Items

- ⚠️ Sources appearing in BOTH positive and negative pairs: ['real_src_0001'] (Acceptable for benchmark distractors, but must be strictly isolated by split if training models).

