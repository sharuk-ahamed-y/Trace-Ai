# TRACE AI — Phase 4 Model Training & Evaluation Report

**Date:** 2026-10-18 12:16:39  
**Status:** TRAINING & EVALUATION COMPLETE  
**Output Directory:** `phase4_v1/` (`models/phase4_v1/`)  
**Seed:** 42  
**Execution Platform:** Windows x64 (CPU Only, PyTorch 2.14.1+cpu)  
**Total Runtime:** 16.6 seconds  

---

## 1. Executive Summary

Phase 4 completed the full training, calibration, and held-out test evaluation of TRACE AI's three core forensic models:
1. **Relationship Classifier:** Binary classifier predicting whether two images originate from the same source image.
2. **Multi-Label Transformation Classifier:** Predicts all active transformations among the 12 supported classes.
3. **AntiSymmetric Direction Classifier:** Predicts causal edge direction (`A_TO_B`, `B_TO_A`, or abstains as `UNCERTAIN`) with **100.0% mathematical swap consistency**.

All models were trained strictly using CPU-friendly MLPs operating on normalized 57-D forensic descriptors (115-D symmetric and 172-D antisymmetric pair vectors). Feature scalers were fit **strictly on Training data** ($N=1,544$), and decision thresholds were tuned **strictly on Validation data** ($N=384$). Held-out test evaluation ($N=324$) was performed exactly once.

---

## 2. Dataset Partition & Safeguards

| Split | Sources | Directed Pairs | Related (Pos) | Unrelated (Neg) | Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | `source_0001`–`source_0070` | 1,544 | 772 (50.0%) | 772 (50.0%) | Scaler fitting & model parameter updates |
| **Validation** | `source_0071`–`source_0085` | 384 | 192 (50.0%) | 192 (50.0%) | Early stopping & threshold calibration |
| **Test** | `source_0086`–`source_0100` | 324 | 162 (50.0%) | 162 (50.0%) | Single-pass held-out performance evaluation |

- **Cross-Split Source Overlap:** **0.00%** (zero source identity leakage).
- **Pair Key Overlap:** **0.00%** (no swapped or unswapped pair leakage).
- **Pretrained ViT Weights:** **0 MB downloaded** (forensic feature descriptor training).

---

## 3. Component 1: Relationship Classifier

- **Input Dimension:** 115-D ($|f_A - f_B|$ [57], $f_A \odot f_B$ [57], $\cos(f_A, f_B)$ [1])
- **Architecture:** MLP ($115 \to 64 \to 32 \to 1$) with BatchNorm, ReLU, and Dropout.
- **Calibrated Decision Threshold (Val):** `0.62`
- **Validation ROC-AUC:** `0.9927`

### Held-Out Test Set Performance (N=324)

| Metric | Test Value | Description |
| :--- | :---: | :--- |
| **Precision** | **0.9877** | Exactness of positive relationship predictions |
| **Recall** | **0.9877** | Sensitivity to related image pairs |
| **F1 Score** | **0.9877** | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.9990** | Area under the ROC curve |
| **PR-AUC (AP)** | **0.9990** | Average precision across thresholds |
| **False Positive Rate (FPR)** | **0.0123** | False alarm rate on unrelated distractor pairs |

### Test Confusion Matrix
| | Predicted Negative | Predicted Positive |
| :--- | :---: | :---: |
| **Actual Negative** | 160 (TN) | 2 (FP) |
| **Actual Positive** | 2 (FN) | 160 (TP) |

---

## 4. Component 2: Multi-Label Transformation Classifier

- **Input Dimension:** 115-D symmetric pairwise features.
- **Architecture:** Multi-label MLP ($115 \to 128 \to 64 \to 12$).
- **Loss:** `BCEWithLogitsLoss` with positive class weighting (`pos_weight`) to handle class imbalance.
- **Overall Held-Out Test Metrics:**
  - **Macro F1:** **0.6197**
  - **Micro F1:** **0.6119**

### Per-Class Performance on Held-Out Test (N=162 related pairs)

| Class | Calibrated Threshold | Precision | Recall | F1 Score | Test Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `crop` | 0.62 | 1.0000 | 0.5000 | 0.6667 | 4 |
| `resize` | 0.83 | 0.6000 | 0.6923 | 0.6429 | 26 |
| `jpeg` | 0.61 | 0.4211 | 0.8000 | 0.5517 | 20 |
| `blur` | 0.56 | 0.6667 | 0.4000 | 0.5000 | 20 |
| `brightness` | 0.59 | 0.5000 | 0.4444 | 0.4706 | 18 |
| `contrast` | 0.52 | 0.4667 | 0.7000 | 0.5600 | 20 |
| `sharpen` | 0.66 | 0.5000 | 0.7500 | 0.6000 | 16 |
| `rotation` | 0.85 | 0.5000 | 0.6667 | 0.5714 | 12 |
| `color` | 0.66 | 0.2500 | 0.6000 | 0.3529 | 10 |
| `noise` | 0.77 | 1.0000 | 0.8750 | 0.9333 | 16 |
| `perspective` | 0.71 | 0.9000 | 0.7500 | 0.8182 | 24 |
| `quality_degradation` | 0.66 | 0.7143 | 0.8333 | 0.7692 | 12 |

---

## 5. Component 3: AntiSymmetric Direction Classifier

- **Input Dimension:** 172-D ($f_A$ [57], $f_B$ [57], $f_A - f_B$ [57], $\cos(f_A, f_B)$ [1])
- **Architecture:** `AntiSymmetricDirectionMLP` with formulation:
  $$\text{logit}(A, B) = \frac{1}{2} \left( g(x_{AB}) - g(x_{BA}) \right)$$
- **Swap Consistency:** **100.0%** (Max numerical swap discrepancy: `1.2e-07`).
  - $P(A \to B) + P(B \to A) \equiv 1.0000000$ strictly by construction.

### Test Performance & Abstention (UNCERTAIN)

- **Raw Direction Accuracy (forced binary):** **0.7531**
- **Calibrated Uncertainty Threshold ($\delta$):** `0.04`
- **Test Coverage (confident decisions):** **91.4%**
- **Confident Direction Accuracy:** **0.7838**

### Coverage vs. Accuracy Trade-Off for Abstention

| Confidence Margin ($\delta$) | Coverage (%) | Accuracy on Confident Pairs |
| :---: | :---: | :---: |
| 0.0 | 100.0% | 0.7500 |
| 0.04 | 81.2% | 0.7564 |
| 0.08 | 77.1% | 0.7568 |
| 0.12 | 71.9% | 0.7536 |
| 0.16 | 70.8% | 0.7500 |
| 0.2 | 65.6% | 0.8095 |
| 0.24 | 60.4% | 0.7931 |
| 0.28 | 57.3% | 0.7818 |
| 0.32 | 50.0% | 0.7917 |
| 0.36 | 49.0% | 0.8085 |
| 0.4 | 46.9% | 0.8222 |
| 0.44 | 37.5% | 0.8333 |

---

## 6. Checkpoints & Saved Artifacts

All model checkpoints and configuration files are preserved in:
`models/phase4_v1/`

```
models/phase4_v1/
├── metadata.json                 # Complete training hyperparameters and evaluation metrics
├── relationship_model.pt         # PyTorch weights + architecture metadata for Relationship
├── transformation_model.pt       # PyTorch weights + architecture metadata for Transformations
├── direction_model.pt            # PyTorch weights + architecture metadata for Direction
├── scaler_115.joblib             # Fitted StandardScaler for 115-D pairwise features
└── scaler_172.joblib             # Fitted StandardScaler for 172-D directional features
```

---

## 7. Known Limitations & Production Guidance

1. **Synthetic Foundation:** Trained on the synthetic isolated benchmark `SIPGD_v2`. While ground truth is mathematically verified, performance on real-world multi-compressed WhatsApp images requires fine-tuning.
2. **Deep Features Omitted:** Trained exclusively on 57-D forensic statistics (physical, noise residual, and histogram features). Vision Transformer (ViT-B/16) deep visual embeddings were bypassed to keep resource usage zero-download.
3. **Backend Integration:** Models are saved in standard PyTorch and Joblib formats. In Phase 5, they can be plugged into `backend/app/services/model_manager.py` without modifying API contracts or frontend components.
