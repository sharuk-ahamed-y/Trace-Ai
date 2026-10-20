# REAL-WORLD BENCHMARK EVALUATION REPORT

**TRACE AI: Image Provenance Using Graphs**  
**Status:** EXECUTED ON AUTHORIZED REAL-WORLD DATASET  
**Total Pairs Evaluated:** 1 (Positive: 1, Negative: 0)  
**Decision Threshold ($\\tau$):** 0.62 (Validation-tuned)  
**Direction Uncertainty Delta ($\\Delta$):** 0.04  

## 1. Pairwise Relationship Metrics

- **Precision:** 1.0000
- **Recall:** 1.0000
- **F1-Score:** 1.0000
- **False Positive Rate (FPR):** 0.0000
- **Brier Calibration Score:** 0.0000
- **Expected Calibration Error (ECE, 10 bins):** 0.0000
- **Confusion Matrix:** TN=0, FP=0, FN=0, TP=1

## 2. Direction Classification Metrics

- **Evaluated Related Pairs:** 1
- **Raw Direction Accuracy:** 0.0000
- **Abstention Coverage:** 1.0000 (100.0%)
- **Confident-Prediction Accuracy:** 0.0000

## 3. Transformation Multi-Label Performance

- **Macro-F1 (Supported Classes with Support > 0):** 0.0000

| Transformation Class | Precision | Recall | F1-Score | Real-World Support |
|:---|:---:|:---:|:---:|:---:|
| `crop` | 0.0 | 0.0 | 0.0 | 1 |
| `resize` | 0.0 | 0.0 | 0.0 | 0 |
| `jpeg` | 0.0 | 0.0 | 0.0 | 0 |
| `blur` | 0.0 | 0.0 | 0.0 | 0 |
| `brightness` | 0.0 | 0.0 | 0.0 | 0 |
| `contrast` | 0.0 | 0.0 | 0.0 | 0 |
| `sharpen` | 0.0 | 0.0 | 0.0 | 0 |
| `rotation` | 0.0 | 0.0 | 0.0 | 0 |
| `color` | 0.0 | 0.0 | 0.0 | 0 |
| `noise` | 0.0 | 0.0 | 0.0 | 0 |
| `perspective` | 0.0 | 0.0 | 0.0 | 0 |
| `quality_degradation` | 0.0 | 0.0 | 0.0 | 0 |

## 4. Benchmark Isolation & Limitations

1. **Strict Metric Separation:** These real-world results are recorded strictly in this report and must never be merged or averaged with synthetic held-out test metrics (`SIPGD_v2`).
2. **Domain Specificity:** Results reflect the camera sensors, lighting conditions, and transmission channels included in this benchmark.
3. **Calibration Notice:** Probability estimates are direct sigmoid outputs; ECE and Brier score reflect uncalibrated network confidence.
