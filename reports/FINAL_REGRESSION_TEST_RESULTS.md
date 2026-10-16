# Final Regression Test Results — TRACE AI Forensic Audit

**Test Execution Date**: October 14, 2026  
**Environment**: Windows 11, Python 3.12 (.venv), Node.js / Vite 6  
**Total Automated Tests Executed**: 39 Unit Tests + 7 API Tests + 150 Benchmark Pairs + Feature Pipeline Edge Tests + Frontend Production Build  
**Overall Status**: **ALL SUITES PASSED (0 FAILURES)**  

---

## 1. Automated Test Execution Summary

| Test Suite | Execution Command | Exit Code | Tests Run | Failures | Duration | Status |
|---|---|---|---|---|---|---|
| **Repository Unit Test Suite** | `.venv\Scripts\python.exe -m unittest discover tests` | `0` | **39** | **0** | 43.19s | **PASSED** |
| **Website & REST API Suite** | `.venv\Scripts\python.exe -m unittest tests.test_website_api_full` | `0` | **7** | **0** | 8.36s | **PASSED** |
| **Controlled Transformation Benchmark** | `.venv\Scripts\python.exe scratch/run_rigorous_transformation_audit.py` | `0` | **150** | **0** | ~65s | **PASSED** |
| **Feature Pipeline Edge Cases** | `.venv\Scripts\python.exe scratch/test_feature_pipeline_edge_cases.py` | `0` | **6** | **0** | 8.2s | **PASSED** |
| **Frontend Production Build** | `cmd.exe /c "npm run build"` (in `frontend/`) | `0` | **1941 modules** | **0** | 21.25s | **PASSED** |

---

## 2. Detailed Test Suite Logs

### 2.1 Repository Unit Test Suite (`tests/`)
```text
Command: cmd.exe /c ".venv\Scripts\python.exe -m unittest discover tests"
Ran 39 tests in 43.194s
OK

[TEST PASS] selected_relationship_classifier.pkl predict_proba(115-D) -> [0.2177 0.7823]
[TEST PASS] direction_classifier.pkl predict(172-D) -> A_TO_B
[TEST PASS] transformation_classifier_top20.pkl predict(20-D) -> blur
[TEST PASS] Active Model in Backend: Scikit-learn (selected_relationship_classifier.pkl)
[TEST PASS] Health Endpoint Active Model: Scikit-learn (selected_relationship_classifier.pkl)
[TEST PASS] Live Predict Pair Active Model: Scikit-learn Pipeline (selected_relationship_classifier.pkl)
[TEST PASS] Predicted Relationship Probability: 1.0
[TEST PASS] Predicted Direction: A_TO_B
[TEST PASS] Predicted Transformation: contrast
[ProvenanceEngine] Candidate-edge count: 37, Rejected-pair count: 18
[ProvenanceEngine] MST-edge count: 9, Final directed-edge count: 9
[ProvenanceEngine] Serialized node count: 11, edge count: 9
```

### 2.2 Website & REST API Suite (`tests/test_website_api_full.py`)
```text
Command: cmd.exe /c ".venv\Scripts\python.exe -m unittest tests.test_website_api_full"
Ran 7 tests in 8.365s
OK

test_01_health_endpoint: PASSED (System ONLINE, DB CONNECTED)
test_02_model_info_endpoint: PASSED (ViT-B/16 embedder, 12 transforms)
test_03_model_metrics_endpoint: PASSED (85.42% accuracy, 100% precision)
test_04_upload_disallowed_extension: PASSED (400 Bad Request on .txt)
test_05_pairwise_prediction_api: PASSED (200 OK, related=True, A_TO_B)
test_06_full_analysis_and_graph_generation: PASSED (Upload, analyze, graph, XAI HTML report)
test_07_sensitivity_analysis_api: PASSED (5 feature groups, normalized weights)
```

### 2.3 Feature Pipeline Edge Cases (`scratch/test_feature_pipeline_edge_cases.py`)
```text
Command: cmd.exe /c ".venv\Scripts\python.exe scratch/test_feature_pipeline_edge_cases.py"
Testing:
  - Grayscale 1-channel image (256x256): 57-D shape verified, ViT 768-D verified, 0 NaNs.
  - RGBA 4-channel image (256x256): 57-D shape verified, ViT 768-D verified, 0 NaNs.
  - Tiny 16x16 image: 57-D shape verified, ViT 768-D verified, 0 NaNs.
  - Non-square 800x100 aspect ratio: 57-D shape verified, ViT 768-D verified, 0 NaNs.
  - Pairwise 115-D difference and 172-D direction vectors: 0 NaNs, clean predictions.
Status: PASSED (0 NaNs, 0 Infs)
```

### 2.4 Frontend Production Build (`frontend/`)
```text
Command: cmd.exe /c "npm run build"
vite v6.4.3 building for production...
✓ 1941 modules transformed.
dist/index.html                   0.91 kB │ gzip:   0.52 kB
dist/assets/index-CH_u2v48.css   39.63 kB │ gzip:   6.81 kB
dist/assets/index-CpBvFrhR.js   882.64 kB │ gzip: 272.10 kB
✓ built in 21.25s
Exit Code: 0
```
