# Project Changelog: Engineering Audit & Performance Hardening

**Project:** Explainable Image Provenance Analysis Via Graph Encoding With Vision Transformer In Forensics  
**Date:** October 14, 2026  
**Version:** 2.5.0-Transformation-Audited  

---

### [2.5.0] — 2026-10-14 (Complete Transformation Audit & Full-Stack Hardening)
#### Added
- **Comprehensive 25-Transformation Benchmark Suite (`scratch/run_rigorous_transformation_audit.py`):**
  - Generated controlled descendants across 14 transformation families and 41 distinct subtypes using permitted validation source images (`src_0003`, `src_0008`, `src_0017`).
  - Evaluated 150 live forward and reverse inference pairs to verify lineage detection, direction prediction, root identification, and exact swap consistency.
  - Tested 5 completely unrelated image pairs and 4 sibling triad graphs.
- **All-Category Forensic Rationale Engine (`src/xai_explainer.py`):**
  - Expanded `explain_edge_prediction` to generate tailored empirical forensic evidence for all 12 transformation families (photometric luminance deltas, dynamic range std, JPEG blockiness, high-pass noise residuals, Laplacian edge intensity, projective warping).
- **Independent Pair XAI Engine (`backend/app/services/xai_engine.py`):**
  - Implemented `generate_xai_report_for_independent_pair` to generate comparative noise residual difference maps, spatial metrics, and explicit Divergence Rationale for disconnected and unrelated image pairs.
- **Edge Case Feature Robustness Suite (`scratch/test_feature_pipeline_edge_cases.py`):**
  - Verified 57-D, 115-D, and 172-D feature vector extraction across grayscale (1-channel), RGBA (4-channel), tiny (16x16), and extreme aspect ratio (800x100) images with zero NaNs or infinite values.

#### Fixed
- **BUG 1 — Provenance Graph Edges & Disconnected Forest Display:**
  - Resolved blank/overlapping canvas bug when analyzing disconnected components (0 tree edges). Added fallback to Cytoscape concentric/grid layout and propagated `rejected_pairs` metadata.
- **BUG 2 — Missing XAI Forensic Evidence Staged:**
  - Fixed `"No XAI Forensic Evidence Staged"` blocker. Persisted independent XAI records in SQLite database and decoupled frontend page guards in `frontend/src/pages/ExplainableAI.jsx`.
- **UI Session Loss on Browser Refresh:**
  - Added `localStorage` mirroring of `currentAnalysisId` in `frontend/src/App.jsx`, preventing silent erasure of analysis results upon page reload.
- **Model Deserialization Robustness:**
  - Standardized `_load_serialized_model` in `backend/app/services/model_manager.py` to seamlessly handle `joblib` binary formats without version crashes.

#### Verified
- **Full Regression Suite**: All 39 unit tests in `tests/` and 7 API tests pass in 43.19s with 0 failures.
- **Frontend Production Build**: `npm run build` completed in 21.25s with 1941 transformed modules and zero compilation warnings.
- **Direction Swap Invariance**: 150/150 (100.00%) swap consistency confirmed with maximum delta of 0.000000.

---

### [2.4.0] — 2026-10-14 (Production Readiness & Technical Audit)
#### Added
- **Isolated Technical Audit Suite (`scratch/execute_full_technical_audit.py`):** Standalone script to audit models, dataset splits, quarantine boundaries, and graph reconstruction metrics.
- **Conservative Candidate-Edge Selection Investigation (`scratch/conservative_edge_selection.py`):** Empirical sweep confirming that candidate edge retention with soft uncertainty flagging preserves true-edge recall better than hard threshold gating.
- **Comprehensive Audit Reports (`reports/`):** Full suite of audit and validation artifacts documenting model provenance, dataset integrity, graph reconstruction, security, and test results.

#### Verified & Maintained
- **Adaptive Kruskal MST Pipeline:** Preserved undirected MST with adaptive cycle ambiguity penalty ($\lambda=0.05, \tau_{\text{cycle}}=0.15$) as the official production graph reconstruction algorithm.
- **Antisymmetric Direction MLP:** Verified 100% swap consistency and 89.47% validation accuracy in live API inference.
- **Multi-Factor Ancestral Invariance Root Selection:** Verified 3/3 (100.0%) root accuracy across all validation graphs.
- **Quarantine Isolation:** Confirmed test graphs `graph_0001`, `graph_0004`, `graph_0009`, `graph_0021` remain strictly unaccessed.

---

### [2.3.0] — 2026-10-13 (Direction Model & Root Selection Optimization)
#### Added
- **`AntisymmetricDirectionMLP` Integration:** Integrated neural network with built-in mathematical swap anti-symmetry $f(A, B) = -f(B, A)$ into `backend/app/services/model_manager.py`.
- **Multi-Factor Ancestral Invariance Root Scoring:** Implemented tournament dominance flow, zero directional losses, and hierarchical DAG path depth in `backend/app/services/provenance_engine.py`.
- **E2E Smoke Test Suite (`scratch/smoke_test_e2e.py`):** 7-phase end-to-end integration test validating upload, pairwise inference, graph generation, XAI explanation, and report export.

#### Fixed
- **Root Identification Failure:** Fixed root accuracy failure (previously 0/3) under Kruskal MST orientations, lifting it to 3/3 (100%).
- **Direction Flip Inconsistency:** Eliminated 31.58% direction contradiction in legacy ExtraTrees classifier.

---

### [2.2.0] — 2026-10-12 (Graph Cycle Prevention & Database Hardening)
#### Added
- **Adaptive Cycle Ambiguity Penalty:** Added graph bottleneck cycle ambiguity penalty to `provenance_engine.py` to discourage transitive shortcuts.
- **Cycle Safeguard:** Added cycle reversal safeguard in directed forest construction to enforce strict DAG topology.
- **SQLite Spatial Metrics Schema Migration:** Added automatic column detection and fallback serialization in `backend/app/database/db.py`.

#### Fixed
- **Database Schema Mismatch:** Resolved potential `OperationalError` when accessing legacy database instances without spatial metrics.

---

### [2.1.0] — 2026-10-06 (Frontend Modernization & Explainability)
#### Added
- **Cytoscape Provenance Graph Viewer:** Interactive graph layout with generational depth highlighting, root badge, and edge confidence tooltips.
- **Explainable AI (XAI) Panel:** Visual Transformer semantic alignment breakdown, noise residual variance, and spatial distortion metrics (MSE, PSNR, SSIM).
- **Forensic Report Export:** Automated HTML and JSON report generator with embedded provenance tree visualizations.
