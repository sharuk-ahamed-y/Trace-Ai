# Root Cause Analysis: Bug 2 — Explainable AI Evidence Missing

## 1. Executive Summary

- **Reported Bug**: After running provenance analysis, the Explainable AI (XAI) page showed:
  `"No XAI Forensic Evidence Staged"`
  The page offered a "Run Provenance Analysis" button but did not display the forensic evidence for the analysis that had already been executed.
- **Root Causes Identified**:
  1. **Edge-Restricted XAI Pipeline**: In `backend/app/api/endpoints.py`, XAI report generation was conditioned solely on `for edge in results["edges"]:`. When an analysis concluded with 0 edges (such as for independent, unrelated, or disconnected image pairs), the loop never ran. `xai_results` remained an empty array `[]`.
  2. **Overloaded Frontend Guard**: In `frontend/src/components/XaiPanel.jsx`, the guard condition was:
     ```javascript
     if (!analysisData || xaiResults.length === 0)
     ```
     This conflated two completely different states:
     - State A: No analysis was ever started.
     - State B: An analysis completed, but no pairwise derivation links met the confidence threshold.
     Because `xaiResults.length === 0`, `XaiPanel.jsx` rendered the empty initial CTA, misleading the user into thinking no analysis had run.
  3. **Volatile In-Memory React State**: In `frontend/src/App.jsx`, `analysisData` was stored in transient `useState(null)` with no persistence mechanism. Any page refresh or tab re-opening wiped `analysisData`, discarding the active analysis session.
  4. **Missing Sensitivity Fallback**: `/api/analysis/{id}/sensitivity` failed to compute feature group sensitivity when `edges` was empty, returning `{"message": "No suitable edges..."}` instead of evaluating the uploaded pair.

---

## 2. Tracing the Complete Data Flow

1. **Analysis Triggered**:
   - `UploadZone.jsx` calls POST `/api/analyze` with session `analysis_id`.
   - `run_provenance_analysis()` evaluates the pairwise ML relationship.
2. **XAI Engine Invocation (Before Fix)**:
   ```python
   # Previous faulty code in endpoints.py:
   for edge in results["edges"]:
       src_node = nodes_map.get(edge["source"])
       tgt_node = nodes_map.get(edge["target"])
       ...
       xai_data = generate_xai_report_for_edge(...)
       xai_results.append(xai_data)
   ```
   When `results["edges"]` was empty, `xai_results` remained `[]`.
3. **Database Persistence**:
   - `save_analysis()` wrote `xai_results: []` to SQLite.
4. **API Response**:
   - Returned `"xai_results": []`.
5. **Frontend Rendering (`XaiPanel.jsx`)**:
   - Guard `if (!analysisData || xaiResults.length === 0)` triggered.
   - Screen rendered "No XAI Forensic Evidence Staged", ignoring `analysisData.analysis_id`.

---

## 3. Files & Functions Modified

| File | Function / Component | Changes Made |
| :--- | :--- | :--- |
| `backend/app/services/xai_engine.py` | `generate_xai_report_for_independent_pair()` | Added function to generate noise residual difference heatmaps, spatial metrics (MSE, PSNR, SSIM), and empirical findings explaining why an image pair was classified as independent/unrelated. |
| `backend/app/api/endpoints.py` | `analyze_provenance()`, `get_analysis_feature_sensitivity()` | Added independent pair XAI generation when `results["edges"]` is empty; added image-pair fallback to `get_analysis_feature_sensitivity()`. |
| `backend/app/database/db.py` | `save_analysis()`, `get_analysis_by_id()` | Preserved `is_independent`, `relationship`, `transformation`, and `confidence` fields during SQLite persistence and deserialization. |
| `frontend/src/App.jsx` | `App()` | Added `localStorage` synchronization for `trace_ai_active_analysis_id` and `trace_ai_active_tab`; restored analysis via `/api/analysis/{id}` on page refresh. |
| `frontend/src/components/XaiPanel.jsx` | `XaiPanel()` | Decoupled `!analysisData` from `xaiResults.length === 0`; added independent pair badge and rationale rendering; bound dynamic `feature_groups` sensitivity attribution bars. |
| `frontend/src/components/HistoryView.jsx` | `handleOpenAnalysis()` | Standardized field extraction from both unified and nested analysis schemas. |

---

## 4. Before & After Comparisons

### A. Raw API Response (`/api/analysis/{analysis_id}/explanation`)

**Before Fix**:
```json
{
  "analysis_id": "anl_20261014_113320_042850",
  "xai_results": []
}
```

**After Fix**:
```json
{
  "analysis_id": "anl_20261014_122633_b1d400",
  "xai_results": [
    {
      "source_id": "ojj",
      "target_id": "WhatsApp_Image_2026-10-13_at_1.31.11_AM",
      "is_independent": true,
      "relationship": "INDEPENDENT / UNRELATED",
      "transformation": "NONE (INDEPENDENT)",
      "confidence": 0.0979,
      "heatmap_url": "/data/visualizations/heatmap_anl_20261014_122633_b1d400_ojj_WhatsApp_Image_2026-10-13_at_1.31.11_AM.png",
      "spatial_metrics": {
        "mse": 2368.7,
        "psnr_db": 14.39,
        "ssim_est": 0.9637,
        "residual_diff_max": 255.0,
        "residual_diff_mean": 11.23
      },
      "evidence": {
        "confidence": 0.0979,
        "is_independent": true,
        "relationship": "INDEPENDENT / UNRELATED",
        "summary": "Forensic Independence: 'ojj' and 'WhatsApp_Image_2026-10-13_at_1.31.11_AM' show no derivation link (P(rel) = 9.8%).",
        "empirical_findings": [
          "ViT CLS Semantic Cosine Similarity: 0.6538 (Insufficient semantic feature overlap for derivation link).",
          "Relationship Classifier Confidence: 9.8% (Did not satisfy derivation confidence threshold).",
          "Spatial Noise Residual Divergence: MSE = 2368.7, PSNR = 14.39 dB, SSIM = 0.9637.",
          "Spectral Comparison: JPEG Quality 100.0% vs 100.0%; Noise Residual Std 22.84 vs 18.91.",
          "Dimension Profile: 900x1600 (147.0 KB) vs 1112x1308 (126.2 KB).",
          "Forensic Conclusion: Images classified as INDEPENDENT / UNRELATED. No directed derivation edge added to forest."
        ]
      }
    }
  ]
}
```

### B. Sensitivity Attribution Response (`/api/analysis/{analysis_id}/sensitivity`)

**Before Fix**:
```json
{
  "message": "No suitable edges for feature sensitivity computation."
}
```

**After Fix**:
```json
{
  "base_prediction": {
    "related": false,
    "related_confidence": 0.0979,
    "direction": "NONE",
    "direction_confidence": 0.9021,
    "cosine_similarity": 0.6538
  },
  "feature_groups": [
    {"group": "Visual Transformer Latent", "importance": 0.2615, "normalized_weight": 0.4187},
    {"group": "High-Pass Noise Residual & JPEG", "importance": 0.0959, "normalized_weight": 0.1536},
    {"group": "Edge & Frequency Gradients", "importance": 0.1132, "normalized_weight": 0.1813},
    {"group": "Color Space Distribution", "importance": 0.0768, "normalized_weight": 0.123},
    {"group": "Physical Image Geometry", "importance": 0.0771, "normalized_weight": 0.1235}
  ]
}
```

---

## 5. UI Presentation & Resilience Verification

1. **Connected Derivation Edge**:
   - Renders "Empirical Forensic Rationale" with green theme.
   - Shows parent $\to$ child lineage, transformation, and noise residual heatmap.
2. **Disconnected / Independent Pair**:
   - Renders "Independent Pair Forensic Rationale" with cyan/amber theme.
   - Shows comparative evaluation $A \leftrightarrow B$ and relationship probability ($9.8\%$ below threshold).
   - Explains noise residual divergence and sensor independence.
3. **Empty / No Active Analysis**:
   - If no analysis is loaded, displays clean call-to-action to upload or run benchmark.
4. **Page Refresh Persistence**:
   - `trace_ai_active_analysis_id` stored in `localStorage`.
   - On refresh, `App.jsx` queries `/api/analysis/${savedId}`, restoring `analysisData`, `nodes`, `edges`, and `xai_results` seamlessly.
