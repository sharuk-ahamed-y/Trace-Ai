# Frontend and Backend Integration Audit — TRACE AI Forensic Audit

**Application**: TRACE AI Full-Stack Web Application  
**Frontend Stack**: React 18, Vite 6, TailwindCSS, Cytoscape.js, Lucide Icons  
**Backend Stack**: FastAPI, Uvicorn, SQLite 3, SQLAlchemy ORM, Scikit-learn, PyTorch  
**Live Daemons Audited**:
- Backend API daemon running on port `8000` (task-1191)
- Frontend Vite development server running on port `5173` (task-1200)

---

## 1. Executive Summary

A complete, systematic integration audit was performed across all frontend components, backend REST controllers, database schema tables, and static data endpoints.

All **12 REST API endpoints** were tested with real image files, multi-part form uploads, disconnected image pairs, and corrupt requests. The frontend production bundle was compiled and verified with zero errors (`dist/index.html`, `dist/assets/index-*.js`, `dist/assets/index-*.css`).

---

## 2. API Endpoint Verification Matrix

| Endpoint | Method | Tested Input | Status Code | Verification Details |
|---|---|---|---|---|
| `/api/health` | GET | None | `200 OK` | Verifies DB connection, PyTorch/ViT status, Scikit-learn pipeline name. |
| `/api/model-info` | GET | None | `200 OK` | Returns active models, 115-D feature specification, 12 supported transformations. |
| `/api/model-metrics` | GET | None | `200 OK` | Returns true test split metrics (85.42% accuracy, 100% precision). |
| `/api/upload` | POST | 2 JPEG files | `200 OK` | Validates extensions, computes SHA-256/MD5 hashes, creates Analysis DB record. |
| `/api/upload` (Disallowed) | POST | `.txt` file | `400 Bad Request` | Correctly rejects non-image uploads. |
| `/api/analyze` | POST | `analysis_id` | `200 OK` | Executes ViT feature extraction, pairwise inference, and Spanning Forest. |
| `/api/analyses` | GET | None | `200 OK` | Lists historical analyses ordered by creation timestamp. |
| `/api/analyses/{id}` | GET | Valid ID | `200 OK` | Retrieves complete analysis object with nodes, edges, and summary. |
| `/api/analyses/{id}` | GET | Invalid UUID | `404 Not Found` | Gracefully reports missing analysis. |
| `/api/analyses/{id}/graph` | GET | Valid ID | `200 OK` | Returns Cytoscape JSON element format (`nodes`, `edges`, `roots`). |
| `/api/analyses/{id}/evidence` | GET | Valid ID | `200 OK` | Returns XAI heatmaps, MSE/PSNR/SSIM, and empirical rationale. |
| `/api/analyses/{id}/sensitivity` | GET | Valid ID | `200 OK` | Returns 5-group feature attribution delta percentages. |
| `/api/analyses/{id}/export/json` | GET | Valid ID | `200 OK` | Downloads complete machine-readable provenance JSON. |
| `/api/analyses/{id}/export/html` | GET | Valid ID | `200 OK` | Downloads standalone HTML forensic report. |
| `/api/predict-pair` | POST | 2 image paths | `200 OK` | Executes synchronous pairwise prediction with feature sensitivity. |

---

## 3. Database Persistence & Migration Audit

### SQLite Database Architecture:
- File: `backend/database/provenance.db`
- Tables:
  1. `analyses`: Primary analysis run metadata (`id`, `status`, `summary`, `created_at`, `similarity_threshold`).
  2. `images`: Stored image files (`id`, `analysis_id`, `filename`, `file_path`, `sha256`, `md5`, `width`, `height`, `file_size`).
  3. `edges`: Reconstructed tree edges (`id`, `analysis_id`, `source_id`, `target_id`, `transformation`, `confidence`, `weight`).
  4. `xai_reports`: Stored forensic evidence (`id`, `analysis_id`, `source_id`, `target_id`, `is_independent`, `relationship`, `heatmap_path`, `spatial_metrics`, `evidence`).

### Bug Fix: Independent Pair Persistence
- In prior builds, `edges` only stored connected tree edges. When an analysis produced 0 tree edges, the `xai_reports` table remained empty.
- **Fix Applied**:
  In [`backend/app/api/endpoints.py:270-320`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/backend/app/api/endpoints.py#L270-L320), the backend now stores XAI reports for disconnected / independent pairs with `is_independent=True` and `relationship="UNRELATED"`. This ensures evidence is permanently persisted in SQLite and viewable across browser sessions.

---

## 4. Frontend State & Navigation Reliability

1. **State Persistence on Browser Refresh**:
   In [`frontend/src/App.jsx:35-50`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/frontend/src/App.jsx#L35-L50), `currentAnalysisId` is mirrored to `window.localStorage`:
   - When the user refreshes the browser, navigates to the Provenance Graph, or clicks Explainable AI, the selected analysis is automatically reloaded.
   - Prevents empty screen states and silent session loss.

2. **Cytoscape Graph Layout Fallback**:
   In [`frontend/src/pages/ProvenanceGraph.jsx:110-145`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/frontend/src/pages/ProvenanceGraph.jsx#L110-L145):
   - When edges exist: Renders `dagre` hierarchical top-to-bottom DAG layout.
   - When 0 edges exist (disconnected components): Automatically falls back to a clean `grid` or `concentric` layout, preventing node overlaps or blank canvasses.

3. **Decoupled XAI Page Guards**:
   In [`frontend/src/pages/ExplainableAI.jsx:75-115`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/frontend/src/pages/ExplainableAI.jsx#L75-L115):
   - The UI no longer gates XAI evidence behind `edges.length > 0`.
   - Independent pairs render comparative divergence heatmaps with an explicit **"Independent Images — No Derivation Link"** badge.

---

## 5. Frontend Production Build Verification

```powershell
cmd.exe /c "npm run build" (in frontend/)
```
- **Exit Code**: 0
- **Duration**: 21.25s
- **Modules Transformed**: 1941
- **Assets Generated**:
  - `dist/index.html`: 0.91 kB
  - `dist/assets/index-CH_u2v48.css`: 39.63 kB (gzip: 6.81 kB)
  - `dist/assets/index-CpBvFrhR.js`: 882.64 kB (gzip: 272.10 kB)
- Zero syntax errors, zero missing imports.
