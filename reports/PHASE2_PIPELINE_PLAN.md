# TRACE AI — Phase 2: Safe Pipeline & Training Preparation Plan

**Document ID:** `reports/PHASE2_PIPELINE_PLAN.md`  
**Phase:** Phase 2 — Architecture, Supervision & Safe Pipeline Preparation  
**Timestamp:** October 18, 2026  
**Status:** DRAFT / PENDING USER APPROVAL  
**Workspace:** `C:\Users\dell\Downloads\Trace-Ai-main\Trace-Ai-main`  

---

## 1. Verified Environment & Hardware Status

The execution environment was audited directly through PowerShell and Python runtime queries:

| Component | Verified Specification | Audit Observation |
| :--- | :--- | :--- |
| **Python Executable** | `C:\Python312\python.exe` | Python 3.12.x 64-bit |
| **Operating System** | Windows 10/11 x64 | PowerShell execution environment |
| **System Memory (RAM)** | **15.67 GB Total** (~16,433,160 KB) | ~6.08 GB Free physical memory |
| **Processor (CPU)** | Intel(R) Core(TM) Architecture | Multi-core x64 CPU |
| **Graphics Controller** | `Intel(R) HD Graphics 520` | Integrated GPU (Adapter RAM: 1.0 GB shared) |
| **Dedicated NVIDIA GPU** | **None** | `nvidia-smi` not found; no CUDA support |
| **PyTorch / CUDA Status** | **Not Installed / False** | `torch` module missing in active Python |
| **Local Pretrained ViT Weights** | **Not Found on Disk** | `C:\Users\dell\.cache\torch` does not exist; no `.pth` weights |

### Missing Python Dependencies in `C:\Python312`:
* `torch` (Required for neural models & tensor math)
* `torchvision` (Required for Vision Transformer architecture)
* `opencv-python` (`cv2`) (Required for template matching & image resizing)
* `networkx` (Required for graph reconstruction & MST)
* `matplotlib` (Required for plotting & visualization)

---

## 2. Inventory of Existing Files Reusable for Training

No existing production files will be modified or deleted. The following modules and assets are verified and available for reuse:

1. **Source Generation Logic ([`create_sample_images.py`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/create_sample_images.py)):**
   * Standalone procedural synthesizer with 14 visual domain categories (landscapes, urban, tech, grids, gradients, textures, fractals, etc.).
   * Fully deterministic (`SEED=42`), uses only standard library, NumPy, and PIL.
   * Matches [`DATASET_SOURCES.csv`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/DATASET_SOURCES.csv) SHA-256 and MD5 hashes.
2. **Transformation Operators ([`src/transforms.py`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/src/transforms.py)):**
   * Implements 12 distinct transformation functions: `crop`, `resize`, `jpeg`, `blur`, `brightness`, `contrast`, `sharpen`, `rotation`, `color`, `noise`, `perspective`, and `quality_degradation`.
   * Parameter samplers for realistic degradation intervals.
3. **Forensic Feature Extraction ([`src/forensic_features.py`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/src/forensic_features.py)):**
   * Computes 17 scalar physical/spectral features (DCT blockiness quality, Laplacian edge moments, high-pass noise residuals, aspect ratio, color moments) + 40 histogram bins (16-bin gray + 24-bin RGB) = **57-D feature vector**.
   * Completely independent of PyTorch and operates purely on NumPy / PIL.
4. **Feature Fusion Logic ([`src/pair_features.py`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/src/pair_features.py)):**
   * `build_115d_pair_feature`: Concatenates $|f_A - f_B|$ (57), $f_A \odot f_B$ (57), and $\text{CosSim}$ (1) for symmetric relationship and transformation tasks.
   * `build_172d_direction_feature`: Concatenates $f_A$ (57), $f_B$ (57), $(f_A - f_B)$ (57), and $\text{CosSim}$ (1) for directional learning.
5. **Existing Feature Arrays:**
   * `image_provenance_dataset/features/individual_features/*.npy`: **795 files** (55-D float32 arrays).
   * `dataset/features/*.npy`: **152 files** (55-D float32 arrays).
6. **Deterministic Forensic Rules ([`src/forensic_arbitrator.py`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/src/forensic_arbitrator.py)):**
   * Anti-symmetric sub-region template matching, low-pass spectral blur attenuation, and 8×8 DCT grid discontinuity analysis.

---

## 3. Dataset Generation Analysis & Overwrite Risk Assessment

### 3.1 Expected Generation Outputs
Running standard dataset generation (`setup_provenance_dataset.py` / `src/generate_provenance_dataset.py`) produces:
* **Images:** ~850 to 1,000 JPEG images (100 root originals + 750–900 manipulated descendants, partitioned into `train/`, `val/`, `test/`).
* **Metadata Manifests:**
  * `images.csv` (complete node inventory with file paths, sizes, dimensions, roles).
  * `edges_train.csv`, `edges_val.csv`, `edges_test.csv` (directed ground-truth edges and transformation parameters).
  * `pairs_train.csv`, `pairs_val.csv`, `pairs_test.csv` (positive pairs + negative distractor pairs with direction labels).
  * `graphs_train.json`, `graphs_val.json`, `graphs_test.json` (DAG topologies).
* **Features:**
  * `forensic_features.csv` (57-D features per image).
  * Individual `.npy` feature files.

### 3.2 Overwrite Risk Analysis
* **Risk in `image_provenance_dataset/`:**
  * Direct generation targeting `image_provenance_dataset/` would overwrite `image_provenance_dataset/README.md` and write into `image_provenance_dataset/features/individual_features/`, potentially mixing newly generated arrays with the 795 legacy `.npy` files.
* **Risk in `dataset/`:**
  * Direct generation targeting `dataset/` would overwrite `dataset/README.md` and collide with the 152 legacy `.npy` files in `dataset/features/`.
* **Protective Solution (Proposed Isolated Directory):**
  * All new dataset generation will target a strictly separated directory:  
    `data/provenance_dataset_v2/` (or `dataset_v2/`).
  * This guarantees that **100% of existing repository files, documentation, and legacy `.npy` files remain untouched**.

---

## 4. Pretrained Vision Transformer (ViT) Assessment

### 4.1 Dependency & Weight Verification
* The existing code in [`src/feature_extractor.py`](file:///c:/Users/dell/Downloads/Trace-Ai-main/Trace-Ai-main/src/feature_extractor.py) wraps `torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)`.
* **Local Availability:** No pretrained ViT checkpoint exists in `C:\Users\dell\.cache\torch` or anywhere in the workspace.
* **Network & Weight Policy:** In accordance with strict instructions ("*Do not download pretrained weights automatically*"), **no automatic downloading of the ~330 MB ImageNet ViT weights will be triggered**.

### 4.2 CPU Feasibility & Strategy
* **Inference on CPU:** Once PyTorch and torchvision are installed, forward inference through frozen ViT-B/16 on CPU requires ~300 MB RAM and ~60–120ms per 224×224 image. This is completely feasible on this 16 GB machine if weights are later approved for download.
* **Zero-Download Alternative for Phase 2-4:**
  * Feature extraction can be completely performed using the **57-D physical and spectral forensic descriptors** (which require zero network downloads and run instantly with NumPy and PIL).
  * The existing 795 `.npy` (55-D) pre-computed visual features can also be directly utilized without downloading anything.
  * This ensures that progress is never blocked by external weight availability.

---

## 5. Design of the Three Separate Training Targets

To support accurate, explainable provenance and prevent forced incorrect predictions, three specialized heads are defined:

```
                          [Image Pair (A, B)]
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
     [Symmetric Features]                     [Antisymmetric Features]
    |f_A - f_B|, f_A * f_B, CosSim                 f_A - f_B, Area Ratio,
              │                                  Spectral Attenuation
              ├────────────────────┐                    │
              ▼                    ▼                    ▼
     Target 1: Relation     Target 2: Transform    Target 3: Direction
     [Related vs Unrelated] [Multi-Label BCE]      [A->B, B->A, UNCERTAIN]
```

### Target 1: Relationship Classification (Related vs. Unrelated)
* **Objective:** Determine whether two suspect images share a common ancestor or whether one is derived from the other.
* **Input Representation:** Symmetric feature fusion:
  $$\mathbf{f}_{\text{rel}}(A, B) = \big[ |\mathbf{f}_A - \mathbf{f}_B|, \; \mathbf{f}_A \odot \mathbf{f}_B, \; \text{CosineSim}(\mathbf{f}_A, \mathbf{f}_B) \big]$$
  Symmetry guarantee: $\mathbf{f}_{\text{rel}}(A, B) \equiv \mathbf{f}_{\text{rel}}(B, A)$.
* **Supervision:** Binary Cross-Entropy Loss with Logits (`BCEWithLogitsLoss`).
  * $y = 1$: Verified parent-child or descendant pairs.
  * $y = 0$: Distractor pairs sampled across disparate root DAGs.
* **Output:** Calibrated probability $P(\text{related}) \in [0.0, 1.0]$. Gated at $0.50$ (or threshold $\tau = 0.55$).

### Target 2: Multi-Label Transformation Recognition
* **Objective:** Identify all transformations applied between parent and child rather than assuming only a single transformation occurred.
* **Input Representation:** Symmetric pair features $\mathbf{f}_{\text{tfm}}(A, B)$.
* **Classes (12 Target Dimensions):**
  1. `crop`
  2. `resize`
  3. `jpeg`
  4. `blur`
  5. `brightness`
  6. `contrast`
  7. `sharpen`
  8. `rotation`
  9. `color`
  10. `noise`
  11. `perspective`
  12. `quality_degradation` (composite multi-bit activation)
* **Supervision:** Independent Multi-Label Binary Cross-Entropy:
  $$\mathcal{L}_{\text{tfm}} = \sum_{k=1}^{12} \text{BCEWithLogits}(\hat{z}_k, y_k)$$
  Where $\mathbf{y} \in \{0, 1\}^{12}$. Multiple bits can be active simultaneously (e.g. crop + resize + jpeg).
* **Negative Handling:** For unrelated pairs ($y_{\text{rel}} = 0$), transformation loss is masked out or set to all-zeros.

### Target 3: Direction Classification (A $\to$ B, B $\to$ A, or UNCERTAIN)
* **Objective:** Predict chronological manipulation direction without forcing a guess when evidence is physically ambiguous.
* **Input Representation:** Antisymmetric signed feature fusion:
  $$\mathbf{f}_{\text{dir}}(A, B) = \big[ \mathbf{f}_A, \; \mathbf{f}_B, \; (\mathbf{f}_A - \mathbf{f}_B), \; \Delta_{\text{area}}, \; \Delta_{\text{spectral}} \big]$$
* **Antisymmetric Formulation:**
  Using an antisymmetric MLP head:
  $$\text{logit}(A, B) = \frac{1}{2}\big( g(\mathbf{f}_{A, B}) - g(\mathbf{f}_{B, A}) \big)$$
  * Guarantees $\text{logit}(B, A) = -\text{logit}(A, B)$ and $P(A \to B) = 1.0 - P(B \to A)$ (100% swap consistency).
* **Supervision & Decision Rule:**
  * If $P(A \to B) \ge 0.60 \implies \mathbf{A \to B}$ (confidence = $P(A \to B)$)
  * If $P(A \to B) \le 0.40 \implies \mathbf{B \to A}$ (confidence = $1.0 - P(A \to B)$)
  * If $0.40 < P(A \to B) < 0.60 \implies \mathbf{UNCERTAIN}$ (flagged for forensic arbitration)

---

## 6. Dataset Supervision Gap & Quality Review

| Target | Synthetic Dataset Capability | Real-World Gap / Limitation | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Relationship** | **Fully Supervised**: 1:1 ratio of positive DAG pairs and negative distractor pairs. | Negatives in synthetic sets are visually disparate; lacks "hard negatives" (e.g., two distinct photos of Eiffel Tower). | Incorporate color/histogram variance in distractor sampling. |
| **Multi-Label Transformation** | **Partially Supervised**: Current generator writes single string per edge. `quality_degradation` contains 3 operations but has 1 label. | Multi-generational ancestors accumulate edits, but pairs are currently generated only for 1-hop edges. | Upgrade manifest builder to output explicit 12-dimensional binary vector $\mathbf{y} \in \{0, 1\}^{12}$ per pair. |
| **Direction** | **Fully Supervised on Degradations**: Crop, resize down, blur, noise, and JPEG have clear ground truth. | **Symmetric Edits**: Pure brightness or contrast shifts have identical entropy/derivatives; direction is physically underdetermined. | Emit `UNCERTAIN` for pairs where $|\Delta_{\text{entropy}}| < \epsilon$ and $|\Delta_{\text{area}}| == 0$. |

---

## 7. CPU-Only Smoke Test Execution & Verification

A self-contained smoke test (`scratch/test_cpu_smoke.py`) was implemented and executed on CPU:

### Smoke Test Results (Verified Fact):
* **Procedural Synthesis:** 2 roots generated, 4 descendants produced (crop, blur, jpeg, resize). Total = 6 images.
* **Forensic Feature Extraction:** 57-D descriptors extracted without external downloads.
* **Pair Construction:** 8 pairs (4 positive, 4 negative distractor) constructed with 115-D symmetric and 172-D antisymmetric features.
* **Feature Normalization Insight:** Raw feature products ($area^2 \approx 6.8 \times 10^{10}$) require `StandardScaler` feature normalization before feeding into neural heads to prevent exploding gradients.
* **Empirical Execution Telemetry:**
  * Execution Time: **6.76 seconds**
  * Peak Memory: **< 120 MB RAM**
  * Relationship Loss: **0.6480**
  * Multi-label Transformation Loss: **0.7007**
  * Direction Loss: **0.9240**
  * Direction Swap Antisymmetry: **$|f(A, B) + f(B, A)| < 10^{-6}$ (100% Consistent)**
  * Scratch Directory Cleanup: **Verified**


---

## 8. Rollback Precautions & Risk Controls

1. **No Production Overwrites:** Production backend code in `backend/` and frontend React code in `frontend/` remain completely untouched during preparation.
2. **Safe Workspace Location:** All experimental scripts and test outputs reside exclusively in `scratch/` or `data/provenance_dataset_v2/`.
3. **No Unapproved Downloads:** No calls to `torchvision.models.vit_b_16(weights=...)` will be made without explicit user consent.
4. **Preserved Checkpoints:** The `models/` directory will only receive new checkpoints once training is validated on a held-out test split.

---

## 9. Exact Next Command to Run

To install the missing environment dependencies in CPU-only mode into user site-packages (avoiding Windows Program Files / system permissions):

```powershell
pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cpu; pip install --user opencv-python networkx matplotlib
```

And to verify the installation via the CPU smoke test:

```powershell
python scratch/test_cpu_smoke.py
```

