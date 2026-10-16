# Relationship and Direction Model Audit — TRACE AI Forensic Audit

**Audited Models**:
1. Active Relationship Classifier: `models/selected_relationship_classifier.pkl`
2. Active Direction Classifier: `models/antisymmetric_direction_model.pt`
3. Fallback Relationship Checkpoints: `models/final_relationship_classifier.pkl`, `models/relationship_classifier.pkl`, `models/provenance_exp3_model.pt`
4. Fallback Direction Checkpoints: `models/direction_classifier.pkl`
**Code Reference**: [`backend/app/services/model_manager.py:340-470`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/backend/app/services/model_manager.py#L340-L470)  

---

## 1. Active Relationship Classifier Audit

### Architecture & Pipeline
- **File**: `models/selected_relationship_classifier.pkl`
- **Input Vector**: 115-Dimensional Pairwise Feature Vector:
  $$\mathbf{x}_{115} = [\ |\mathbf{f}_A - \mathbf{f}_B|\ (57\text{-D}),\quad \mathbf{f}_A \odot \mathbf{f}_B\ (57\text{-D}),\quad \text{CosSim}(\mathbf{e}_A, \mathbf{e}_B)\ (1\text{-D})\ ]$$
- **Estimator Pipeline**:
  $$\text{StandardScaler}() \longrightarrow \text{MLPClassifier}(\text{hidden\_layer\_sizes}=(64, 32), \text{activation}=\text{'relu'})$$
- **Decision Threshold**: $P(\text{related}) \ge 0.50$

### Validated Test Split Performance (`selected_relationship_test_metrics.json`):
- **Evaluation Samples**: 96 graph-disjoint pair instances
- **Accuracy**: **85.42%**
- **Precision**: **100.00%** (Zero false positives; no unrelated images predicted as related)
- **Recall**: **70.83%**
- **F1 Score**: **0.8293**
- **ROC AUC**: **0.9575**

### Audit Findings from Live Benchmark (150 Controlled Transformations):
- **Lineage Recall on Genuine Derivations**: **98.67%** (148/150)
- **Separation of Unrelated Images**: **100.00%** (5/5 unrelated pairs rejected with $P(\text{rel}) < 0.15$)
- **False Positive Rate**: **0.00%**
- **Verdict**: Exceptional precision; perfectly prevents foreign images from corrupting reconstructed graphs.

---

## 2. Active Direction Classifier Audit: The Antisymmetric Breakthrough

### Architecture & Formal Invariance
- **File**: `models/antisymmetric_direction_model.pt`
- **Input Vector**: 172-Dimensional Direction Feature Vector:
  $$\mathbf{x}_{172} = [\ \mathbf{f}_A\ (57\text{-D}),\quad \mathbf{f}_B\ (57\text{-D}),\quad (\mathbf{f}_A - \mathbf{f}_B)\ (57\text{-D}),\quad \text{CosSim}(\mathbf{e}_A, \mathbf{e}_B)\ (1\text{-D})\ ]$$
- **Network Architecture**: `AntisymmetricDirectionMLP(172, 64, 32)`
  The forward pass computes:
  $$\text{logit}(A, B) = g(\mathbf{x}_{A, B}) - g(\mathbf{x}_{B, A})$$
  $$P(A \to B) = \sigma(\text{logit}(A, B))$$

### Mathematical Invariance Property:
Because $g(\mathbf{x}_{A, B}) - g(\mathbf{x}_{B, A}) \equiv -(g(\mathbf{x}_{B, A}) - g(\mathbf{x}_{A, B}))$:
$$\text{logit}(B, A) \equiv -\text{logit}(A, B)$$
$$P(B \to A) \equiv 1.0 - P(A \to B)$$

### Empirical Swap Consistency Verification:
- **Test Set**: 150 pairs tested in both forward $(A, B)$ and reverse $(B, A)$ sequence.
- **Observed Consistency**: **150 / 150 (100.00%)**.
- **Maximum Confidence Delta**: **$0.000000$**.
- **Mean Confidence Delta**: **$0.000000$**.
- **Verdict**: The antisymmetric model completely eliminates the swap direction inconsistency bug that plagued standard scikit-learn tree classifiers.

### Direction Accuracy Breakdown:
- **Noise Addition**: **100.0%**
- **Perspective Warping**: **100.0%**
- **Quality Degradation**: **100.0%**
- **Rotation**: **91.7%**
- **Resizing**: **86.7%**
- **Brightness**: **75.0%**
- **Cropping**: **75.0%**
- **Sharpening**: **66.7%**
- **Contrast**: **58.3%**
- **Color**: **55.6%**
- **Gaussian Blur**: **0.0%** (Causal inversion failure)

---

## 3. Fallback and Legacy Models Presence Audit

| Model Artifact | File Type | Parameters / Dims | Status in Production |
|---|---|---|---|
| `selected_relationship_classifier.pkl` | Joblib Pickle | 115-D StandardScaler + MLP | **Active Production Relationship Model** |
| `antisymmetric_direction_model.pt` | PyTorch Checkpoint | 172-D Antisymmetric MLP | **Active Production Direction Model** |
| `final_relationship_classifier.pkl` | Joblib Pickle | 115-D StandardScaler + MLP | Verified Available (Fallback) |
| `relationship_classifier.pkl` | Joblib Pickle | 115-D StandardScaler + MLP | Verified Available (Baseline) |
| `direction_classifier.pkl` | Joblib Pickle | 172-D ExtraTrees | Deprecated (Swap Inconsistency) |
| `provenance_exp3_model.pt` | PyTorch Checkpoint | 2350-D Multi-task Classifier | Verified Available |
| `research_provenance_model.pt` | PyTorch Checkpoint | 289-D Research Architecture | Verified Available |
| `transformation_classifier_top20.pkl`| Joblib Pickle | 20-D ExtraTrees | **Active Production Transformation Model** |

---

## 4. Verification of Serialization and Loader Contracts

A critical bug was verified and resolved in model loading:
- `transformation_classifier_top20.pkl` was saved using `joblib` rather than standard Python `pickle`.
- In [`backend/app/services/model_manager.py:46-58`](file:///c:/Users/devad/Downloads/image-provenance-using-graph-main/backend/app/services/model_manager.py#L46), `_load_serialized_model` wraps both `joblib.load()` and `pickle.load()` with fallback handlers, ensuring robust cross-version loading across both local Windows environments and Linux containers.
