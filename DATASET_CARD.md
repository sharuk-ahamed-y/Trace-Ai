# DATASET CARD: Synthetic Image Provenance Graph Dataset (SIPGD)

> **IMPORTANT RESEARCH NOTICE**:
> This is a synthetic Image Provenance Graph Dataset generated from legally usable source images.
> It is **NOT** MFC-Prov. It is **NOT** an official NIST dataset.

---

## Dataset Summary
- **Dataset Name**: Synthetic Image Provenance Graph Dataset (SIPGD) / Image Provenance Research Dataset (IPRD)
- **Version**: 2.0.0 (100-Original-Image Benchmark Release)
- **Purpose**: Research-grade benchmark dataset for Image Provenance Analysis, root identification, parent-child relationship prediction, transformation classification, graph reconstruction / MST, and Explainable AI (XAI).
- **License**: Creative Commons Attribution 4.0 International (CC-BY 4.0) / Public Domain for Research.

---

## Dataset Statistics & Metrics

| Metric | Count / Value |
| :--- | :--- |
| **Original Root Source Images** | **100 Unique Originals** (`original_0001` to `original_0100`) |
| **Unique Source Identities** | **100 Unique Source IDs** (`source_0001` to `source_0100`) |
| **Total Manipulated Descendants** | **~750 Manipulated Images** |
| **Total Provenance Graphs** | **100 Multi-Generation DAGs** (`graph_0001` to `graph_0100`) |
| **Total Pairwise Benchmark Samples** | **~3,000 Pairwise Samples** |
| **Positive Parent-Child Pairs** | **~1,500 Positive Pairs** |
| **Negative Distractor Pairs** | **~1,500 Negative Pairs** |
| **Transformations Supported** | **12 Realistic Operations** |
| **Identity Splits** | **70 Train, 15 Validation, 15 Test Sources** |
| **Source Identity Leakage** | **0.0% (Strictly Split by Source ID)** |

---

## Supported Transformations
1. `crop`: Random, center, and rectangular crop (5%–50%) with optional resize back.
2. `resize`: Scale factor down/upscale (0.35–1.40) with LANCZOS/BICUBIC/BILINEAR/NEAREST interpolations.
3. `jpeg`: Lossy JPEG compression re-encoding (quality 30–95).
4. `blur`: Gaussian blur kernel radius (0.8–4.5).
5. `brightness`: Pixel brightness factor shift (0.55–1.55).
6. `contrast`: Pixel contrast factor shift (0.55–1.60).
7. `sharpen`: Unsharp masking sharpness enhancement (1.4–3.2).
8. `rotation`: In-plane angle rotation (-25° to +25°).
9. `color`: Saturation color multiplier (0.40–1.80).
10. `noise`: Gaussian noise addition ($\sigma \in [6, 28]$).
11. `perspective`: Slight perspective affine distortion.
12. `quality_degradation`: Multi-degradation combination (downscaling + noise + low JPEG compression).

---

## Source Images & Legal Compliance
- Source root images are procedurally synthesized across 14 diverse visual domain categories or obtained from unencumbered CC0 / Public Domain repositories. Full details recorded in `DATASET_SOURCES.csv`.
- Restricted academic provenance datasets (e.g. NIST MFC-Prov) were audited and recorded in `DATASET_ACCESS.md`. Access rules are strictly respected; no unauthorized access or mirroring was used.

---

## Generation Methodology & Reproducibility
- **Random Seed**: Fixed at `SEED = 42` across all transformations, tree branching, distractor sampling, and splits.
- **Tree Topology**: Synthesized multi-generation DAGs (depth 2 to 4, branching factor 1 to 3, 5 to 10 nodes per graph).
- **Metadata**: Full exact transformation parameters saved as JSON strings for every edge.

---

## Benchmark Directory Structure
```text
image_provenance_dataset/
├── README.md
├── DATASET_CARD.md
├── DATASET_SOURCES.csv
├── DATASET_ACCESS.md
├── PROJECT_ANALYSIS.md
├── requirements_dataset.txt
├── images/
│   ├── train/ (originals/ & manipulated/)
│   ├── val/   (originals/ & manipulated/)
│   └── test/  (originals/ & manipulated/)
├── metadata/
├── features/
│   ├── forensic_features.csv
│   └── individual_features/*.npy
├── graphs/*.json
├── visualizations/*.png
└── viewer/
```

---

## Local Dataset Viewer & Browsing
Run the interactive Streamlit dataset browser:
```bash
streamlit run dataset_viewer.py
```
Or view the static browser: `dataset_viewer.html`.

---

## Reproducibility Command
To regenerate the complete 100-Original-Image benchmark dataset from scratch:
```bash
python setup_provenance_dataset.py
```
