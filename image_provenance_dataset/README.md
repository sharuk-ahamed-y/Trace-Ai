# Synthetic Image Provenance Graph Dataset (SIPGD)

## Dataset Overview
SIPGD is a research-grade Image Provenance Analysis benchmark dataset containing 100 multi-generational edit trees, branching manipulation histories, explicit root ground truth, distractor images, and forensic features.

## Statistics
- **Original Root Source Images**: 100 (source_0001 to source_0100)
- **Total Generated Images**: 795
- **Provenance Graphs**: 100 Multi-Level DAGs
- **Directed Edges**: 695
- **Total Pairwise Samples**: 2780
- **Positive Pairs**: 1390
- **Negative Distractor Pairs**: 1390
- **Transformations**: 12 (crop, resize, jpeg, blur, brightness, contrast, sharpen, rotation, color, noise, perspective, quality_degradation)
- **Identity Splits**: 70 Train, 15 Validation, 15 Test (0% data leakage)

## Folder Structure
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
└── visualizations/*.png
```
