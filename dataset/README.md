# Synthetic Image Provenance Graph Dataset (SIPGD)

## Overview
SIPGD is a research-grade Image Provenance Analysis benchmark dataset containing multi-generational edit trees, branching manipulation histories, explicit root ground truth, distractor images, and forensic features.

## Statistics
- **Original Root Source Images**: 21
- **Total Generated Images**: 152
- **Provenance Graphs**: 21
- **Directed Edges**: 131
- **Total Pairwise Samples**: 524
- **Positive Pairs**: 262
- **Negative Distractor Pairs**: 262
- **Transformations**: 12 (crop, resize, jpeg, blur, brightness, contrast, sharpen, rotation, color, noise, perspective, quality_degradation)
- **Splits**: Train (70%), Val (15%), Test (15%) split strictly by original source ID.

## Directory Structure
```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── metadata/
│   ├── images.csv
│   ├── pairs_train.csv
│   ├── pairs_val.csv
│   ├── pairs_test.csv
│   ├── edges_train.csv
│   ├── edges_val.csv
│   ├── edges_test.csv
│   ├── graphs_train.json
│   ├── graphs_val.json
│   └── graphs_test.json
└── features/
    ├── forensic_features.csv
    └── *.npy
```
