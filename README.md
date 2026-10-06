# Transformation-Aware Image Provenance Graph Reconstruction

> A local, reproducible prototype for constructing labelled image provenance graphs with pretrained Vision Transformer features.

## Overview

This project reconstructs relationships among images derived through controlled edits. Given a set of related images, it builds a provenance graph, predicts the direction of each relationship, and identifies the transformation applied to each edge.

The supported transformations are crop, resize, JPEG compression, Gaussian blur, and brightness adjustment.

```text
Original images
      |
      v
Controlled transformation generation
      |
      v
Pretrained ViT-B/16 feature extraction
      |
      v
Pairwise cosine distances and MST construction
      |
      v
Edge direction and transformation prediction
      |
      v
Labelled directed provenance graph
```

Example:

```text
original.jpg -> crop.jpg        [crop, 0.94]
crop.jpg     -> compressed.jpg  [jpeg, 0.89]
```

## Features

- Generates controlled parent-child image pairs with recorded ground truth.
- Uses a pretrained ViT-B/16 model as a frozen visual feature extractor.
- Constructs an undirected relationship graph using a Minimum Spanning Tree (MST).
- Predicts edge direction with a lightweight MLP classifier.
- Predicts transformation type and confidence for each directed edge.
- Exports a graph visualization, CSV predictions, and JSON predictions.
- Runs on CPU by default and uses CUDA automatically when available.

## Repository Structure

```text
image-provenance-prototype/
|-- README.md
|-- requirements.txt
|-- train.py                  # Generate data, extract embeddings, and train classifiers
|-- demo.py                   # Run the end-to-end demonstration
|-- infer.py                  # Run inference on a supplied image set
|-- data/
|   |-- originals/            # Add source images here
|   |-- generated/            # Generated child images
|   `-- labels/edges.csv      # Ground-truth parent-child records
|-- models/                   # Saved classifiers and cached embeddings
|-- outputs/                  # Graph image, CSV, and JSON predictions
`-- src/
    |-- generate_dataset.py
    |-- feature_extractor.py
    |-- pair_features.py
    |-- train_classifiers.py
    |-- graph_builder.py
    |-- visualize_graph.py
    `-- metrics.py
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/image-provenance-prototype.git
cd image-provenance-prototype

python -m venv .venv
```

Activate the environment.

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

Install the project dependencies.

```bash
pip install -r requirements.txt
```
