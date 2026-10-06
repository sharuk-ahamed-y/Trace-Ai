"""
visualize_graph.py — Render the directed provenance graph as a PNG.

Nodes   : image basenames
Edges   : arrows with label "transformation (confidence)" e.g. "crop (0.94)"
Layout  : spring layout (via NetworkX + matplotlib)
"""

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")   # headless backend for CI / server environments
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from src.config import OUTPUTS_DIR, TRANSFORMATION_NAMES

# Colour map for transformation types
_TRANSFORM_COLORS = {
    "crop":       "#E74C3C",
    "resize":     "#3498DB",
    "jpeg":       "#2ECC71",
    "blur":       "#9B59B6",
    "brightness": "#F39C12",
}
_DEFAULT_COLOR = "#95A5A6"


def render_provenance_graph(
    DG: nx.DiGraph,
    output_path: Path = OUTPUTS_DIR / "provenance_graph.png",
    title: str = "Image Provenance Graph",
) -> Path:
    """
    Save *DG* as a labelled directed graph PNG.

    Edge labels are formatted as "transformation (confidence)".
    Node labels are image basenames.
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_facecolor("#1A1A2E")
    fig.patch.set_facecolor("#1A1A2E")

    # Node label mapping: full path → basename
    node_labels = {n: Path(n).name for n in DG.nodes()}

    # Layout
    if len(DG.nodes()) <= 2:
        pos = nx.shell_layout(DG)
    else:
        pos = nx.spring_layout(DG, seed=42, k=2.5 / max(1, len(DG.nodes()) ** 0.5))

    # Node drawing
    nx.draw_networkx_nodes(
        DG, pos, ax=ax,
        node_color="#16213E",
        node_size=1800,
        edgecolors="#E94560",
        linewidths=2,
    )
    nx.draw_networkx_labels(
        DG, pos, labels=node_labels, ax=ax,
        font_color="#EAEAEA", font_size=7, font_weight="bold",
    )

    # Edge colours by transformation
    edge_colors = []
    edge_labels = {}
    for u, v, data in DG.edges(data=True):
        tfm  = data.get("transformation", "?")
        conf = data.get("tfm_conf", 0.0)
        edge_colors.append(_TRANSFORM_COLORS.get(tfm, _DEFAULT_COLOR))
        edge_labels[(u, v)] = f"{tfm}\n({conf:.2f})"

    nx.draw_networkx_edges(
        DG, pos, ax=ax,
        edge_color=edge_colors,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        width=2.5,
        connectionstyle="arc3,rad=0.12",
        node_size=1800,
        min_source_margin=20,
        min_target_margin=20,
    )

    # Edge labels
    nx.draw_networkx_edge_labels(
        DG, pos, edge_labels=edge_labels, ax=ax,
        font_color="#F0F0F0", font_size=7,
        bbox={"boxstyle": "round,pad=0.3", "fc": "#0F3460", "alpha": 0.80,
              "ec": "none"},
        label_pos=0.40,
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color=_TRANSFORM_COLORS.get(t, _DEFAULT_COLOR), label=t)
        for t in TRANSFORMATION_NAMES
    ]
    ax.legend(
        handles=legend_handles, loc="upper left", fontsize=8,
        facecolor="#16213E", edgecolor="#E94560", labelcolor="white",
    )

    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=15)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Graph] Saved -> {output_path}")
    return output_path
