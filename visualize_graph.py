"""
visualize_graph.py — Standalone tool to visualize image provenance graphs.

Usage:
  python visualize_graph.py --graph_id graph_0001
  python visualize_graph.py --all
"""

import argparse
import io
import json
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from src.config import ROOT

DATASET_DIR     = ROOT / "dataset"
META_DIR        = DATASET_DIR / "metadata"
VISUALIZATIONS  = ROOT / "visualizations"

_COLOR_MAP = {
    "crop": "#E74C3C",
    "resize": "#3498DB",
    "jpeg": "#2ECC71",
    "blur": "#9B59B6",
    "brightness": "#F39C12",
    "contrast": "#E67E22",
    "sharpen": "#1ABC9C",
    "rotation": "#D35400",
    "color": "#8E44AD",
    "noise": "#C0392B",
    "perspective": "#16A085",
    "quality_degradation": "#7F8C8D",
}
_DEFAULT_COLOR = "#95A5A6"


def render_graph_from_json(graph_data: dict, output_path: Path) -> Path:
    """
    Render a single provenance graph structure to PNG.
    """
    DG = nx.DiGraph()
    graph_id = graph_data.get("graph_id", "Unknown Graph")
    root_node = graph_data.get("root_image", "")

    for node in graph_data.get("nodes", []):
        node_id = node["id"]
        role = node.get("role", "manipulated")
        DG.add_node(node_id, role=role, is_root=(node_id == root_node))

    for edge in graph_data.get("edges", []):
        s = edge["source"]
        t = edge["target"]
        op = edge.get("operation", "?")
        conf = edge.get("confidence", 1.0)
        DG.add_edge(s, t, operation=op, confidence=conf)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("#121212")
    fig.patch.set_facecolor("#121212")

    pos = nx.spring_layout(DG, seed=42, k=2.0 / max(1, math.sqrt(len(DG.nodes()))))

    # Node color by role
    node_colors = []
    for n in DG.nodes():
        if n == root_node or DG.nodes[n].get("is_root"):
            node_colors.append("#FFD700")  # Gold for Root
        elif DG.nodes[n].get("role") == "distractor":
            node_colors.append("#E74C3C")  # Red for distractor
        else:
            node_colors.append("#4A90E2")  # Blue for manipulated

    labels = {n: n.split("_")[-1] if "_" in n else n for n in DG.nodes()}

    nx.draw_networkx_nodes(
        DG, pos, ax=ax, node_color=node_colors, node_size=1600, edgecolors="#FFFFFF", linewidths=2
    )
    nx.draw_networkx_labels(
        DG, pos, labels=labels, ax=ax, font_color="#FFFFFF", font_size=8, font_weight="bold"
    )

    edge_colors = []
    edge_labels = {}
    for u, v, data in DG.edges(data=True):
        op = data.get("operation", "?")
        conf = data.get("confidence", 1.0)
        edge_colors.append(_COLOR_MAP.get(op, _DEFAULT_COLOR))
        edge_labels[(u, v)] = f"{op}\n({conf:.2f})" if conf < 1.0 else op

    nx.draw_networkx_edges(
        DG, pos, ax=ax, edge_color=edge_colors, arrows=True, arrowstyle="-|>", arrowsize=20, width=2.5
    )
    nx.draw_networkx_edge_labels(
        DG, pos, edge_labels=edge_labels, ax=ax, font_color="#F0F0F0", font_size=7,
        bbox={"boxstyle": "round,pad=0.2", "fc": "#1E1E1E", "ec": "none", "alpha": 0.8}
    )

    # Legend
    legend_elements = [
        mpatches.Patch(color="#FFD700", label="Original Root"),
        mpatches.Patch(color="#4A90E2", label="Manipulated Node"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", facecolor="#1E1E1E", edgecolor="white", labelcolor="white")

    ax.set_title(f"Image Provenance Graph: {graph_id}", color="white", fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Visualization] Saved graph -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize Provenance Graph")
    parser.add_argument("--graph_id", type=str, default="graph_0001", help="Graph ID to render")
    parser.add_argument("--all", action="store_true", help="Render all graphs")
    args = parser.parse_args()

    VISUALIZATIONS.mkdir(parents=True, exist_ok=True)

    # Find graphs in graphs_train.json / val / test
    found_graphs = []
    for split in ["train", "val", "test"]:
        gjson = META_DIR / f"graphs_{split}.json"
        if gjson.exists():
            with open(gjson, "r", encoding="utf-8") as f:
                data = json.load(f)
                found_graphs.extend(data)

    if not found_graphs:
        print(f"[ERROR] No graph JSON files found in {META_DIR}. Run setup_provenance_dataset.py first.")
        sys.exit(1)

    if args.all:
        print(f"[Visualization] Rendering {len(found_graphs)} graph(s) ...")
        for g in found_graphs:
            out_path = VISUALIZATIONS / f"{g['graph_id']}.png"
            render_graph_from_json(g, out_path)
    else:
        target = [g for g in found_graphs if g.get("graph_id") == args.graph_id]
        if not target:
            target = [found_graphs[0]]
            print(f"[WARN] Graph ID {args.graph_id} not found. Rendering first graph {target[0]['graph_id']} ...")
        out_path = VISUALIZATIONS / f"{target[0]['graph_id']}.png"
        render_graph_from_json(target[0], out_path)


if __name__ == "__main__":
    import math
    main()
