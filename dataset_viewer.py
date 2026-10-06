"""
dataset_viewer.py — Streamlit local dataset browser and visual inspector for SIPGD.

Features:
  - Dashboard: Live metrics, transformation distributions, identity splits
  - Original Image Gallery: Grid of 100 originals with search, filter, hashes, and detailed specs
  - Provenance Graph Viewer: Visual provenance trees, node details, and transformation parameters
  - Side-by-Side Image Comparison: Compare Image A vs Image B with ground-truth relationship metadata
  - Metadata Table View: Search, filter, sort, and inspect all CSVs and JSON graph ground truths
  - Download Center: Direct download buttons for complete ZIP archive and individual files

Usage:
  streamlit run dataset_viewer.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Config setup
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "image_provenance_dataset"
META_DIR    = DATASET_DIR / "metadata"
IMAGES_DIR  = DATASET_DIR / "images"
GRAPHS_DIR  = DATASET_DIR / "graphs"
VIS_DIR     = DATASET_DIR / "visualizations"
ZIP_FILE    = ROOT / "image_provenance_dataset.zip"

st.set_page_config(
    page_title="SIPGD — Image Provenance Dataset Viewer",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_metadata():
    images_csv = META_DIR / "images.csv"
    sources_csv = ROOT / "DATASET_SOURCES.csv"

    if not images_csv.exists():
        return None, None, None, None

    df_images  = pd.read_csv(images_csv)
    df_sources = pd.read_csv(sources_csv) if sources_csv.exists() else pd.DataFrame()

    pairs_train = pd.read_csv(META_DIR / "pairs_train.csv") if (META_DIR / "pairs_train.csv").exists() else pd.DataFrame()
    pairs_val   = pd.read_csv(META_DIR / "pairs_val.csv") if (META_DIR / "pairs_val.csv").exists() else pd.DataFrame()
    pairs_test  = pd.read_csv(META_DIR / "pairs_test.csv") if (META_DIR / "pairs_test.csv").exists() else pd.DataFrame()

    df_pairs = pd.concat([pairs_train, pairs_val, pairs_test], ignore_index=True)
    return df_images, df_sources, df_pairs, (pairs_train, pairs_val, pairs_test)


# Sidebar Navigation
st.sidebar.title("🌳 SIPGD Viewer")
st.sidebar.caption("Synthetic Image Provenance Graph Dataset")
menu = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🖼️ Original Image Gallery", "🌳 Provenance Graph Viewer", "🔍 Image Comparison View", "📑 Metadata Tables", "💾 Download Center"]
)

df_images, df_sources, df_pairs, (pairs_train, pairs_val, pairs_test) = load_metadata()

if df_images is None:
    st.error(f"Dataset not found at `{DATASET_DIR}`. Please run `python setup_provenance_dataset.py` first.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dashboard
# ─────────────────────────────────────────────────────────────────────────────
if menu == "📊 Dashboard":
    st.header("📊 Dataset Overview & Statistics")
    st.markdown("Live benchmark metrics computed directly from active dataset metadata.")

    n_originals = df_images[df_images["role"] == "original"]["image_id"].nunique()
    n_manipulated = df_images[df_images["role"] == "manipulated"]["image_id"].nunique()
    n_graphs = df_images["graph_id"].nunique()
    n_pairs = len(df_pairs)
    pos_pairs = len(df_pairs[df_pairs["related"] == 1])
    neg_pairs = len(df_pairs[df_pairs["related"] == 0])

    train_sources = df_images[df_images["split"] == "train"]["source_id"].nunique()
    val_sources   = df_images[df_images["split"] == "val"]["source_id"].nunique()
    test_sources  = df_images[df_images["split"] == "test"]["source_id"].nunique()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Original Images", f"{n_originals}")
    c2.metric("Manipulated Images", f"{n_manipulated}")
    c3.metric("Provenance Graphs", f"{n_graphs}")
    c4.metric("Positive Pairs", f"{pos_pairs}")
    c5.metric("Negative Pairs", f"{neg_pairs}")

    st.markdown("---")
    st.subheader("Identity Splits & Transformation Counts")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Source Identity Split (0% Data Leakage)")
        split_df = pd.DataFrame({
            "Split": ["Train", "Validation", "Test"],
            "Source Identities": [train_sources, val_sources, test_sources],
            "Fraction": [f"{train_sources/n_originals*100:.0f}%", f"{val_sources/n_originals*100:.0f}%", f"{test_sources/n_originals*100:.0f}%"]
        })
        st.dataframe(split_df, use_container_width=True)

    with col_b:
        st.markdown("#### Transformation Distribution")
        tfm_counts = df_images[df_images["transformation_from_parent"] != "NONE"]["transformation_from_parent"].value_counts()
        st.bar_chart(tfm_counts)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Original Image Gallery
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "🖼️ Original Image Gallery":
    st.header("🖼️ Original Source Image Gallery (100 Root Originals)")

    col1, col2 = st.columns([1, 2])
    with col1:
        split_filter = st.selectbox("Filter by Split", ["All", "train", "val", "test"])
    with col2:
        search_query = st.text_input("Search (by ID, source_id, graph_id)", "")

    orig_df = df_images[df_images["role"] == "original"]
    if split_filter != "All":
        orig_df = orig_df[orig_df["split"] == split_filter]

    if search_query:
        orig_df = orig_df[
            orig_df["image_id"].str.contains(search_query, case=False) |
            orig_df["source_id"].str.contains(search_query, case=False) |
            orig_df["graph_id"].str.contains(search_query, case=False)
        ]

    st.caption(f"Showing {len(orig_df)} original source images.")

    cols = st.columns(4)
    for idx, (_, row) in enumerate(orig_df.iterrows()):
        col = cols[idx % 4]
        img_path = ROOT / row["file_path"]

        with col:
            if img_path.exists():
                st.image(str(img_path), use_column_width=True)
            st.markdown(f"**{row['image_id']}**")
            st.caption(f"Graph: `{row['graph_id']}` | Split: `{row['split']}`")

            # Descendants count
            n_desc = len(df_images[df_images["graph_id"] == row["graph_id"]]) - 1
            st.caption(f"Descendants: **{n_desc}** | Size: {row['width']}x{row['height']}")

            with st.expander("Spec Details"):
                src_row = df_sources[df_sources["source_id"] == row["source_id"]]
                if not src_row.empty:
                    st.write(f"Category: `{src_row.iloc[0]['category']}`")
                    st.write(f"MD5: `{src_row.iloc[0]['md5_hash'][:12]}...`")
                    st.write(f"SHA256: `{src_row.iloc[0]['sha256_hash'][:12]}...`")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Provenance Graph Viewer
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "🌳 Provenance Graph Viewer":
    st.header("🌳 Provenance Graph Reconstruction Viewer")

    graphs = sorted(df_images["graph_id"].unique().tolist())
    selected_graph = st.selectbox("Select Provenance Graph", graphs)

    g_nodes = df_images[df_images["graph_id"] == selected_graph]
    root_node = g_nodes[g_nodes["role"] == "original"].iloc[0]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Provenance Tree Visualization ({selected_graph})")
        vis_path = VIS_DIR / f"{selected_graph}.png"
        if vis_path.exists():
            st.image(str(vis_path), use_column_width=True)
        else:
            st.info("Graph image rendering available via `python visualize_graph.py`.")

    with col2:
        st.subheader("Graph Ground Truth Specs")
        st.markdown(f"**Root Image ID**: `{root_node['image_id']}`")
        st.markdown(f"**Source ID**: `{root_node['source_id']}`")
        st.markdown(f"**Split**: `{root_node['split']}`")
        st.markdown(f"**Total Nodes**: {len(g_nodes)}")

        # JSON file preview
        gjson_path = GRAPHS_DIR / f"{selected_graph}.json"
        if gjson_path.exists():
            with open(gjson_path, "r", encoding="utf-8") as f:
                gdata = json.load(f)
            with st.expander("View Graph JSON Ground Truth"):
                st.json(gdata)

    st.markdown("---")
    st.subheader("Interactive Graph Nodes Inspection")

    node_ids = g_nodes["image_id"].tolist()
    selected_node_id = st.selectbox("Select Node to Inspect", node_ids)
    node_data = g_nodes[g_nodes["image_id"] == selected_node_id].iloc[0]

    col_n1, col_n2 = st.columns([1, 2])
    with col_n1:
        n_img_path = ROOT / node_data["file_path"]
        if n_img_path.exists():
            st.image(str(n_img_path), use_column_width=True)

    with col_n2:
        st.write(f"**Image ID**: `{node_data['image_id']}`")
        st.write(f"**Role**: `{node_data['role']}`")
        st.write(f"**Generation Level**: {node_data['generation']}")
        st.write(f"**Parent Node ID**: `{node_data['parent_id']}`")
        st.write(f"**Transformation from Parent**: `{node_data['transformation_from_parent']}`")
        st.write(f"**Dimensions**: {node_data['width']} x {node_data['height']} ({node_data['file_size'] / 1024:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Image Comparison View
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "🔍 Image Comparison View":
    st.header("🔍 Pairwise Image Comparison & Label Verification")

    all_image_ids = df_images["image_id"].tolist()

    c1, c2 = st.columns(2)
    with c1:
        img_a_id = st.selectbox("Select Image A", all_image_ids, index=0)
    with c2:
        img_b_id = st.selectbox("Select Image B", all_image_ids, index=min(1, len(all_image_ids)-1))

    row_a = df_images[df_images["image_id"] == img_a_id].iloc[0]
    row_b = df_images[df_images["image_id"] == img_b_id].iloc[0]

    col_img_a, col_img_b = st.columns(2)
    with col_img_a:
        path_a = ROOT / row_a["file_path"]
        if path_a.exists():
            st.image(str(path_a), caption=f"Image A: {img_a_id}", use_column_width=True)
        st.write(f"Graph: `{row_a['graph_id']}` | Gen: {row_a['generation']}")

    with col_img_b:
        path_b = ROOT / row_b["file_path"]
        if path_b.exists():
            st.image(str(path_b), caption=f"Image B: {img_b_id}", use_column_width=True)
        st.write(f"Graph: `{row_b['graph_id']}` | Gen: {row_b['generation']}")

    st.markdown("---")
    st.subheader("Ground Truth Pairwise Relationship")

    pair_match = df_pairs[
        ((df_pairs["image_a"] == img_a_id) & (df_pairs["image_b"] == img_b_id)) |
        ((df_pairs["image_a"] == img_b_id) & (df_pairs["image_b"] == img_a_id))
    ]

    if not pair_match.empty:
        p_info = pair_match.iloc[0]
        st.success(f"**Related**: {'YES (1)' if p_info['related'] == 1 else 'NO (0)'}")
        st.info(f"**Direction**: `{p_info['direction']}` | **Transformation**: `{p_info['transformation']}`")
        st.write(f"Graph ID: `{p_info['graph_id']}` | Confidence GT: {p_info.get('confidence_ground_truth', 1.0)}")
    else:
        st.warning("No explicit pairwise record found for this combination (Unrelated / Distractor Pair).")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Metadata Tables
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "📑 Metadata Tables":
    st.header("📑 Metadata CSV & JSON Browser")

    table_choice = st.selectbox(
        "Select Metadata Table",
        ["images.csv", "pairs_train.csv", "pairs_val.csv", "pairs_test.csv", "edges_train.csv", "edges_val.csv", "edges_test.csv", "DATASET_SOURCES.csv"]
    )

    t_path = META_DIR / table_choice if table_choice != "DATASET_SOURCES.csv" else ROOT / "DATASET_SOURCES.csv"

    if t_path.exists():
        df_table = pd.read_csv(t_path)
        st.dataframe(df_table, use_container_width=True)
        st.caption(f"Showing {len(df_table)} rows from `{table_choice}`.")
    else:
        st.error(f"Table `{table_choice}` not found.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Download Center
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "💾 Download Center":
    st.header("💾 SIPGD Benchmark Download Center")
    st.markdown("Download complete benchmark zip package and metadata files directly.")

    st.subheader("Download Complete Package")
    if ZIP_FILE.exists():
        with open(ZIP_FILE, "rb") as f:
            st.download_button(
                label="📦 DOWNLOAD COMPLETE DATASET (ZIP)",
                data=f,
                file_name="image_provenance_dataset.zip",
                mime="application/zip",
                use_container_width=True,
            )
    else:
        st.error("ZIP archive not built yet. Run `python setup_provenance_dataset.py`.")

    st.markdown("---")
    st.subheader("Individual Metadata Downloads")

    d1, d2, d3 = st.columns(3)
    with d1:
        if (META_DIR / "images.csv").exists():
            st.download_button("Download images.csv", data=open(META_DIR / "images.csv", "rb"), file_name="images.csv")
    with d2:
        if (META_DIR / "pairs_train.csv").exists():
            st.download_button("Download pairs_train.csv", data=open(META_DIR / "pairs_train.csv", "rb"), file_name="pairs_train.csv")
    with d3:
        if (META_DIR / "edges_train.csv").exists():
            st.download_button("Download edges_train.csv", data=open(META_DIR / "edges_train.csv", "rb"), file_name="edges_train.csv")
