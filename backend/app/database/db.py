"""
backend/app/database/db.py — SQLite Database Connection & Schema Initialization
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.app.config import DB_PATH

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Table 1: Analyses
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analyses (
        id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        status TEXT NOT NULL,
        image_count INTEGER NOT NULL,
        root_image_id TEXT,
        root_confidence REAL,
        execution_time_ms REAL,
        similarity_threshold REAL,
        confidence_filter REAL
    )
    """)

    # Table 2: Images
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id TEXT NOT NULL,
        analysis_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        url TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        md5 TEXT NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        file_size INTEGER NOT NULL,
        is_root BOOLEAN DEFAULT 0,
        PRIMARY KEY (analysis_id, id),
        FOREIGN KEY (analysis_id) REFERENCES analyses (id) ON DELETE CASCADE
    )
    """)

    # Table 3: Graph Edges
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS graph_edges (
        id TEXT PRIMARY KEY,
        analysis_id TEXT NOT NULL,
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        transformation TEXT NOT NULL,
        confidence REAL NOT NULL,
        weight REAL NOT NULL,
        parameters TEXT,
        FOREIGN KEY (analysis_id) REFERENCES analyses (id) ON DELETE CASCADE
    )
    """)

    # Table 4: XAI Results
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS xai_results (
        id TEXT PRIMARY KEY,
        analysis_id TEXT NOT NULL,
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        heatmap_url TEXT,
        evidence_json TEXT NOT NULL,
        FOREIGN KEY (analysis_id) REFERENCES analyses (id) ON DELETE CASCADE
    )
    """)

    conn.commit()
    conn.close()

def save_analysis(
    analysis_id: str,
    timestamp: str,
    status: str,
    image_count: int,
    root_image_id: Optional[str],
    root_confidence: float,
    execution_time_ms: float,
    similarity_threshold: float,
    confidence_filter: float,
    images: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    xai_results: List[Dict[str, Any]]
):
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Insert analysis
    cursor.execute("""
    INSERT OR REPLACE INTO analyses (
        id, timestamp, status, image_count, root_image_id, root_confidence, execution_time_ms, similarity_threshold, confidence_filter
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (analysis_id, timestamp, status, image_count, root_image_id, root_confidence, execution_time_ms, similarity_threshold, confidence_filter))

    # Delete existing items if re-analyzing
    cursor.execute("DELETE FROM images WHERE analysis_id = ?", (analysis_id,))
    cursor.execute("DELETE FROM graph_edges WHERE analysis_id = ?", (analysis_id,))
    cursor.execute("DELETE FROM xai_results WHERE analysis_id = ?", (analysis_id,))

    # Insert images
    for img in images:
        cursor.execute("""
        INSERT OR REPLACE INTO images (id, analysis_id, filename, file_path, url, sha256, md5, width, height, file_size, is_root)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            img["id"], analysis_id, img["filename"], img["file_path"], img["url"],
            img["sha256"], img["md5"], img["width"], img["height"], img["file_size"],
            1 if img["id"] == root_image_id else 0
        ))

    # Insert graph_edges
    for idx, e in enumerate(edges):
        edge_id = f"edge_{analysis_id}_{idx}"
        params_str = json.dumps(e.get("parameters", {}))
        cursor.execute("""
        INSERT INTO graph_edges (id, analysis_id, source_id, target_id, transformation, confidence, weight, parameters)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            edge_id, analysis_id, e["source"], e["target"],
            e.get("transformation", "unknown"), e.get("confidence", 0.0),
            e.get("weight", 0.0), params_str
        ))

    # Insert xai_results
    for idx, x in enumerate(xai_results):
        xai_id = f"xai_{analysis_id}_{idx}"
        evidence_str = json.dumps(x.get("evidence", {}))
        cursor.execute("""
        INSERT INTO xai_results (id, analysis_id, source_id, target_id, heatmap_url, evidence_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            xai_id, analysis_id, x["source_id"], x["target_id"],
            x.get("heatmap_url", ""), evidence_str
        ))

    conn.commit()
    conn.close()

def get_analysis_by_id(analysis_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
    analysis_row = cursor.fetchone()
    if not analysis_row:
        conn.close()
        return None
    
    analysis = dict(analysis_row)

    cursor.execute("SELECT * FROM images WHERE analysis_id = ?", (analysis_id,))
    images = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM graph_edges WHERE analysis_id = ?", (analysis_id,))
    raw_edges = [dict(r) for r in cursor.fetchall()]
    edges = []
    for e in raw_edges:
        try:
            e["parameters"] = json.loads(e["parameters"]) if e["parameters"] else {}
        except Exception:
            e["parameters"] = {}
        edges.append(e)

    cursor.execute("SELECT * FROM xai_results WHERE analysis_id = ?", (analysis_id,))
    raw_xai = [dict(r) for r in cursor.fetchall()]
    xai_results = []
    for x in raw_xai:
        try:
            x["evidence"] = json.loads(x["evidence_json"]) if x["evidence_json"] else {}
        except Exception:
            x["evidence"] = {}
        xai_results.append(x)

    conn.close()
    return {
        "analysis": analysis,
        "images": images,
        "edges": edges,
        "xai_results": xai_results
    }

def get_all_analyses(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analyses ORDER BY timestamp DESC LIMIT ? OFFSET ?", (limit, offset))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
