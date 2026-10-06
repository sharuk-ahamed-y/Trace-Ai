"""
backend/app/services/report_generator.py — Standalone HTML/PDF Provenance Report Generator
"""

import json
from pathlib import Path
from typing import Dict, Any
from backend.app.config import REPORT_DIR

def generate_html_report(analysis_data: Dict[str, Any], output_path: Path) -> Path:
    analysis = analysis_data["analysis"]
    images = analysis_data["images"]
    edges = analysis_data["edges"]
    xai_results = analysis_data.get("xai_results", [])

    root_img = next((img for img in images if img["id"] == analysis.get("root_image_id")), images[0] if images else {})

    edges_html = ""
    for idx, e in enumerate(edges):
        edges_html += f"""
        <tr>
            <td>{idx + 1}</td>
            <td><code>{e['source_id']}</code></td>
            <td><code>{e['target_id']}</code></td>
            <td><span class="badge badge-transform">{e.get('transformation', 'N/A').upper()}</span></td>
            <td>{(e.get('confidence', 0.0) * 100):.1f}%</td>
            <td>{e.get('weight', 0.0):.4f}</td>
        </tr>
        """

    images_html = ""
    for img in images:
        is_root = "ROOT ORIGINAL" if img["id"] == analysis.get("root_image_id") else "DERIVED"
        badge_cls = "badge-root" if img["id"] == analysis.get("root_image_id") else "badge-derived"
        images_html += f"""
        <div class="image-card">
            <div class="card-header">
                <span class="node-id">{img['filename']}</span>
                <span class="badge {badge_cls}">{is_root}</span>
            </div>
            <div class="card-body">
                <p><strong>Dimensions:</strong> {img['width']} x {img['height']} px</p>
                <p><strong>File Size:</strong> {img['file_size'] / 1024:.1f} KB</p>
                <p><strong>SHA256:</strong> <small>{img['sha256'][:16]}...</small></p>
                <p><strong>MD5:</strong> <small>{img['md5']}</small></p>
            </div>
        </div>
        """

    xai_html = ""
    for x in xai_results:
        ev = x.get("evidence", {})
        findings = ev.get("empirical_findings", [])
        findings_html = "".join([f"<li>{f}</li>" for f in findings])
        heatmap_img = f'<img src="{x.get("heatmap_url", "")}" alt="Heatmap" style="max-width:100%; border-radius:8px; border:1px solid #00FF88;">' if x.get("heatmap_url") else ""

        xai_html += f"""
        <div class="xai-box">
            <h4>Edge Transformation Evidence: <code>{x['source_id']}</code> &rarr; <code>{x['target_id']}</code></h4>
            {heatmap_img}
            <div style="margin-top: 12px;">
                <p><strong>Confidence Score:</strong> {(ev.get('confidence', 0.85) * 100):.1f}%</p>
                <p><strong>Summary:</strong> {ev.get('summary', 'Provenential dependency detected.')}</p>
                <ul class="evidence-list">
                    {findings_html}
                </ul>
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>TRACEAI — Image Provenance Forensic Report ({analysis['id']})</title>
    <style>
        :root {{
            --bg-dark: #050505;
            --card-bg: #0A0D14;
            --border-color: #1E293B;
            --neon-green: #00FF88;
            --text-main: #E2E8F0;
            --text-muted: #94A3B8;
        }}
        body {{
            background-color: var(--bg-dark);
            color: var(--text-main);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 40px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        .header {{
            border-bottom: 2px solid var(--neon-green);
            padding-bottom: 20px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{
            color: var(--neon-green);
            margin: 0;
            font-size: 28px;
            letter-spacing: 1px;
        }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        .meta-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 15px;
            border-radius: 8px;
        }}
        .meta-card .label {{
            font-size: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
        }}
        .meta-card .val {{
            font-size: 20px;
            font-weight: bold;
            color: var(--neon-green);
            margin-top: 5px;
        }}
        .section-title {{
            color: var(--neon-green);
            border-left: 4px solid var(--neon-green);
            padding-left: 10px;
            font-size: 20px;
            margin-top: 40px;
            margin-bottom: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 30px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background-color: #0F172A;
            color: var(--neon-green);
            font-size: 13px;
            text-transform: uppercase;
        }}
        .badge {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        }}
        .badge-root {{ background: #00FF88; color: #000; }}
        .badge-derived {{ background: #3B82F6; color: #fff; }}
        .badge-transform {{ background: #8B5CF6; color: #fff; }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .image-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 15px;
        }}
        .xai-box {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .evidence-list {{
            color: var(--text-muted);
            padding-left: 20px;
        }}
        .footer {{
            margin-top: 60px;
            text-align: center;
            font-size: 12px;
            color: var(--text-muted);
            border-top: 1px solid var(--border-color);
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>TRACEAI FORENSIC REPORT</h1>
                <p style="margin: 5px 0 0 0; color: var(--text-muted);">Image Provenance Graph & Transformation Analysis</p>
            </div>
            <div style="text-align: right;">
                <p style="margin:0; font-weight:bold;">ID: {analysis['id']}</p>
                <p style="margin:0; font-size:12px; color:var(--text-muted);">{analysis['timestamp']}</p>
            </div>
        </div>

        <div class="meta-grid">
            <div class="meta-card">
                <div class="label">Total Images</div>
                <div class="val">{analysis['image_count']}</div>
            </div>
            <div class="meta-card">
                <div class="label">Identified Root Node</div>
                <div class="val">{analysis.get('root_image_id', 'N/A')}</div>
            </div>
            <div class="meta-card">
                <div class="label">Root Confidence</div>
                <div class="val">{(analysis.get('root_confidence', 0.95) * 100):.1f}%</div>
            </div>
            <div class="meta-card">
                <div class="label">Execution Time</div>
                <div class="val">{analysis.get('execution_time_ms', 0):.0f} ms</div>
            </div>
        </div>

        <div class="section-title">Reconstructed Provenance Graph Edges</div>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Source (Parent)</th>
                    <th>Target (Child)</th>
                    <th>Transformation</th>
                    <th>Confidence</th>
                    <th>Weight</th>
                </tr>
            </thead>
            <tbody>
                {edges_html}
            </tbody>
        </table>

        <div class="section-title">Image Node Metadata</div>
        <div class="image-grid">
            {images_html}
        </div>

        <div class="section-title">Explainable AI (XAI) Forensic Evidence</div>
        {xai_html if xai_html else '<p style="color:var(--text-muted);">No XAI evidence entries.</p>'}

        <div class="footer">
            <p>Generated automatically by <strong>TRACEAI Provenance Laboratory Engine</strong> — Powered by ViT-B/16 & Minimum Spanning Trees.</p>
        </div>
    </div>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
