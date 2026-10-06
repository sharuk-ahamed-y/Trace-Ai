import React, { useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import { Network, Info, Download, ZoomIn, ZoomOut, RefreshCw, Crown, GitBranch, Layers, Table, Sparkles } from 'lucide-react';

cytoscape.use(dagre);

export default function ProvenanceGraphViewer({ analysisData, setActiveTab }) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [selectedEdge, setSelectedEdge] = useState(null);
  const [layoutName, setLayoutName] = useState('dagre');

  useEffect(() => {
    if (!analysisData || !containerRef.current) return;

    const nodes = (analysisData.nodes || []).map(n => ({
      data: {
        id: n.id,
        label: n.filename || n.id,
        is_root: Boolean(n.is_root),
        url: n.url,
        width: n.width,
        height: n.height,
        file_size: n.file_size,
        generation: n.generation || 0
      }
    }));

    const edges = (analysisData.edges || []).map((e, idx) => ({
      data: {
        id: e.id || `edge_${idx}`,
        source: e.source || e.source_id,
        target: e.target || e.target_id,
        label: `${(e.transformation || 'TRANSFORM').toUpperCase()} (${((e.confidence || 0.9) * 100).toFixed(0)}%)`,
        transformation: e.transformation || 'crop',
        confidence: e.confidence || 0.90,
        weight: e.weight || 0.10
      }
    }));

    const cy = cytoscape({
      container: containerRef.current,
      elements: [...nodes, ...edges],
      style: [
        {
          selector: 'node',
          style: {
            'label': 'data(label)',
            'color': '#F8FAFC',
            'font-family': 'JetBrains Mono, monospace',
            'font-size': '11px',
            'text-valign': 'bottom',
            'text-margin-y': 6,
            'background-color': '#0F172A',
            'border-width': 2,
            'border-color': '#10B981',
            'width': 48,
            'height': 48,
            'overlay-padding': '4px'
          }
        },
        {
          selector: 'node[?is_root]',
          style: {
            'background-color': '#10B981',
            'border-color': '#FFFFFF',
            'border-width': 3,
            'color': '#10B981',
            'font-weight': 'bold',
            'width': 58,
            'height': 58,
            'shadow-blur': 15,
            'shadow-color': '#10B981'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 3,
            'line-color': '#3B82F6',
            'target-arrow-color': '#3B82F6',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(label)',
            'font-size': '10px',
            'font-family': 'JetBrains Mono, monospace',
            'color': '#94A3B8',
            'text-background-color': '#050505',
            'text-background-opacity': 0.85,
            'text-background-padding': '3px',
            'text-rotation': 'autorotate'
          }
        },
        {
          selector: 'node:selected',
          style: {
            'border-color': '#F59E0B',
            'border-width': 4
          }
        },
        {
          selector: 'edge:selected',
          style: {
            'line-color': '#F59E0B',
            'target-arrow-color': '#F59E0B',
            'width': 5
          }
        }
      ],
      layout: {
        name: layoutName,
        rankDir: 'TB',
        nodeSep: 60,
        rankSep: 80,
        animate: true
      }
    });

    cy.on('tap', 'node', (evt) => {
      setSelectedNode(evt.target.data());
      setSelectedEdge(null);
    });

    cy.on('tap', 'edge', (evt) => {
      setSelectedEdge(evt.target.data());
      setSelectedNode(null);
    });

    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        setSelectedNode(null);
        setSelectedEdge(null);
      }
    });

    cyRef.current = cy;

    return () => {
      cy.destroy();
    };
  }, [analysisData, layoutName]);

  if (!analysisData) {
    return (
      <div className="max-w-4xl mx-auto py-20 text-center space-y-4">
        <Network className="w-12 h-12 text-cyber-muted mx-auto" />
        <h3 className="text-lg font-semibold text-white">No Active Provenance Graph</h3>
        <p className="text-xs text-cyber-muted font-mono">Upload an image set or run a benchmark demo to visualize reconstructed trees.</p>
        <button
          onClick={() => setActiveTab('analysis')}
          className="bg-cyber-neon text-black font-semibold px-4 py-2 rounded-xl text-xs shadow-[0_0_12px_rgba(16,185,129,0.3)]"
        >
          Go to Upload Zone
        </button>
      </div>
    );
  }

  const rootNode = (analysisData.nodes || []).find(n => n.is_root || n.id === analysisData.root_image_id) || analysisData.nodes?.[0];
  const summary = analysisData.summary || {};
  const bfsLevels = analysisData.bfs || [];

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Top Banner Stats */}
      <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
        <div className="bg-cyber-card p-4 rounded-xl border border-cyber-border">
          <div className="text-[10px] text-cyber-muted font-mono uppercase">Total Uploads</div>
          <div className="text-xl font-bold text-white mt-1">{summary.total_images || analysisData.nodes?.length || 0}</div>
        </div>
        <div className="bg-cyber-card p-4 rounded-xl border border-cyber-border">
          <div className="text-[10px] text-cyber-muted font-mono uppercase">MST Edges</div>
          <div className="text-xl font-bold text-cyber-blue mt-1">{summary.mst_edges || analysisData.edges?.length || 0}</div>
        </div>
        <div className="bg-cyber-card p-4 rounded-xl border border-cyber-border">
          <div className="text-[10px] text-cyber-muted font-mono uppercase">Rejected Pairs</div>
          <div className="text-xl font-bold text-amber-400 mt-1">{summary.rejected_pairs || 0}</div>
        </div>
        <div className="bg-cyber-card p-4 rounded-xl border border-cyber-border">
          <div className="text-[10px] text-cyber-muted font-mono uppercase">Identified Root</div>
          <div className="text-sm font-bold text-cyber-neon truncate mt-1">{analysisData.root_image_id || 'N/A'}</div>
        </div>
        <div className="bg-cyber-card p-4 rounded-xl border border-cyber-border">
          <div className="text-[10px] text-cyber-muted font-mono uppercase">Root Confidence</div>
          <div className="text-xl font-bold text-cyber-neon mt-1">{((analysisData.root_confidence || 0.95) * 100).toFixed(1)}%</div>
        </div>
        <div className="bg-cyber-card p-4 rounded-xl border border-cyber-border">
          <div className="text-[10px] text-cyber-muted font-mono uppercase">BFS Max Depth</div>
          <div className="text-xl font-bold text-purple-400 mt-1">Level {summary.max_bfs_depth || 0}</div>
        </div>
      </div>

      {/* SECTION 1: Root Original Image Card */}
      <div className="bg-gradient-to-r from-[#064E3B]/40 via-cyber-card to-cyber-card p-6 rounded-2xl border border-cyber-neon/40 flex flex-col md:flex-row items-center justify-between gap-6 shadow-[0_0_20px_rgba(16,185,129,0.15)]">
        <div className="flex items-center space-x-4">
          <div className="w-14 h-14 rounded-2xl bg-cyber-neon/20 border-2 border-cyber-neon text-cyber-neon flex items-center justify-center flex-shrink-0 shadow-[0_0_15px_rgba(16,185,129,0.4)]">
            <Crown className="w-7 h-7" />
          </div>
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <span className="text-xs font-mono font-bold bg-cyber-neon text-black px-2 py-0.5 rounded-full uppercase">
                Most Likely Original Root
              </span>
              <span className="text-xs font-mono text-cyber-neon font-bold">
                {((analysisData.root_confidence || 0.95) * 100).toFixed(1)}% Confidence
              </span>
            </div>
            <h3 className="text-lg font-bold text-white tracking-tight">{analysisData.root_image_id}</h3>
            <p className="text-xs text-cyber-muted font-mono max-w-2xl leading-relaxed">
              {analysisData.root_reason || analysisData.xai?.root_explanation || "Determined via indegree=0 graph structure and maximal net outgoing directional confidence."}
            </p>
          </div>
        </div>
        <button
          onClick={() => setActiveTab('xai')}
          className="bg-cyber-neon/10 hover:bg-cyber-neon text-cyber-neon hover:text-black font-semibold text-xs px-4 py-2.5 rounded-xl border border-cyber-neon/40 transition-all flex items-center space-x-2 flex-shrink-0"
        >
          <Sparkles className="w-4 h-4" />
          <span>View XAI Evidence</span>
        </button>
      </div>

      {/* SECTION 2: Main Cytoscape Graph & Inspection Sidebar */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Cytoscape Canvas */}
        <div className="lg:col-span-3 bg-cyber-card rounded-2xl border border-cyber-border relative h-[580px] overflow-hidden">
          {/* Controls Bar */}
          <div className="absolute top-4 left-4 z-10 flex items-center space-x-2 bg-cyber-dark/90 p-1.5 rounded-xl border border-cyber-border backdrop-blur">
            <button
              onClick={() => cyRef.current?.zoom(cyRef.current.zoom() * 1.2)}
              className="p-1.5 hover:bg-cyber-border/60 rounded text-cyber-muted hover:text-white"
              title="Zoom In"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={() => cyRef.current?.zoom(cyRef.current.zoom() * 0.8)}
              className="p-1.5 hover:bg-cyber-border/60 rounded text-cyber-muted hover:text-white"
              title="Zoom Out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <button
              onClick={() => cyRef.current?.fit()}
              className="p-1.5 hover:bg-cyber-border/60 rounded text-cyber-muted hover:text-white"
              title="Fit View"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <div className="h-4 w-px bg-cyber-border"></div>
            <select
              value={layoutName}
              onChange={(e) => setLayoutName(e.target.value)}
              className="bg-cyber-dark text-xs font-mono text-cyber-muted border-none rounded px-2 py-1 focus:outline-none"
            >
              <option value="dagre">Hierarchical Tree (Dagre)</option>
              <option value="concentric">Concentric Circles</option>
              <option value="cose">Force Directed (CoSe)</option>
              <option value="breadthfirst">Breadth-First</option>
            </select>
          </div>

          <div className="absolute top-4 right-4 z-10">
            <a
              href={`/api/download/${analysisData.analysis_id}/html`}
              target="_blank"
              rel="noreferrer"
              className="flex items-center space-x-2 bg-cyber-neon text-black font-semibold text-xs px-3 py-2 rounded-xl shadow-[0_0_10px_rgba(16,185,129,0.3)] hover:bg-cyber-neon/90"
            >
              <Download className="w-4 h-4" />
              <span>Export HTML Report</span>
            </a>
          </div>

          <div ref={containerRef} className="w-full h-full bg-[#030508]" />
        </div>

        {/* Sidebar Inspector */}
        <div className="lg:col-span-1 bg-cyber-card p-5 rounded-2xl border border-cyber-border space-y-4 font-mono text-xs">
          <div className="flex items-center space-x-2 text-white border-b border-cyber-border pb-3">
            <Info className="w-4 h-4 text-cyber-neon" />
            <span className="font-semibold text-sm">Node & Edge Inspector</span>
          </div>

          {selectedNode ? (
            <div className="space-y-3">
              <div className="p-3 bg-cyber-dark rounded-xl border border-cyber-border space-y-1">
                <span className="text-[10px] text-cyber-muted uppercase">Selected Node ID</span>
                <p className="font-bold text-cyber-neon text-sm">{selectedNode.id}</p>
                {selectedNode.is_root && (
                  <span className="inline-block bg-cyber-neon/20 text-cyber-neon text-[10px] px-2 py-0.5 rounded border border-cyber-neon/40 font-bold mt-1">
                    ROOT ORIGINAL IMAGE
                  </span>
                )}
              </div>
              <div className="space-y-1.5 text-cyber-muted">
                <p><strong className="text-white">Filename:</strong> {selectedNode.label}</p>
                <p><strong className="text-white">Dimensions:</strong> {selectedNode.width} × {selectedNode.height} px</p>
                <p><strong className="text-white">File Size:</strong> {(selectedNode.file_size / 1024).toFixed(1)} KB</p>
                <p><strong className="text-white">Generation:</strong> Level {selectedNode.generation}</p>
              </div>
              <button
                onClick={() => setActiveTab('xai')}
                className="w-full mt-3 bg-cyber-blue/20 text-cyber-blue hover:bg-cyber-blue hover:text-black py-2 rounded-xl border border-cyber-blue/40 font-bold transition-all text-xs"
              >
                Inspect XAI Evidence
              </button>
            </div>
          ) : selectedEdge ? (
            <div className="space-y-3">
              <div className="p-3 bg-cyber-dark rounded-xl border border-cyber-border space-y-1">
                <span className="text-[10px] text-cyber-muted uppercase">Directed Relation Edge</span>
                <p className="font-bold text-cyber-blue text-sm">{selectedEdge.source} &rarr; {selectedEdge.target}</p>
              </div>
              <div className="space-y-1.5 text-cyber-muted">
                <p><strong className="text-white">Transformation:</strong> <span className="text-cyber-neon font-bold uppercase">{selectedEdge.transformation}</span></p>
                <p><strong className="text-white">Edge Confidence:</strong> {(selectedEdge.confidence * 100).toFixed(1)}%</p>
              </div>
              <button
                onClick={() => setActiveTab('xai')}
                className="w-full mt-3 bg-cyber-neon text-black py-2 rounded-xl font-bold transition-all text-xs"
              >
                View Heatmap & Evidence
              </button>
            </div>
          ) : (
            <div className="py-12 text-center text-cyber-muted text-xs space-y-2">
              <p>Click on any graph node or edge to inspect detailed metadata and forensic metrics.</p>
            </div>
          )}
        </div>
      </div>

      {/* SECTION 3: BFS Traversal & Generation Hierarchy */}
      <div className="bg-cyber-card p-6 rounded-2xl border border-cyber-border space-y-4">
        <div className="flex items-center space-x-2 text-white border-b border-cyber-border pb-3">
          <Layers className="w-5 h-5 text-purple-400" />
          <h3 className="font-bold text-base">BFS Provenance Traversal & Generation Hierarchy</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {bfsLevels.map((lvl) => (
            <div key={lvl.level} className="bg-cyber-dark/80 p-4 rounded-xl border border-cyber-border space-y-2 font-mono">
              <div className="flex items-center justify-between text-xs border-b border-cyber-border/60 pb-2">
                <span className="font-bold text-purple-400 uppercase">Generation {lvl.level}</span>
                <span className="text-[10px] text-cyber-muted">{lvl.images.length} Image(s)</span>
              </div>
              <div className="space-y-1.5">
                {lvl.images.map((imgId) => (
                  <div
                    key={imgId}
                    className={`p-2 rounded border text-xs truncate flex items-center justify-between ${
                      imgId === analysisData.root_image_id
                        ? 'bg-cyber-neon/10 border-cyber-neon text-cyber-neon font-bold'
                        : 'bg-cyber-card border-cyber-border text-slate-200'
                    }`}
                  >
                    <span className="truncate">{imgId}</span>
                    {imgId === analysisData.root_image_id && (
                      <span className="text-[10px] uppercase text-cyber-neon font-bold ml-1">ROOT</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* SECTION 4: Image Relationship Table */}
      <div className="bg-cyber-card p-6 rounded-2xl border border-cyber-border space-y-4">
        <div className="flex items-center space-x-2 text-white border-b border-cyber-border pb-3">
          <Table className="w-5 h-5 text-cyber-blue" />
          <h3 className="font-bold text-base">Reconstructed Edge Relation Matrix</h3>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr className="border-b border-cyber-border text-cyber-muted uppercase text-[10px]">
                <th className="py-2.5 px-3">Parent (Source)</th>
                <th className="py-2.5 px-3">Child (Target)</th>
                <th className="py-2.5 px-3">Predicted Transformation</th>
                <th className="py-2.5 px-3">Relationship Confidence</th>
                <th className="py-2.5 px-3">Direction Confidence</th>
                <th className="py-2.5 px-3">Overall Edge Confidence</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-cyber-border/40 text-slate-300">
              {(analysisData.edges || []).map((e, idx) => (
                <tr key={idx} className="hover:bg-cyber-dark/60 transition-colors">
                  <td className="py-2.5 px-3 font-bold text-cyber-neon">{e.source}</td>
                  <td className="py-2.5 px-3 font-bold text-cyber-blue">{e.target}</td>
                  <td className="py-2.5 px-3">
                    <span className="px-2 py-0.5 rounded bg-cyber-neon/10 text-cyber-neon border border-cyber-neon/30 uppercase font-bold text-[10px]">
                      {e.transformation}
                    </span>
                  </td>
                  <td className="py-2.5 px-3">{((e.related_prob || e.confidence) * 100).toFixed(1)}%</td>
                  <td className="py-2.5 px-3">{((e.dir_conf || e.confidence) * 100).toFixed(1)}%</td>
                  <td className="py-2.5 px-3 font-bold text-white">{((e.confidence || 0.9) * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
