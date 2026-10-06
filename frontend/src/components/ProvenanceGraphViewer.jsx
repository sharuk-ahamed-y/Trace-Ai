import React, { useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import { Network, Info, Download, ZoomIn, ZoomOut, RefreshCw, ShieldAlert, Cpu } from 'lucide-react';
import axios from 'axios';

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
        sha256: n.sha256,
        md5: n.md5,
        generation_level: n.generation_level || 0
      }
    }));

    const edges = (analysisData.edges || []).map((e, idx) => ({
      data: {
        id: e.id || `edge_${idx}`,
        source: e.source || e.source_id,
        target: e.target || e.target_id,
        label: (e.transformation || 'TRANSFORM').toUpperCase(),
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
            'border-color': '#00E5FF',
            'width': 45,
            'height': 45,
            'overlay-padding': '4px'
          }
        },
        {
          selector: 'node[?is_root]',
          style: {
            'background-color': '#00FF88',
            'border-color': '#FFFFFF',
            'border-width': 3,
            'color': '#00FF88',
            'font-weight': 'bold',
            'width': 55,
            'height': 55,
            'shadow-blur': 15,
            'shadow-color': '#00FF88'
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
            'text-background-opacity': 0.8,
            'text-background-padding': '3px',
            'text-rotation': 'autorotate'
          }
        },
        {
          selector: 'node:selected',
          style: {
            'border-color': '#FF007F',
            'border-width': 4
          }
        },
        {
          selector: 'edge:selected',
          style: {
            'line-color': '#FF007F',
            'target-arrow-color': '#FF007F',
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
          className="bg-cyber-neon text-black font-semibold px-4 py-2 rounded-xl text-xs"
        >
          Go to Upload Zone
        </button>
      </div>
    );
  }

  const rootNode = (analysisData.nodes || []).find(n => n.id === analysisData.root_image_id) || analysisData.nodes?.[0];

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Top Banner Stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="bg-cyber-card p-4 rounded-xl border border-cyber-border">
          <div className="text-[10px] text-cyber-muted font-mono uppercase">Total Nodes</div>
          <div className="text-xl font-bold text-white mt-1">{analysisData.nodes?.length || 0}</div>
        </div>
        <div className="bg-cyber-card p-4 rounded-xl border border-cyber-border">
          <div className="text-[10px] text-cyber-muted font-mono uppercase">Reconstructed Edges</div>
          <div className="text-xl font-bold text-cyber-blue mt-1">{analysisData.edges?.length || 0}</div>
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
          <div className="text-[10px] text-cyber-muted font-mono uppercase">Execution Speed</div>
          <div className="text-xl font-bold text-white mt-1">{analysisData.execution_time_ms || 0} ms</div>
        </div>
      </div>

      {/* Main Canvas & Inspection Sidebar */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Cytoscape Canvas */}
        <div className="lg:col-span-3 bg-cyber-card rounded-2xl border border-cyber-border relative h-[600px] overflow-hidden">
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
              className="flex items-center space-x-2 bg-cyber-neon text-black font-semibold text-xs px-3 py-2 rounded-xl shadow-[0_0_10px_rgba(0,255,136,0.3)] hover:bg-cyber-neon/90"
            >
              <Download className="w-4 h-4" />
              <span>Download Report</span>
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
                <p><strong className="text-white">Gen Level:</strong> {selectedNode.generation_level}</p>
                <p className="truncate"><strong className="text-white">SHA256:</strong> {selectedNode.sha256?.substring(0, 16)}...</p>
                <p><strong className="text-white">MD5:</strong> {selectedNode.md5}</p>
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
                <p><strong className="text-white">Confidence:</strong> {(selectedEdge.confidence * 100).toFixed(1)}%</p>
                <p><strong className="text-white">MST Weight:</strong> {selectedEdge.weight.toFixed(4)}</p>
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
    </div>
  );
}
