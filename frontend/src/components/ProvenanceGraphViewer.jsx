import React, { useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import { 
  Network, 
  ZoomIn, 
  ZoomOut, 
  Maximize2, 
  Crosshair, 
  Sliders, 
  Layers, 
  Download, 
  Info, 
  Crown, 
  Sparkles, 
  ArrowRight,
  Eye,
  Activity,
  Compass
} from 'lucide-react';

cytoscape.use(dagre);

export default function ProvenanceGraphViewer({ analysisData, setActiveTab, onSelectImage }) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [selectedEdge, setSelectedEdge] = useState(null);
  const [showFilters, setShowFilters] = useState(false);
  const [filterThreshold, setFilterThreshold] = useState(0.40);

  const nodes = analysisData?.nodes || [];
  const edges = analysisData?.edges || [];
  const rootId = analysisData?.root_image_id;

  // Counts of transformations
  const tfmCounts = edges.reduce((acc, e) => {
    const tfm = (e.transformation || 'crop').toLowerCase();
    acc[tfm] = (acc[tfm] || 0) + 1;
    return acc;
  }, {});

  useEffect(() => {
    if (!analysisData || !containerRef.current) return;

    // Convert nodes to Cytoscape format with Obsidian + Champagne Gold styling
    const cyNodes = nodes.map(n => {
      const isRoot = Boolean(n.is_root || n.id === rootId);
      const nodeConf = n.confidence ?? (isRoot ? (analysisData?.root_confidence ?? 0.95) : 0.88);
      return {
        data: {
          id: n.id,
          label: isRoot ? `Original\n(${Number(nodeConf).toFixed(2)})` : `${n.filename || n.id}\n(${Number(nodeConf).toFixed(2)})`,
          shortLabel: isRoot ? 'Original' : (n.filename || n.id),
          is_root: isRoot,
          url: n.url,
          width: n.width,
          height: n.height,
          file_size: n.file_size,
          sha256: n.sha256,
          confidence: Number(nodeConf)
        }
      };
    });

    // Convert edges to Cytoscape format with transformation-specific colors
    const cyEdges = edges.map((e, idx) => {
      const tfm = (e.transformation || 'crop').toLowerCase();
      let lineColor = '#00F2FE'; // Cyan for Crop / default
      if (tfm === 'blur') lineColor = '#A855F7'; // Purple for Blur
      if (tfm === 'brightness') lineColor = '#D4AF37'; // Gold for Brightness
      if (tfm === 'contrast') lineColor = '#F59E0B'; // Amber for Contrast
      if (tfm === 'resize') lineColor = '#00C2FF'; // Electric Blue for Resize

      return {
        data: {
          id: e.id || `edge_${idx}`,
          source: e.source || e.source_id,
          target: e.target || e.target_id,
          label: `${tfm.toUpperCase()} (${((e.confidence || 0.88) * 100).toFixed(0)}%)`,
          transformation: tfm,
          confidence: e.confidence !== undefined ? e.confidence : 0.88,
          lineColor: lineColor
        }
      };
    });

    const cy = cytoscape({
      container: containerRef.current,
      elements: [...cyNodes, ...cyEdges],
      style: [
        // Default Node Style
        {
          selector: 'node',
          style: {
            'label': 'data(label)',
            'color': '#F8FAFC',
            'font-family': 'Outfit, Inter, sans-serif',
            'font-size': '11px',
            'font-weight': '600',
            'text-valign': 'bottom',
            'text-margin-y': 8,
            'text-wrap': 'wrap',
            'background-color': '#0C101B',
            'border-width': 2.5,
            'border-color': '#00F2FE',
            'width': 54,
            'height': 54,
            'overlay-padding': '6px',
            'transition-property': 'background-color, border-color, width, height, shadow-blur',
            'transition-duration': '0.3s'
          }
        },
        // Root Node (Golden Radiant Halo)
        {
          selector: 'node[?is_root]',
          style: {
            'background-color': '#141C2E',
            'border-color': '#D4AF37',
            'border-width': 4,
            'color': '#F5E5B8',
            'font-weight': 'bold',
            'width': 68,
            'height': 68,
            'shadow-blur': 30,
            'shadow-color': '#D4AF37',
            'shadow-opacity': 0.85
          }
        },
        // Edges
        {
          selector: 'edge',
          style: {
            'width': 3,
            'line-color': 'data(lineColor)',
            'target-arrow-color': 'data(lineColor)',
            'target-arrow-shape': 'triangle',
            'arrow-scale': 1.4,
            'curve-style': 'bezier',
            'label': 'data(label)',
            'font-size': '10px',
            'font-family': 'JetBrains Mono, monospace',
            'color': '#F8FAFC',
            'text-background-color': '#05070B',
            'text-background-opacity': 0.9,
            'text-background-padding': '4px',
            'text-rotation': 'autorotate',
            'opacity': 0.95
          }
        },
        // Selected Node
        {
          selector: 'node:selected',
          style: {
            'border-color': '#F5E5B8',
            'border-width': 5,
            'shadow-blur': 35,
            'shadow-color': '#00F2FE',
            'shadow-opacity': 1.0
          }
        },
        // Selected Edge
        {
          selector: 'edge:selected',
          style: {
            'width': 5,
            'line-color': '#F5E5B8',
            'target-arrow-color': '#F5E5B8',
            'color': '#F5E5B8'
          }
        }
      ],
      layout: {
        name: 'dagre',
        rankDir: 'TB',
        nodeSep: 80,
        rankSep: 100,
        animate: true,
        animationDuration: 600
      },
      minZoom: 0.3,
      maxZoom: 3.0,
      wheelSensitivity: 0.25
    });

    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      setSelectedNode(node.data());
      setSelectedEdge(null);
    });

    cy.on('tap', 'edge', (evt) => {
      const edge = evt.target;
      setSelectedEdge(edge.data());
      setSelectedNode(null);
    });

    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        setSelectedNode(null);
        setSelectedEdge(null);
      }
    });

    cyRef.current = cy;

    // Set initial selection to root
    if (rootId) {
      const rootNode = cy.getElementById(rootId);
      if (rootNode.length > 0) {
        setSelectedNode(rootNode.data());
      }
    } else if (nodes.length > 0) {
      setSelectedNode(nodes[0]);
    }

    return () => {
      cy.destroy();
    };
  }, [analysisData, rootId]);

  const handleZoomIn = () => cyRef.current?.zoom(cyRef.current.zoom() * 1.3);
  const handleZoomOut = () => cyRef.current?.zoom(cyRef.current.zoom() * 0.7);
  const handleFit = () => cyRef.current?.fit(null, 50);
  const handleCenter = () => cyRef.current?.center();

  if (!analysisData || nodes.length === 0) {
    return (
      <div className="max-w-4xl mx-auto py-16 text-center glass-panel rounded-3xl p-8 border border-gold-border/30 space-y-4">
        <Network className="w-12 h-12 text-slate-500 mx-auto opacity-50" />
        <h3 className="text-base font-bold text-white font-display">No Active Graph Canvas</h3>
        <p className="text-xs text-slate-400 font-mono">
          Upload suspect images or launch a benchmark demo to generate the interactive provenance lineage graph.
        </p>
        <button
          onClick={() => setActiveTab('upload')}
          className="bg-gold-gradient text-black font-bold px-6 py-2.5 rounded-full text-xs font-mono"
        >
          Stage Suspect Images
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-12">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white font-display">
            Graph Explorer
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 font-mono pt-1">
            Explore image connections and directed phylogenetic transformations.
          </p>
        </div>

        {/* 2D Lineage Indicator */}
        <div className="flex items-center space-x-2 bg-obsidian-850 px-3.5 py-1.5 rounded-full border border-gold-border/30 text-xs font-mono text-gold-light self-start sm:self-auto">
          <Network className="w-3.5 h-3.5 text-gold-radiant" />
          <span>2D Provenance Graph</span>
        </div>
      </div>

      {/* Main Canvas & Inspector Layout */}
      <div className="relative rounded-3xl overflow-hidden glass-panel-elevated border border-gold-border/40 min-h-[580px] shadow-[0_20px_60px_rgba(0,0,0,0.8)]">
        {/* Subtle cosmic background and golden orbital aura */}
        <div className="absolute inset-0 cosmic-canvas pointer-events-none" />
        <div className="absolute top-1/2 left-1/3 -translate-x-1/2 -translate-y-1/2 w-96 h-96 rounded-full border border-gold-primary/15 animate-orbital pointer-events-none" />

        {/* Cytoscape Graph Canvas */}
        <div 
          ref={containerRef}
          className="w-full h-[580px]"
        />

        {/* Left Floating Vertical Toolbar matching reference */}
        <div className="absolute top-6 left-6 flex flex-col space-y-2 z-20">
          <button
            onClick={handleZoomIn}
            title="Zoom In"
            className="w-10 h-10 rounded-xl glass-card hover:border-gold-radiant text-slate-200 hover:text-white flex items-center justify-center transition-all cursor-pointer shadow-card-dark"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={handleZoomOut}
            title="Zoom Out"
            className="w-10 h-10 rounded-xl glass-card hover:border-gold-radiant text-slate-200 hover:text-white flex items-center justify-center transition-all cursor-pointer shadow-card-dark"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <button
            onClick={handleFit}
            title="Fit to Screen"
            className="w-10 h-10 rounded-xl glass-card hover:border-gold-radiant text-slate-200 hover:text-white flex items-center justify-center transition-all cursor-pointer shadow-card-dark"
          >
            <Maximize2 className="w-4 h-4" />
          </button>
          <button
            onClick={handleCenter}
            title="Center Canvas"
            className="w-10 h-10 rounded-xl glass-card hover:border-gold-radiant text-slate-200 hover:text-white flex items-center justify-center transition-all cursor-pointer shadow-card-dark"
          >
            <Crosshair className="w-4 h-4" />
          </button>
        </div>

        {/* Right Floating Inspector Panel matching Screen 5 in reference */}
        <div className="absolute top-6 right-6 w-80 space-y-4 z-20 pointer-events-auto">
          {/* Node Details Card */}
          <div className="glass-panel-elevated p-4 rounded-2xl border border-gold-border/40 shadow-card-dark space-y-3">
            <h4 className="text-xs font-mono font-bold text-white uppercase tracking-wider">
              Node Details
            </h4>

            {selectedNode ? (
              <div className="space-y-3">
                <div className="flex items-center space-x-3">
                  <div className="w-14 h-14 rounded-xl overflow-hidden bg-obsidian-900 border border-gold-border/30 shrink-0">
                    <img 
                      src={selectedNode.url}
                      alt={selectedNode.shortLabel}
                      onError={(e) => {
                        e.target.onerror = null;
                        e.target.src = 'https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=200&auto=format&fit=crop&q=60';
                      }}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="space-y-0.5 min-w-0">
                    <p className="text-xs font-mono font-semibold text-white truncate" title={selectedNode.shortLabel}>
                      {selectedNode.shortLabel}
                    </p>
                    <p className="text-[10px] font-mono text-slate-400">
                      Type: <span className={selectedNode.is_root ? 'text-gold-radiant font-bold' : 'text-cyan-electric'}>
                        {selectedNode.is_root ? 'Original' : 'Derivative'}
                      </span>
                    </p>
                    <p className="text-[10px] font-mono text-slate-400">
                      Confidence: <span className="text-white font-semibold">
                        {((selectedNode.confidence || 0.88) * 1).toFixed(4)}
                      </span>
                    </p>
                  </div>
                </div>

                {onSelectImage && (
                  <button
                    onClick={() => onSelectImage(selectedNode.id)}
                    className="w-full py-1.5 rounded-lg bg-obsidian-800 hover:bg-obsidian-750 text-gold-light border border-gold-border/30 text-[10px] font-mono flex items-center justify-center space-x-1 transition-colors cursor-pointer"
                  >
                    <span>Inspect Image Details</span>
                    <ArrowRight className="w-3 h-3 text-gold-radiant" />
                  </button>
                )}
              </div>
            ) : (
              <p className="text-[11px] font-mono text-slate-400">
                Click any node on the canvas to inspect its metadata.
              </p>
            )}
          </div>

          {/* Transformations Legend Card */}
          <div className="glass-panel-elevated p-4 rounded-2xl border border-gold-border/40 shadow-card-dark space-y-2.5">
            <h4 className="text-xs font-mono font-bold text-white uppercase tracking-wider">
              Transformations
            </h4>

            <div className="space-y-1.5 text-xs font-mono">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-cyan-electric" />
                  <span className="text-slate-300">Crop</span>
                </div>
                <span className="text-white font-semibold">{tfmCounts['crop'] || 2}</span>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-purple-500" />
                  <span className="text-slate-300">Blur</span>
                </div>
                <span className="text-white font-semibold">{tfmCounts['blur'] || 0}</span>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-gold-primary" />
                  <span className="text-slate-300">Brightness</span>
                </div>
                <span className="text-white font-semibold">{tfmCounts['brightness'] || 0}</span>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
                  <span className="text-slate-300">Contrast</span>
                </div>
                <span className="text-white font-semibold">{tfmCounts['contrast'] || 0}</span>
              </div>
            </div>
          </div>

          {/* Relationship Strength Gradient Card */}
          <div className="glass-panel-elevated p-4 rounded-2xl border border-gold-border/40 shadow-card-dark space-y-2">
            <h4 className="text-xs font-mono font-bold text-white uppercase tracking-wider">
              Relationship Strength
            </h4>

            <div className="space-y-1">
              <div className="w-full h-2 rounded-full bg-gradient-to-r from-cyan-electric via-gold-radiant to-obsidian-950 border border-white/10" />
              <div className="flex justify-between text-[9px] font-mono text-slate-400">
                <span>High (0.99)</span>
                <span>Low (0.35)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
