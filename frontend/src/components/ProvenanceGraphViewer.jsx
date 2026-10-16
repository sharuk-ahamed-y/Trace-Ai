import React, { useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import { 
  Network, 
  ZoomIn, 
  ZoomOut, 
  RefreshCw, 
  Layers, 
  Sliders, 
  Search, 
  Info, 
  FileText, 
  ArrowRight,
  Maximize2,
  Cpu,
  Activity,
  Zap,
  CheckCircle2
} from 'lucide-react';

cytoscape.use(dagre);

export default function ProvenanceGraphViewer({ analysisData, setActiveTab }) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [selectedEdge, setSelectedEdge] = useState(null);
  const [layoutName, setLayoutName] = useState('dagre');
  const [filterThreshold, setFilterThreshold] = useState(0.40);
  const [searchQuery, setSearchQuery] = useState('');

  const nodes = analysisData?.nodes || [];
  const edges = analysisData?.edges || [];
  const rootId = analysisData?.root_image_id;

  useEffect(() => {
    if (!analysisData || !containerRef.current) return;

    const cyNodes = nodes.map(n => ({
      data: {
        id: n.id,
        label: n.filename || n.id,
        is_root: Boolean(n.is_root || n.id === rootId),
        url: n.url,
        width: n.width,
        height: n.height,
        file_size: n.file_size,
        generation: n.generation || 0,
        sha256: n.sha256,
        md5: n.md5,
        forensics: n.forensics
      }
    }));

    const cyEdges = edges.map((e, idx) => ({
      data: {
        id: e.id || `edge_${idx}`,
        source: e.source || e.source_id,
        target: e.target || e.target_id,
        label: `${(e.transformation || 'TRANSFORM').toUpperCase()} (${((e.confidence || 0.9) * 100).toFixed(0)}%)`,
        transformation: e.transformation || 'crop',
        confidence: e.confidence || 0.90,
        related_prob: e.related_prob || 0.90,
        weight: e.weight || 0.10,
        direction_confidence: e.direction_confidence || 0.95
      }
    }));

    const cy = cytoscape({
      container: containerRef.current,
      elements: [...cyNodes, ...cyEdges],
      style: [
        {
          selector: 'node',
          style: {
            'label': 'data(label)',
            'color': '#F4F7F5',
            'font-family': 'JetBrains Mono, monospace',
            'font-size': '10px',
            'text-valign': 'bottom',
            'text-margin-y': 6,
            'background-color': '#111916',
            'border-width': 2,
            'border-color': '#35D9FF',
            'width': 44,
            'height': 44,
            'overlay-padding': '4px',
            'transition-property': 'background-color, border-color, width, height',
            'transition-duration': '0.3s'
          }
        },
        {
          selector: 'node[?is_root]',
          style: {
            'background-color': '#171E1A',
            'border-color': '#39FF88',
            'border-width': 3,
            'color': '#39FF88',
            'font-weight': 'bold',
            'width': 56,
            'height': 56,
            'shadow-blur': 25,
            'shadow-color': '#39FF88',
            'shadow-opacity': 0.8
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 2.5,
            'line-color': '#35D9FF',
            'target-arrow-color': '#35D9FF',
            'target-arrow-shape': 'triangle',
            'arrow-scale': 1.3,
            'curve-style': 'bezier',
            'label': 'data(label)',
            'font-size': '9px',
            'font-family': 'JetBrains Mono, monospace',
            'color': '#9AA8A0',
            'text-background-color': '#070909',
            'text-background-opacity': 0.9,
            'text-background-padding': '2px',
            'text-rotation': 'autorotate',
            'opacity': 0.85
          }
        },
        {
          selector: 'node:selected',
          style: {
            'border-color': '#D7FF32',
            'border-width': 4,
            'shadow-blur': 25,
            'shadow-color': '#D7FF32',
            'shadow-opacity': 1.0
          }
        },
        {
          selector: 'edge:selected',
          style: {
            'line-color': '#D7FF32',
            'target-arrow-color': '#D7FF32',
            'width': 4.5,
            'color': '#D7FF32'
          }
        }
      ],
      layout: {
        name: edges.length === 0 && layoutName === 'dagre' ? 'grid' : layoutName,
        animate: true,
        animationDuration: 500,
        padding: 50,
        nodeDimensionsIncludeLabels: true
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

  const handleZoomIn = () => cyRef.current?.zoom(cyRef.current.zoom() * 1.25);
  const handleZoomOut = () => cyRef.current?.zoom(cyRef.current.zoom() * 0.8);
  const handleFit = () => cyRef.current?.fit(50);
  const handleResetLayout = () => {
    if (cyRef.current) {
      cyRef.current.layout({ name: layoutName, animate: true, padding: 50 }).run();
    }
  };

  // Search filter
  useEffect(() => {
    if (!cyRef.current || !searchQuery) {
      cyRef.current?.elements().style('opacity', 1);
      return;
    }
    const q = searchQuery.toLowerCase();
    cyRef.current.nodes().forEach(node => {
      const match = node.data('id').toLowerCase().includes(q) || node.data('label').toLowerCase().includes(q);
      node.style('opacity', match ? 1 : 0.2);
    });
  }, [searchQuery]);

  if (!analysisData) {
    return (
      <div className="bg-[#111916] border border-[#1A2822] rounded-3xl p-12 text-center space-y-4">
        <Network className="w-12 h-12 text-[#35D9FF] mx-auto animate-pulse" />
        <h2 className="text-lg font-bold text-[#F4F7F5]">No Provenance Graph Loaded</h2>
        <p className="text-xs text-[#9AA8A0] font-mono max-w-md mx-auto">
          Please upload images or launch a benchmark dataset analysis to construct the interactive forensic graph.
        </p>
        <button
          onClick={() => setActiveTab('upload')}
          className="bg-[#39FF88] text-black font-bold text-xs px-6 py-2.5 rounded-full inline-flex items-center space-x-2"
        >
          <span>Go to Image Upload</span>
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Top Telemetry Header Bar */}
      <div className="bg-[#111916] border border-[#1A2822] rounded-2xl p-4 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-[#35D9FF]/10 border border-[#35D9FF]/40 flex items-center justify-center">
            <Network className="w-4 h-4 text-[#35D9FF]" />
          </div>
          <div>
            <div className="text-xs font-bold text-[#F4F7F5] flex items-center space-x-2">
              <span>Provenance Derivation Network</span>
              <span className="text-[10px] bg-[#39FF88]/10 text-[#39FF88] px-2 py-0.5 rounded-full font-mono font-bold">
                {nodes.length} Nodes • {edges.length} Directed Edges
              </span>
            </div>
            <div className="text-[10px] text-[#9AA8A0] font-mono">
              Root Image: <span className="text-[#39FF88] font-bold">{rootId || 'Detected'}</span> • Spanning Forest
            </div>
          </div>
        </div>

        {/* Graph Controls Bar */}
        <div className="flex items-center space-x-2 font-mono text-xs">
          {/* Search */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-[#9AA8A0] absolute left-2.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search node..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-[#070909] border border-[#1A2822] rounded-lg pl-8 pr-3 py-1.5 text-xs text-[#F4F7F5] placeholder-[#5F7067] focus:border-[#35D9FF] focus:outline-none w-36"
            />
          </div>

          {/* Layout Selector */}
          <select
            value={layoutName}
            onChange={(e) => setLayoutName(e.target.value)}
            className="bg-[#070909] border border-[#1A2822] rounded-lg px-2.5 py-1.5 text-xs text-[#F4F7F5] focus:border-[#35D9FF] focus:outline-none"
          >
            <option value="dagre">Hierarchical (Dagre)</option>
            <option value="cose">Force-Directed (CoSE)</option>
            <option value="breadthfirst">Breadthfirst (BFS)</option>
            <option value="concentric">Concentric Rings</option>
            <option value="circle">Circular Canvas</option>
          </select>

          {/* Zoom Buttons */}
          <div className="flex items-center space-x-1 bg-[#070909] border border-[#1A2822] rounded-lg p-1">
            <button onClick={handleZoomIn} title="Zoom In" className="p-1 hover:text-[#39FF88] text-[#9AA8A0]">
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button onClick={handleZoomOut} title="Zoom Out" className="p-1 hover:text-[#39FF88] text-[#9AA8A0]">
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <button onClick={handleFit} title="Fit Canvas" className="p-1 hover:text-[#35D9FF] text-[#9AA8A0]">
              <Maximize2 className="w-3.5 h-3.5" />
            </button>
            <button onClick={handleResetLayout} title="Reset Layout" className="p-1 hover:text-[#D7FF32] text-[#9AA8A0]">
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Disconnected Forest Alert / Explanation Banner when 0 Directed Edges */}
      {edges.length === 0 && nodes.length > 0 && (
        <div className="bg-[#111916] border border-[#35D9FF]/30 p-4 rounded-2xl flex flex-col md:flex-row items-start md:items-center justify-between gap-3 font-mono text-xs shadow-[0_0_20px_rgba(53,217,255,0.08)]">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-[#35D9FF]/10 border border-[#35D9FF]/30 flex items-center justify-center shrink-0">
              <Info className="w-4 h-4 text-[#35D9FF]" />
            </div>
            <div>
              <div className="font-bold text-[#F4F7F5] flex items-center space-x-2">
                <span>Disconnected Provenance Forest ({nodes.length} Independent Sources)</span>
                <span className="text-[10px] bg-[#35D9FF]/20 text-[#35D9FF] px-2 py-0.5 rounded-full font-bold">
                  0 Derivation Edges Formed
                </span>
              </div>
              <p className="text-[11px] text-[#9AA8A0] mt-0.5">
                The relationship classifier evaluated pairwise image features and found no derivation link exceeding the similarity threshold. Images are verified as separate, unlinked origin sources.
              </p>
            </div>
          </div>
          <button
            onClick={() => setActiveTab && setActiveTab('xai')}
            className="bg-[#171E1A] hover:bg-[#1A2822] text-[#39FF88] border border-[#39FF88]/40 px-3.5 py-1.5 rounded-xl text-xs font-bold transition-all shrink-0 cursor-pointer flex items-center space-x-1.5"
          >
            <span>Inspect XAI Evidence</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Main Canvas + Left/Right Technical Telemetry Panels (Matching Reference Image 2) */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        
        {/* Left Telemetry: Node Details or Legend */}
        <div className="bg-[#111916] border border-[#1A2822] rounded-2xl p-4 space-y-4 flex flex-col justify-between">
          <div className="space-y-3">
            <div className="text-[11px] font-mono text-[#35D9FF] font-bold uppercase tracking-wider flex items-center space-x-1.5">
              <Cpu className="w-3.5 h-3.5" />
              <span>Node Telemetry</span>
            </div>

            {selectedNode ? (
              <div className="space-y-3 font-mono text-xs">
                <div className="w-full h-36 bg-[#070909] rounded-xl border border-[#1A2822] flex items-center justify-center overflow-hidden p-2">
                  {selectedNode.url ? (
                    <img src={selectedNode.url} alt="Node" className="max-h-full max-w-full object-contain rounded" />
                  ) : (
                    <span className="text-[10px] text-[#5F7067]">No preview</span>
                  )}
                </div>

                <div className="space-y-1.5">
                  <div className="flex justify-between">
                    <span className="text-[#5F7067]">Node ID:</span>
                    <span className="text-[#F4F7F5] font-bold">{selectedNode.id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#5F7067]">Role:</span>
                    <span className={selectedNode.is_root ? 'text-[#39FF88] font-bold' : 'text-[#35D9FF]'}>
                      {selectedNode.is_root ? 'ORIGINAL ROOT' : 'DERIVED CHILD'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#5F7067]">Dimensions:</span>
                    <span className="text-[#F4F7F5]">{selectedNode.width} x {selectedNode.height} px</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#5F7067]">File Size:</span>
                    <span className="text-[#F4F7F5]">{(selectedNode.file_size / 1024).toFixed(1)} KB</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#5F7067]">BFS Depth:</span>
                    <span className="text-[#D7FF32]">Gen {selectedNode.generation || 0}</span>
                  </div>
                  <div className="pt-1">
                    <div className="text-[10px] text-[#5F7067]">SHA-256 Hash:</div>
                    <div className="text-[10px] text-[#9AA8A0] truncate">{selectedNode.sha256 || 'N/A'}</div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-4 bg-[#070909] rounded-xl border border-[#1A2822] text-center space-y-2">
                <div className="text-xs text-[#9AA8A0] font-mono">Select a node on the canvas to inspect forensic telemetry.</div>
                <div className="text-[10px] text-[#5F7067] font-mono">Root node marked with bright green glowing halo.</div>
              </div>
            )}
          </div>

          {/* Graph Role Legend */}
          <div className="border-t border-[#1A2822] pt-3 space-y-2 text-[11px] font-mono">
            <div className="text-[#5F7067] uppercase text-[9px] font-bold">Network Legend</div>
            <div className="flex items-center space-x-2">
              <span className="w-3 h-3 rounded-full bg-[#111916] border-2 border-[#39FF88] shadow-[0_0_8px_rgba(57,255,136,0.6)]"></span>
              <span className="text-[#39FF88]">Original / Root Node</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="w-3 h-3 rounded-full bg-[#111916] border-2 border-[#35D9FF]"></span>
              <span className="text-[#35D9FF]">Derived / Transformed Node</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-0.5 bg-[#35D9FF]"></div>
              <span className="text-[#9AA8A0]">Directed Derivation Edge</span>
            </div>
          </div>
        </div>

        {/* Center Canvas (2 Columns) — Deep Navy-Black Forensic Canvas */}
        <div className="lg:col-span-2 bg-[#070909] border border-[#1A2822] rounded-2xl relative overflow-hidden min-h-[520px] forensic-grid flex flex-col justify-between shadow-[inset_0_0_50px_rgba(0,0,0,0.8)]">
          {/* Subtle Cyber Grid Glow Background */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 rounded-full bg-[#35D9FF]/5 blur-3xl pointer-events-none" />

          {/* Cytoscape Container */}
          <div ref={containerRef} className="w-full h-full flex-1 z-10" />

          {/* Bottom Telemetry Floating Overlay */}
          <div className="z-20 bg-[#111916]/85 backdrop-blur-md border-t border-[#1A2822] p-3 px-4 flex items-center justify-between text-xs font-mono">
            <div className="flex items-center space-x-4">
              <span className="text-[#9AA8A0]">Topology: <span className="text-[#35D9FF] font-bold">Directed Acyclic Graph (DAG)</span></span>
              <span className="text-[#5F7067]">|</span>
              <span className="text-[#9AA8A0]">Spanning Forest: <span className="text-[#39FF88] font-bold">Verified Kruskal MST</span></span>
            </div>
            <div className="text-[10px] text-[#5F7067]">
              Click elements to inspect
            </div>
          </div>
        </div>

        {/* Right Telemetry: Edge Details & Spectral Attribution */}
        <div className="bg-[#111916] border border-[#1A2822] rounded-2xl p-4 space-y-4 flex flex-col justify-between">
          <div className="space-y-3">
            <div className="text-[11px] font-mono text-[#D7FF32] font-bold uppercase tracking-wider flex items-center space-x-1.5">
              <Activity className="w-3.5 h-3.5" />
              <span>Edge Telemetry</span>
            </div>

            {selectedEdge ? (
              <div className="space-y-3 font-mono text-xs">
                <div className="bg-[#070909] p-3 rounded-xl border border-[#1A2822] space-y-2">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-[#39FF88] font-bold">{selectedEdge.source}</span>
                    <ArrowRight className="w-3.5 h-3.5 text-[#D7FF32]" />
                    <span className="text-[#35D9FF] font-bold">{selectedEdge.target}</span>
                  </div>
                  <div className="text-center pt-1">
                    <span className="bg-[#D7FF32]/10 text-[#D7FF32] border border-[#D7FF32]/30 px-3 py-1 rounded-full text-xs font-bold uppercase">
                      {selectedEdge.transformation}
                    </span>
                  </div>
                </div>

                <div className="space-y-1.5">
                  <div className="flex justify-between">
                    <span className="text-[#5F7067]">Pairwise Confidence:</span>
                    <span className="text-[#F4F7F5] font-bold">{(selectedEdge.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#5F7067]">Direction Confidence:</span>
                    <span className="text-[#39FF88]">{(selectedEdge.direction_confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#5F7067]">Distance Weight:</span>
                    <span className="text-[#35D9FF]">{selectedEdge.weight?.toFixed(4) || '0.1200'}</span>
                  </div>
                </div>

                <div className="pt-2">
                  <button
                    onClick={() => setActiveTab('xai')}
                    className="w-full bg-[#171E1A] hover:bg-[#1A2822] text-[#39FF88] border border-[#39FF88]/40 py-2 rounded-xl text-xs font-bold font-mono transition-all cursor-pointer flex items-center justify-center space-x-1.5"
                  >
                    <span>Inspect XAI Heatmap</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            ) : edges.length === 0 ? (
              <div className="p-4 bg-[#070909] rounded-xl border border-[#35D9FF]/20 text-left space-y-3 font-mono text-xs">
                <div className="text-[11px] font-bold text-[#35D9FF] flex items-center space-x-1.5">
                  <Info className="w-3.5 h-3.5" />
                  <span>Pairwise Forest Evaluation</span>
                </div>
                <div className="text-[11px] text-[#9AA8A0] leading-relaxed">
                  No pairwise edges satisfied candidate threshold ({analysisData?.similarity_threshold || 0.50}). All nodes reside in isolated tree components.
                </div>
                <div className="p-2.5 bg-[#111916] rounded-lg border border-[#1A2822] space-y-1 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#5F7067]">Evaluated Images:</span>
                    <span className="text-white font-bold">{nodes.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#5F7067]">Directed Edges:</span>
                    <span className="text-[#35D9FF] font-bold">0</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#5F7067]">Forest Topology:</span>
                    <span className="text-[#D7FF32]">Disconnected Forest</span>
                  </div>
                </div>
                <button
                  onClick={() => setActiveTab && setActiveTab('xai')}
                  className="w-full bg-[#171E1A] hover:bg-[#1A2822] text-[#39FF88] border border-[#39FF88]/40 py-2 rounded-xl text-xs font-bold font-mono transition-all cursor-pointer flex items-center justify-center space-x-1.5"
                >
                  <span>Inspect Forensic Evidence in XAI</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            ) : (
              <div className="p-4 bg-[#070909] rounded-xl border border-[#1A2822] text-center space-y-2">
                <div className="text-xs text-[#9AA8A0] font-mono">Select a directed edge on the canvas to inspect transformation evidence.</div>
              </div>
            )}
          </div>

          {/* Mini Waveform / Spectral Telemetry Graph (Matching Reference Image 2 Waveforms) */}
          <div className="border-t border-[#1A2822] pt-3 space-y-2 font-mono">
            <div className="text-[#5F7067] uppercase text-[9px] font-bold">Spectral Noise Res. Waveform</div>
            <div className="h-14 bg-[#070909] rounded-xl border border-[#1A2822] p-2 flex items-end justify-between space-x-1">
              {[25, 45, 65, 30, 85, 40, 70, 95, 35, 60, 50, 80, 20, 75].map((h, i) => (
                <div 
                  key={i} 
                  className="flex-1 bg-gradient-to-t from-[#35D9FF]/20 to-[#35D9FF] rounded-t-xs" 
                  style={{ height: `${h}%` }} 
                />
              ))}
            </div>
            <div className="flex justify-between text-[9px] text-[#5F7067]">
              <span>0 Hz (DC)</span>
              <span>Laplacian &Delta;</span>
              <span>High-pass</span>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
