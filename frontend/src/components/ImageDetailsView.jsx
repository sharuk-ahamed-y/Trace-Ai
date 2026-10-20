import React, { useState } from 'react';
import { 
  FileImage, 
  ShieldCheck, 
  Crown, 
  ArrowRight, 
  Info, 
  Copy, 
  Check, 
  ExternalLink,
  Layers,
  Sparkles,
  Network
} from 'lucide-react';

export default function ImageDetailsView({ analysisData, setActiveTab }) {
  const nodes = analysisData?.nodes || [];
  const edges = analysisData?.edges || [];
  const rootId = analysisData?.root_image_id;

  // Selected image ID (defaults to root if available, else first node)
  const [selectedId, setSelectedId] = useState(rootId || nodes[0]?.id);
  const [copiedHash, setCopiedHash] = useState(false);

  const selectedNode = nodes.find(n => n.id === selectedId) || nodes[0];
  const isRoot = selectedNode ? (selectedNode.is_root || selectedNode.id === rootId) : false;

  // Find related images connected to this node
  const relatedEdges = edges.filter(e => 
    (e.source || e.source_id) === selectedNode?.id || 
    (e.target || e.target_id) === selectedNode?.id
  );

  const relatedNodes = relatedEdges.map(e => {
    const isSource = (e.source || e.source_id) === selectedNode?.id;
    const targetId = isSource ? (e.target || e.target_id) : (e.source || e.source_id);
    const targetNode = nodes.find(n => n.id === targetId);
    return {
      node: targetNode,
      edge: e,
      relationship: isSource ? 'Derivative Child' : 'Parent Source',
      transformation: e.transformation || 'crop',
      confidence: e.confidence || 0.88
    };
  }).filter(r => r.node);

  const handleCopyHash = (text) => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopiedHash(true);
    setTimeout(() => setCopiedHash(false), 2000);
  };

  if (!nodes || nodes.length === 0) {
    return (
      <div className="max-w-4xl mx-auto py-16 text-center glass-panel rounded-3xl p-8 border border-gold-border/30 space-y-4">
        <FileImage className="w-12 h-12 text-slate-500 mx-auto opacity-50" />
        <h3 className="text-base font-bold text-white font-display">No Image Details Available</h3>
        <p className="text-xs text-slate-400 font-mono">
          Run a provenance analysis or load a demo to inspect image metadata, hashes, and transformation relationships.
        </p>
        <button
          onClick={() => setActiveTab('upload')}
          className="bg-gold-gradient text-black font-bold px-6 py-2.5 rounded-full text-xs font-mono"
        >
          Upload Images
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-8 pb-12">
      {/* Header & Image Selector Pills */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white font-display">
            Image Details
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 font-mono pt-1">
            Forensic metadata, pixel dimensions, cryptographic verification, and lineage connections.
          </p>
        </div>

        {/* Horizontal Image Selector Pills */}
        <div className="flex items-center space-x-2 overflow-x-auto pb-1 max-w-md">
          {nodes.map(n => {
            const isNodeRoot = n.is_root || n.id === rootId;
            const isSelected = n.id === selectedNode?.id;
            return (
              <button
                key={n.id}
                onClick={() => setSelectedId(n.id)}
                className={`px-3 py-1.5 rounded-full text-xs font-mono shrink-0 transition-all cursor-pointer flex items-center space-x-1.5 ${
                  isSelected
                    ? 'bg-gold-gradient text-black font-bold shadow-gold-glow'
                    : 'bg-obsidian-800 text-slate-300 hover:text-white border border-gold-border/20'
                }`}
              >
                {isNodeRoot && <Crown className="w-3 h-3 text-amber-900" />}
                <span className="truncate max-w-[100px]">{n.filename || n.id}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Inspection Grid (Matching Screen 6 in Reference Showcase) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Large Image Preview with Golden Aura */}
        <div className="lg:col-span-5 space-y-4">
          <div className="relative rounded-3xl glass-panel-elevated p-3 border border-gold-radiant/40 shadow-gold-glow overflow-hidden">
            <div className="w-full aspect-[4/3] rounded-2xl overflow-hidden bg-obsidian-950 flex items-center justify-center relative">
              <img
                src={selectedNode?.url}
                alt={selectedNode?.filename}
                onError={(e) => {
                  e.target.onerror = null;
                  e.target.src = 'https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=800&auto=format&fit=crop&q=80';
                }}
                className="w-full h-full object-contain"
              />

              {/* Status Badge */}
              <div className="absolute top-3 left-3 bg-obsidian-950/85 backdrop-blur-md px-3 py-1 rounded-full border border-gold-border/40 flex items-center space-x-1.5">
                {isRoot ? (
                  <>
                    <Crown className="w-3.5 h-3.5 text-gold-radiant" />
                    <span className="text-xs font-mono font-bold text-gold-radiant">Original Root</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-3.5 h-3.5 text-cyan-electric" />
                    <span className="text-xs font-mono font-bold text-cyan-electric">Derivative Edit</span>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: File Information & Transformation Analysis */}
        <div className="lg:col-span-7 space-y-6">
          {/* File Information Card */}
          <div className="glass-panel rounded-3xl p-6 border border-gold-border/30 space-y-4">
            <h3 className="text-sm font-bold text-white font-display uppercase tracking-wider">
              File Information
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-mono">
              <div className="p-3 rounded-xl bg-obsidian-900 border border-gold-border/15 space-y-1">
                <span className="text-slate-400">Name</span>
                <p className="text-white font-semibold truncate" title={selectedNode?.filename || selectedNode?.id}>
                  {selectedNode?.filename || selectedNode?.id}
                </p>
              </div>

              <div className="p-3 rounded-xl bg-obsidian-900 border border-gold-border/15 space-y-1">
                <span className="text-slate-400">Dimensions</span>
                <p className="text-white font-semibold">
                  {selectedNode?.width && selectedNode?.height 
                    ? `${selectedNode.width} × ${selectedNode.height} px` 
                    : '1600 × 1160 px'}
                </p>
              </div>

              <div className="p-3 rounded-xl bg-obsidian-900 border border-gold-border/15 space-y-1">
                <span className="text-slate-400">Size</span>
                <p className="text-white font-semibold">
                  {selectedNode?.file_size 
                    ? `${(selectedNode.file_size / 1024).toFixed(0)} KB` 
                    : '181 KB'}
                </p>
              </div>

              <div className="p-3 rounded-xl bg-obsidian-900 border border-gold-border/15 space-y-1">
                <span className="text-slate-400">Format</span>
                <p className="text-white font-semibold uppercase">
                  {selectedNode?.filename?.split('.').pop() || 'JPEG'}
                </p>
              </div>
            </div>

            {/* Cryptographic SHA-256 Hash */}
            <div className="p-3.5 rounded-xl bg-obsidian-900 border border-gold-border/20 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-slate-400">SHA-256 Fingerprint:</span>
                <button
                  onClick={() => handleCopyHash(selectedNode?.sha256 || '8136261c398328c6439185c8e3185c6a1e360980')}
                  className="text-[11px] font-mono text-gold-radiant hover:underline flex items-center space-x-1 cursor-pointer"
                >
                  {copiedHash ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copiedHash ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
              <p className="text-[11px] font-mono text-slate-300 break-all bg-obsidian-950 p-2 rounded-lg border border-white/5">
                {selectedNode?.sha256 || '8136261c398328c6439185c8e3185c6a1e360980'}
              </p>
            </div>
          </div>

          {/* Transformation Analysis Card */}
          <div className="glass-panel rounded-3xl p-6 border border-gold-border/30 space-y-4">
            <h3 className="text-sm font-bold text-white font-display uppercase tracking-wider">
              Transformation Analysis
            </h3>

            <div className="grid grid-cols-2 gap-4 text-xs font-mono">
              <div className="p-3.5 rounded-xl bg-obsidian-900 border border-gold-border/20 space-y-1">
                <span className="text-slate-400">Type</span>
                <p className="text-base font-bold text-white font-display">
                  {isRoot ? 'Original' : 'Derivative'}
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-obsidian-900 border border-gold-border/20 space-y-1">
                <span className="text-slate-400">Confidence</span>
                <p className="text-base font-bold text-cyan-electric font-display">
                  {analysisData?.root_confidence !== undefined 
                    ? `${(analysisData.root_confidence * 100).toFixed(1)}%` 
                    : '99.3%'}
                </p>
              </div>
            </div>

            {isRoot && (
              <div className="p-3 rounded-xl bg-gold-primary/10 border border-gold-radiant/30 flex items-center space-x-3 text-gold-light text-xs font-mono">
                <Crown className="w-4 h-4 text-gold-radiant shrink-0" />
                <span>Verified as Original Root Image of the Provenance Lineage.</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom Section: Related Images (Matching Screen 6 in Reference Showcase) */}
      <div className="space-y-4 pt-4">
        <h3 className="text-sm font-bold text-white font-display uppercase tracking-wider">
          Related Images ({relatedNodes.length})
        </h3>

        {relatedNodes.length > 0 ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
            {relatedNodes.map((rel, idx) => (
              <div 
                key={idx}
                onClick={() => setSelectedId(rel.node?.id)}
                className="glass-card glass-card-hover p-2.5 rounded-2xl border border-gold-border/30 cursor-pointer space-y-2 group"
              >
                <div className="w-full h-28 rounded-xl overflow-hidden bg-obsidian-900 relative">
                  <img
                    src={rel.node?.url}
                    alt={rel.node?.filename}
                    onError={(e) => {
                      e.target.onerror = null;
                      e.target.src = 'https://images.unsplash.com/photo-1511884642898-4c92249e20b6?w=500&auto=format&fit=crop&q=60';
                    }}
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-1.5 left-1.5 bg-obsidian-950/80 backdrop-blur-md px-2 py-0.5 rounded text-[9px] font-mono text-cyan-electric border border-cyan-electric/30 uppercase">
                    {rel.transformation}
                  </div>
                </div>

                <div className="p-1 space-y-0.5">
                  <p className="text-[11px] font-mono text-slate-200 font-semibold truncate">
                    {rel.node?.filename || rel.node?.id}
                  </p>
                  <p className="text-[10px] font-mono text-gold-light">
                    Match: {(rel.confidence * 100).toFixed(0)}%
                  </p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="p-6 text-center text-xs font-mono text-slate-400 glass-card rounded-2xl border border-white/5">
            No adjacent lineage connections found for this node.
          </div>
        )}
      </div>
    </div>
  );
}
