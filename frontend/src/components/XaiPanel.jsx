import React, { useState } from 'react';
import { FileSearch, Sparkles, AlertCircle, Eye, SlidersHorizontal, CheckCircle2 } from 'lucide-react';

export default function XaiPanel({ analysisData }) {
  const xaiResults = analysisData?.xai_results || [];
  const [selectedIndex, setSelectedIndex] = useState(0);

  if (!analysisData || xaiResults.length === 0) {
    return (
      <div className="max-w-4xl mx-auto py-20 text-center space-y-4">
        <FileSearch className="w-12 h-12 text-cyber-muted mx-auto" />
        <h3 className="text-lg font-semibold text-white">No XAI Forensic Evidence Available</h3>
        <p className="text-xs text-cyber-muted font-mono">Run a graph analysis first to generate empirical noise residual heatmaps and decision explanations.</p>
      </div>
    );
  }

  const currentXai = xaiResults[selectedIndex] || xaiResults[0];
  const evidence = currentXai.evidence || {};
  const findings = evidence.empirical_findings || [];
  const spatialMetrics = currentXai.spatial_metrics || {};

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="bg-cyber-card p-6 rounded-2xl border border-cyber-border flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <FileSearch className="w-5 h-5 text-cyber-neon" />
            <span>Explainable AI (XAI) Forensic Evidence</span>
          </h2>
          <p className="text-xs text-cyber-muted font-mono mt-1">
            Empirical noise residual spatial heatmaps & decision explanations derived from ViT-B/16 embeddings and Laplacian noise statistics.
          </p>
        </div>

        {/* Edge Selector Dropdown */}
        <div className="flex items-center space-x-3">
          <span className="text-xs font-mono text-cyber-muted">Target Edge:</span>
          <select
            value={selectedIndex}
            onChange={(e) => setSelectedIndex(parseInt(e.target.value))}
            className="bg-cyber-dark text-xs font-mono text-cyber-neon border border-cyber-border rounded-xl px-3 py-2 focus:outline-none"
          >
            {xaiResults.map((x, idx) => (
              <option key={idx} value={idx}>
                {x.source_id} &rarr; {x.target_id}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column: Spatial Noise Residual Heatmap */}
        <div className="bg-cyber-card p-6 rounded-2xl border border-cyber-border space-y-4">
          <div className="flex items-center justify-between border-b border-cyber-border pb-3">
            <span className="text-xs font-mono font-bold text-white uppercase tracking-wider flex items-center space-x-2">
              <Eye className="w-4 h-4 text-cyber-blue" />
              <span>Spatial Noise Residual Heatmap</span>
            </span>
            <span className="text-[10px] font-mono bg-cyber-blue/20 text-cyber-blue px-2 py-0.5 rounded border border-cyber-blue/40">
              JET COLORMAP OVERLAY
            </span>
          </div>

          <div className="bg-[#030508] p-3 rounded-xl border border-cyber-border flex items-center justify-center">
            {currentXai.heatmap_url || currentXai.heatmap_b64 ? (
              <img
                src={currentXai.heatmap_b64 || currentXai.heatmap_url}
                alt="XAI Noise Residual Heatmap"
                className="max-h-[380px] rounded-lg object-contain shadow-[0_0_15px_rgba(0,229,255,0.2)]"
              />
            ) : (
              <div className="py-20 text-xs text-cyber-muted font-mono">Heatmap rendering...</div>
            )}
          </div>

          {/* Spatial Metrics Bar */}
          <div className="grid grid-cols-3 gap-3 font-mono text-xs">
            <div className="bg-cyber-dark p-3 rounded-xl border border-cyber-border">
              <span className="text-[10px] text-cyber-muted block">Mean Squared Error (MSE)</span>
              <span className="font-bold text-white text-sm">{spatialMetrics.mse || 'N/A'}</span>
            </div>
            <div className="bg-cyber-dark p-3 rounded-xl border border-cyber-border">
              <span className="text-[10px] text-cyber-muted block">PSNR (dB)</span>
              <span className="font-bold text-cyber-blue text-sm">{spatialMetrics.psnr_db || 'N/A'} dB</span>
            </div>
            <div className="bg-cyber-dark p-3 rounded-xl border border-cyber-border">
              <span className="text-[10px] text-cyber-muted block">SSIM Estimate</span>
              <span className="font-bold text-cyber-neon text-sm">{spatialMetrics.ssim_est || 'N/A'}</span>
            </div>
          </div>
        </div>

        {/* Right Column: Empirical Findings & Reasoning */}
        <div className="bg-cyber-card p-6 rounded-2xl border border-cyber-border space-y-5 font-mono text-xs">
          <div className="border-b border-cyber-border pb-3">
            <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center space-x-2">
              <Sparkles className="w-4 h-4 text-cyber-neon" />
              <span>Empirical Decision Rationale</span>
            </span>
            <p className="text-[11px] text-cyber-muted mt-1">
              Statistical forensic evidence establishing <code className="text-cyber-neon">{currentXai.source_id}</code> as parent of <code className="text-cyber-blue">{currentXai.target_id}</code>.
            </p>
          </div>

          {/* Summary Box */}
          <div className="bg-cyber-dark p-4 rounded-xl border border-cyber-border space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-cyber-muted">Model Confidence:</span>
              <span className="text-cyber-neon font-bold text-sm">
                {((evidence.confidence || 0.90) * 100).toFixed(1)}%
              </span>
            </div>
            <p className="text-slate-200 font-sans leading-relaxed">
              {evidence.summary || 'Provenential dependency detected based on ViT feature similarity and noise statistics.'}
            </p>
          </div>

          {/* Evidence Bullet Points */}
          <div className="space-y-3">
            <h4 className="font-semibold text-white uppercase text-[11px] tracking-wider text-cyber-muted">
              Key Forensic Evidence Markers ({findings.length})
            </h4>
            {findings.length === 0 ? (
              <p className="text-cyber-muted">No specific delta markers computed.</p>
            ) : (
              <div className="space-y-2">
                {findings.map((item, idx) => (
                  <div key={idx} className="flex items-start space-x-2.5 bg-cyber-dark/60 p-3 rounded-xl border border-cyber-border">
                    <CheckCircle2 className="w-4 h-4 text-cyber-neon flex-shrink-0 mt-0.5" />
                    <span className="text-slate-200 font-sans leading-snug">{item}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
