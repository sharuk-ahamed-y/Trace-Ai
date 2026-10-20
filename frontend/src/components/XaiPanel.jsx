import React, { useState } from 'react';
import { 
  BrainCircuit, 
  Sparkles, 
  AlertCircle, 
  Eye, 
  CheckCircle2, 
  Activity, 
  Layers, 
  ArrowRight, 
  ShieldCheck, 
  Info,
  FileText,
  Sliders,
  Maximize2
} from 'lucide-react';

export default function XaiPanel({ analysisData, setActiveTab }) {
  const xaiResults = analysisData?.xai_results || [];
  const nodes = analysisData?.nodes || [];
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [showFullText, setShowFullText] = useState(false);

  const currentXai = xaiResults[selectedIndex] || xaiResults[0];

  // Map nodes by id for image previews
  const nodeMap = nodes.reduce((acc, n) => {
    acc[n.id] = n;
    return acc;
  }, {});

  const srcNode = currentXai ? nodeMap[currentXai.source_id] : null;
  const tgtNode = currentXai ? nodeMap[currentXai.target_id] : null;

  if (!analysisData) {
    return (
      <div className="max-w-4xl mx-auto py-20 text-center glass-panel rounded-3xl p-10 border border-gold-border/30 space-y-6 shadow-card-dark">
        <div className="w-16 h-16 rounded-2xl bg-gold-primary/10 border border-gold-primary/30 flex items-center justify-center text-gold-radiant mx-auto shadow-gold-glow">
          <BrainCircuit className="w-8 h-8 stroke-[1.8]" />
        </div>
        <div className="space-y-2">
          <h2 className="text-xl sm:text-2xl font-bold text-white font-display">
            No Active Forensic Dossier Loaded
          </h2>
          <p className="text-xs sm:text-sm text-slate-400 font-mono max-w-md mx-auto leading-relaxed">
            Explainable AI generates empirical noise residual difference heatmaps, spatial difference metrics, and grounded transformation reasoning when an image analysis is run.
          </p>
        </div>
        <div className="flex items-center justify-center space-x-3 pt-2">
          <button
            onClick={() => setActiveTab('upload')}
            className="bg-gold-gradient hover:opacity-95 text-black font-bold px-6 py-3 rounded-full text-xs font-mono shadow-gold-glow transition-all cursor-pointer"
          >
            Upload Images for Analysis
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-8 pb-16">
      {/* Top Header & Edge Selector */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-3">
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white font-display">
              Explainable AI (XAI)
            </h1>
            <span className="bg-gold-primary/15 border border-gold-primary/30 text-gold-light text-[11px] font-mono px-3 py-1 rounded-full flex items-center space-x-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-electric animate-pulse" />
              <span>Empirical Grounding</span>
            </span>
          </div>
          <p className="text-xs sm:text-sm text-slate-400 font-mono pt-1">
            Inspection of noise residual differentials, spatial metrics, and verifiable derivation rationale.
          </p>
        </div>

        {/* Edge Selector Dropdown if multiple edges exist */}
        {xaiResults.length > 0 && (
          <div className="flex items-center space-x-2.5 glass-card px-3.5 py-1.5 rounded-xl border border-gold-border/30 self-start md:self-auto">
            <span className="text-xs font-mono text-slate-400">Inspecting Edge:</span>
            <select
              value={selectedIndex}
              onChange={(e) => setSelectedIndex(Number(e.target.value))}
              className="bg-obsidian-900 border border-gold-border/40 text-white rounded-lg px-2.5 py-1 text-xs font-mono focus:outline-none cursor-pointer"
            >
              {xaiResults.map((xr, idx) => (
                <option key={idx} value={idx}>
                  #{idx + 1}: {xr.source_id} → {xr.target_id} ({xr.transformation?.toUpperCase()} {(xr.confidence * 100).toFixed(0)}%)
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {currentXai ? (
        <div className="space-y-8">
          {/* Edge Summary Bar */}
          <div className="glass-panel p-6 rounded-3xl border border-gold-border/30 flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-card-dark">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 rounded-2xl bg-cyan-electric/10 border border-cyan-electric/40 flex items-center justify-center text-cyan-electric shrink-0 shadow-cyan-glow">
                <BrainCircuit className="w-6 h-6 stroke-[1.8]" />
              </div>
              <div className="space-y-0.5">
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">
                    Edge Provenance Link:
                  </span>
                  <span className="font-mono text-sm font-bold text-white">
                    {currentXai.source_id}
                  </span>
                  <ArrowRight className="w-3.5 h-3.5 text-gold-radiant" />
                  <span className="font-mono text-sm font-bold text-white">
                    {currentXai.target_id}
                  </span>
                </div>
                <div className="text-xs font-mono text-slate-300">
                  {currentXai.evidence?.summary || `Derived relationship via ${currentXai.transformation?.toUpperCase()}`}
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-4 self-start md:self-auto">
              <div className="text-right">
                <div className="text-[10px] font-mono text-slate-400 uppercase">Transformation</div>
                <div className="text-sm font-mono font-bold text-gold-light uppercase">
                  {currentXai.transformation || 'Unresolved'}
                </div>
              </div>
              <div className="text-right border-l border-gold-border/20 pl-4">
                <div className="text-[10px] font-mono text-slate-400 uppercase">Confidence</div>
                <div className="text-sm font-mono font-bold text-cyan-electric">
                  {((currentXai.confidence || 0) * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>

          {/* Triad Visual Comparison: Source Image vs Noise Residual Heatmap vs Target Image */}
          <div className="glass-panel p-6 sm:p-8 rounded-3xl border border-gold-border/30 space-y-5 shadow-card-dark">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-white font-display uppercase tracking-wider flex items-center space-x-2">
                  <Eye className="w-4 h-4 text-cyan-electric" />
                  <span>Noise Residual Difference & Spatial Heatmap</span>
                </h3>
                <p className="text-xs font-mono text-slate-400 pt-0.5">
                  High-pass 3x3 median filter residual differential overlaid on parent canvas.
                </p>
              </div>
              <span className="text-[11px] font-mono text-cyan-electric bg-cyan-electric/10 border border-cyan-electric/30 px-3 py-1 rounded-full hidden sm:inline-block">
                OpenCV Colormap Jet
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
              {/* Panel 1: Source / Parent Image */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-slate-300 font-semibold">Parent ({currentXai.source_id})</span>
                  <span className="text-slate-500">{srcNode?.width ? `${srcNode.width}x${srcNode.height}` : 'Source Canvas'}</span>
                </div>
                <div className="w-full h-56 rounded-2xl overflow-hidden bg-obsidian-900 border border-gold-border/20 flex items-center justify-center relative group">
                  {srcNode?.url ? (
                    <img 
                      src={srcNode.url} 
                      alt={currentXai.source_id}
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="text-xs font-mono text-slate-500 p-4 text-center">
                      Source Image Preview ({currentXai.source_id})
                    </div>
                  )}
                  <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/70 text-[10px] font-mono text-slate-300">
                    Source
                  </div>
                </div>
              </div>

              {/* Panel 2: Noise Residual Heatmap (Empirical) */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-gold-light font-semibold">Empirical Residual Difference</span>
                  <span className="text-cyan-electric">Jet Heatmap</span>
                </div>
                <div className="w-full h-56 rounded-2xl overflow-hidden bg-obsidian-900 border border-gold-border/40 flex items-center justify-center relative shadow-[0_0_20px_rgba(0,242,254,0.15)]">
                  {currentXai.heatmap_b64 || currentXai.heatmap_url ? (
                    <img 
                      src={currentXai.heatmap_b64 || currentXai.heatmap_url} 
                      alt="Empirical Noise Residual Heatmap"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="text-xs font-mono text-slate-400 p-4 text-center space-y-1">
                      <AlertCircle className="w-5 h-5 text-amber-400 mx-auto" />
                      <div>Heatmap rendering unavailable for this pair</div>
                    </div>
                  )}
                  <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/80 text-[10px] font-mono text-gold-light border border-gold-border/30">
                    Residual Differential
                  </div>
                </div>
              </div>

              {/* Panel 3: Target / Child Image */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-slate-300 font-semibold">Derived Child ({currentXai.target_id})</span>
                  <span className="text-slate-500">{tgtNode?.width ? `${tgtNode.width}x${tgtNode.height}` : 'Child Canvas'}</span>
                </div>
                <div className="w-full h-56 rounded-2xl overflow-hidden bg-obsidian-900 border border-gold-border/20 flex items-center justify-center relative group">
                  {tgtNode?.url ? (
                    <img 
                      src={tgtNode.url} 
                      alt={currentXai.target_id}
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="text-xs font-mono text-slate-500 p-4 text-center">
                      Target Image Preview ({currentXai.target_id})
                    </div>
                  )}
                  <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/70 text-[10px] font-mono text-slate-300">
                    Derived Target
                  </div>
                </div>
              </div>
            </div>

            {/* Heatmap Legend */}
            <div className="p-3 rounded-xl bg-obsidian-900/80 border border-gold-border/15 flex flex-wrap items-center justify-between gap-3 text-[11px] font-mono text-slate-400">
              <div className="flex items-center space-x-2">
                <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
                <span>Red/Yellow: High spatial residual divergence (altered edge, localized framing or color shift)</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
                <span>Blue/Navy: Consistent underlying sensor noise distribution</span>
              </div>
            </div>
          </div>

          {/* Real Spatial Difference Metrics */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="glass-panel p-5 rounded-2xl border border-gold-border/30 space-y-1 shadow-card-dark">
              <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
                Mean Squared Error (MSE)
              </div>
              <div className="text-2xl font-bold font-mono text-white">
                {currentXai.spatial_metrics?.mse !== undefined ? currentXai.spatial_metrics.mse : 'N/A'}
              </div>
              <div className="text-[10px] font-mono text-slate-500">
                Pixel-level residual variance
              </div>
            </div>

            <div className="glass-panel p-5 rounded-2xl border border-gold-border/30 space-y-1 shadow-card-dark">
              <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
                Peak Signal-to-Noise (PSNR)
              </div>
              <div className="text-2xl font-bold font-mono text-gold-light">
                {currentXai.spatial_metrics?.psnr_db !== undefined ? `${currentXai.spatial_metrics.psnr_db} dB` : 'N/A'}
              </div>
              <div className="text-[10px] font-mono text-slate-500">
                Signal fidelity metric
              </div>
            </div>

            <div className="glass-panel p-5 rounded-2xl border border-gold-border/30 space-y-1 shadow-card-dark">
              <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
                SSIM Approximation
              </div>
              <div className="text-2xl font-bold font-mono text-cyan-electric">
                {currentXai.spatial_metrics?.ssim_est !== undefined ? currentXai.spatial_metrics.ssim_est : 'N/A'}
              </div>
              <div className="text-[10px] font-mono text-slate-500">
                Structural similarity coefficient
              </div>
            </div>

            <div className="glass-panel p-5 rounded-2xl border border-gold-border/30 space-y-1 shadow-card-dark">
              <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
                Mean Residual Difference
              </div>
              <div className="text-2xl font-bold font-mono text-slate-200">
                {currentXai.spatial_metrics?.residual_diff_mean !== undefined ? currentXai.spatial_metrics.residual_diff_mean : 'N/A'}
              </div>
              <div className="text-[10px] font-mono text-slate-500">
                High-pass intensity deviation
              </div>
            </div>
          </div>

          {/* Grounded Empirical Findings */}
          <div className="glass-panel p-6 sm:p-8 rounded-3xl border border-gold-border/30 space-y-5 shadow-card-dark">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white font-display uppercase tracking-wider flex items-center space-x-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span>Grounded Forensic Evidence</span>
              </h3>
              {currentXai.evidence?.full_text && (
                <button
                  onClick={() => setShowFullText(!showFullText)}
                  className="text-xs font-mono text-gold-light hover:text-white underline cursor-pointer"
                >
                  {showFullText ? 'Hide Technical Dossier' : 'View Complete Technical Text'}
                </button>
              )}
            </div>

            {/* Structured Empirical Findings List */}
            {currentXai.evidence?.empirical_findings && currentXai.evidence.empirical_findings.length > 0 ? (
              <div className="space-y-2.5">
                {currentXai.evidence.empirical_findings.map((finding, idx) => (
                  <div 
                    key={idx}
                    className="p-3.5 rounded-xl bg-obsidian-900 border border-gold-border/15 flex items-start space-x-3 text-xs font-mono text-slate-300"
                  >
                    <span className="w-5 h-5 rounded-md bg-gold-primary/10 border border-gold-primary/30 text-gold-radiant flex items-center justify-center shrink-0 text-[10px] font-bold">
                      {idx + 1}
                    </span>
                    <span className="leading-relaxed">{finding}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-4 rounded-xl bg-obsidian-900 text-xs font-mono text-slate-400">
                {currentXai.evidence?.summary || 'Empirical findings not generated for this edge.'}
              </div>
            )}

            {/* Expandable Technical Dossier */}
            {showFullText && currentXai.evidence?.full_text && (
              <div className="mt-4 p-4 rounded-2xl bg-black/80 border border-gold-border/20 font-mono text-xs text-slate-300 whitespace-pre-wrap overflow-x-auto leading-relaxed">
                {currentXai.evidence.full_text}
              </div>
            )}
          </div>

          {/* Model Scope & Explainability Disclosure (Honest Disclosures) */}
          <div className="glass-panel p-6 sm:p-8 rounded-3xl border border-gold-border/30 space-y-4 shadow-card-dark">
            <h3 className="text-sm font-bold text-white font-display uppercase tracking-wider flex items-center space-x-2">
              <Info className="w-4 h-4 text-gold-radiant" />
              <span>Explainability Scope & Capabilities Disclosure</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
              <div className="p-4 rounded-2xl bg-obsidian-900/60 border border-emerald-500/20 space-y-2">
                <div className="text-emerald-400 font-semibold flex items-center space-x-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>Implemented & Supported Capabilities</span>
                </div>
                <ul className="space-y-1.5 text-slate-400 text-[11px] list-disc list-inside">
                  <li>High-Pass Noise Residual Difference Heatmaps (OpenCV Median Filter + Jet colormap)</li>
                  <li>Empirical Spatial Difference Metrics (MSE, PSNR, SSIM approximation)</li>
                  <li>ViT-B/16 CLS Semantic Cosine Embedding Alignment</li>
                  <li>Grounded Geometric & Photometric Rule Disclosures (Crop, Blur, Resize, Luminance)</li>
                </ul>
              </div>

              <div className="p-4 rounded-2xl bg-obsidian-900/60 border border-gold-border/20 space-y-2">
                <div className="text-gold-light font-semibold flex items-center space-x-1.5">
                  <Info className="w-3.5 h-3.5" />
                  <span>Current Limitations & Non-Fabricated Metrics</span>
                </div>
                <ul className="space-y-1.5 text-slate-400 text-[11px] list-disc list-inside">
                  <li>SHAP values and Grad-CAM activation heatmaps are not enabled in this baseline.</li>
                  <li>Phase 4 Deep Learning models remain disabled (ENABLE_PHASE4_MODELS=false).</li>
                  <li>Values displayed reflect actual pixel mathematics and computed forensic statistics only.</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="p-8 text-center glass-panel rounded-3xl border border-gold-border/30 space-y-2">
          <AlertCircle className="w-8 h-8 text-amber-400 mx-auto" />
          <h3 className="text-sm font-bold text-white font-display">No XAI Records Found</h3>
          <p className="text-xs font-mono text-slate-400">
            Analysis completed with no directed or evaluated edges.
          </p>
        </div>
      )}
    </div>
  );
}
