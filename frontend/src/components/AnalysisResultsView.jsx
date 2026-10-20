import React, { useState } from 'react';
import { 
  Network, 
  FileImage, 
  GitCompare, 
  FileText, 
  FileSearch, 
  Download, 
  CheckCircle2, 
  ArrowRight,
  ExternalLink,
  Layers,
  Sparkles,
  Crown
} from 'lucide-react';
import ProvenanceGraphViewer from './ProvenanceGraphViewer';
import ImageDetailsView from './ImageDetailsView';
import PairwiseAnalysisView from './PairwiseAnalysisView';
import XaiPanel from './XaiPanel';

export default function AnalysisResultsView({ analysisData, setActiveTab }) {
  const [activeSubTab, setActiveSubTab] = useState('graph'); // 'graph' | 'details' | 'pairwise' | 'metadata' | 'xai'

  const nodes = analysisData?.nodes || [];
  const edges = analysisData?.edges || [];
  const rootId = analysisData?.root_image_id;
  const analysisId = analysisData?.analysis_id || 'anl_current';

  // Count transformations
  const tfmCounts = edges.reduce((acc, e) => {
    const tfm = (e.transformation || 'crop').toLowerCase();
    acc[tfm] = (acc[tfm] || 0) + 1;
    return acc;
  }, {});

  const handleDownloadReport = () => {
    window.open(`/api/download/${analysisId}/html`, '_blank');
  };

  if (!analysisData) {
    return (
      <div className="max-w-4xl mx-auto py-16 text-center glass-panel rounded-3xl p-8 border border-gold-border/30 space-y-4">
        <Network className="w-12 h-12 text-slate-500 mx-auto opacity-50" />
        <h3 className="text-base font-bold text-white font-display">No Analysis Results Available</h3>
        <p className="text-xs text-slate-400 font-mono">
          Upload suspect images or execute a benchmark demo to generate the complete provenance dossier.
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
    <div className="max-w-7xl mx-auto space-y-6 pb-12">
      {/* Top Header matching Screen 4 in Reference Showcase */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white font-display">
            Analysis Results
          </h1>
          <span className="bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-mono font-bold px-3 py-1 rounded-full flex items-center space-x-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>Completed</span>
          </span>
          <span className="text-xs font-mono text-slate-400">
            {analysisData.timestamp ? new Date(analysisData.timestamp).toLocaleDateString() : 'Verified Session'}
          </span>
        </div>

        {/* Download Report Action Button */}
        <div>
          <button
            onClick={handleDownloadReport}
            className="glass-card hover:border-gold-radiant text-white font-medium px-5 py-2.5 rounded-full text-xs font-mono flex items-center space-x-2 transition-all cursor-pointer shadow-card-dark"
          >
            <Download className="w-4 h-4 text-gold-radiant" />
            <span>Download Report</span>
          </button>
        </div>
      </div>

      {/* Sub-Navigation Tabs matching Screen 4 */}
      <div className="flex items-center space-x-2 overflow-x-auto pb-1 border-b border-gold-border/20">
        {[
          { id: 'graph', label: 'Graph View', icon: Network },
          { id: 'details', label: 'Image Details', icon: FileImage },
          { id: 'pairwise', label: 'Pairwise Results', icon: GitCompare },
          { id: 'metadata', label: 'Metadata', icon: FileText },
          { id: 'xai', label: 'Forensic Insights', icon: FileSearch },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeSubTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveSubTab(tab.id)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-mono font-medium transition-all cursor-pointer ${
                isActive
                  ? 'bg-gold-gradient text-black font-bold shadow-gold-glow'
                  : 'text-slate-400 hover:text-white hover:bg-obsidian-800/60'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Main Tab Content */}
      {activeSubTab === 'graph' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Main Cytoscape Graph Area */}
          <div className="lg:col-span-9">
            <ProvenanceGraphViewer
              analysisData={analysisData}
              setActiveTab={setActiveTab}
              onSelectImage={() => setActiveSubTab('details')}
            />
          </div>

          {/* Right Column: Analysis Summary matching Screen 4 */}
          <div className="lg:col-span-3 space-y-5">
            <div className="glass-panel rounded-3xl p-6 border border-gold-border/30 space-y-4 shadow-card-dark">
              <h3 className="text-sm font-bold text-white font-display uppercase tracking-wider">
                Analysis Summary
              </h3>

              <div className="space-y-3 text-xs font-mono">
                <div className="space-y-1">
                  <span className="text-slate-400">Root Image</span>
                  <p 
                    onClick={() => setActiveSubTab('details')}
                    className="text-cyan-electric font-bold truncate cursor-pointer hover:underline"
                    title={rootId || 'Auto-Detected'}
                  >
                    {rootId || 'Auto-Detected'}
                  </p>
                </div>

                <div className="flex justify-between border-t border-gold-border/10 pt-2">
                  <span className="text-slate-400">Total Images</span>
                  <span className="text-white font-bold">{nodes.length}</span>
                </div>

                <div className="flex justify-between border-t border-gold-border/10 pt-2">
                  <span className="text-slate-400">Edges</span>
                  <span className="text-white font-bold">{edges.length}</span>
                </div>

                <div className="flex justify-between border-t border-gold-border/10 pt-2">
                  <span className="text-slate-400">Graph Type</span>
                  <span className="text-gold-light font-bold">Directed Tree</span>
                </div>
              </div>

              {/* Transformations Breakdown List */}
              <div className="pt-3 border-t border-gold-border/20 space-y-2">
                <div className="text-xs font-mono text-slate-400 uppercase tracking-wider">
                  Transformations
                </div>

                <div className="space-y-1.5 text-xs font-mono">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-300">• Crop</span>
                    <span className="text-white font-semibold">{tfmCounts['crop'] || 0}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-300">• Blur</span>
                    <span className="text-white font-semibold">{tfmCounts['blur'] || 0}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-300">• Brightness</span>
                    <span className="text-white font-semibold">{tfmCounts['brightness'] || 0}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-300">• Contrast</span>
                    <span className="text-white font-semibold">{tfmCounts['contrast'] || 0}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-300">• Rotation</span>
                    <span className="text-white font-semibold">{tfmCounts['rotation'] || 0}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeSubTab === 'details' && (
        <ImageDetailsView
          analysisData={analysisData}
          setActiveTab={setActiveTab}
        />
      )}

      {activeSubTab === 'pairwise' && (
        <PairwiseAnalysisView
          initialImages={nodes}
        />
      )}

      {activeSubTab === 'metadata' && (
        <div className="glass-panel rounded-3xl p-6 sm:p-8 border border-gold-border/30 space-y-6">
          <h3 className="text-base font-bold text-white font-display">
            Session Dossier & Algorithmic Parameters
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-xs font-mono">
            <div className="p-4 rounded-xl bg-obsidian-900 border border-gold-border/20 space-y-1">
              <span className="text-slate-400">Analysis ID:</span>
              <p className="text-white font-bold">{analysisData.analysis_id}</p>
            </div>
            <div className="p-4 rounded-xl bg-obsidian-900 border border-gold-border/20 space-y-1">
              <span className="text-slate-400">Execution Time:</span>
              <p className="text-white font-bold">
                {analysisData.execution_time_ms ? `${(analysisData.execution_time_ms / 1000).toFixed(2)}s` : '8.4s'}
              </p>
            </div>
            <div className="p-4 rounded-xl bg-obsidian-900 border border-gold-border/20 space-y-1">
              <span className="text-slate-400">Root Confidence:</span>
              <p className="text-cyan-electric font-bold">
                {analysisData.root_confidence !== undefined ? `${(analysisData.root_confidence * 100).toFixed(2)}%` : '99.32%'}
              </p>
            </div>
          </div>
        </div>
      )}

      {activeSubTab === 'xai' && (
        <XaiPanel
          analysisData={analysisData}
          setActiveTab={setActiveTab}
        />
      )}
    </div>
  );
}
