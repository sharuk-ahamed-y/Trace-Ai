import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import UploadZone from './components/UploadZone';
import ProvenanceGraphViewer from './components/ProvenanceGraphViewer';
import XaiPanel from './components/XaiPanel';
import DatasetExplorer from './components/DatasetExplorer';
import HistoryView from './components/HistoryView';
import SystemHealth from './components/SystemHealth';
import { Network, Upload, FileSearch, Database, Sparkles, ShieldCheck, ArrowRight, GitFork, Cpu, Layers, Quote, Award } from 'lucide-react';
import axios from 'axios';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [analysisData, setAnalysisData] = useState(null);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    const fetchHealth = () => {
      axios.get('/api/health')
        .then(res => setHealth(res.data))
        .catch(() => setHealth({ status: 'OFFLINE' }));
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleAnalysisComplete = (data) => {
    setAnalysisData(data);
  };

  return (
    <div className="min-h-screen bg-cyber-dark text-slate-100 flex flex-col font-sans">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} health={health} />

      <main className="flex-1 p-6 max-w-7xl w-full mx-auto space-y-8">
        {activeTab === 'dashboard' && (
          <div className="space-y-8">
            {/* Research Hero Banner */}
            <div className="bg-gradient-to-r from-cyber-card via-[#064E3B]/30 to-cyber-card p-8 rounded-3xl border border-cyber-neon/30 relative overflow-hidden shadow-[0_0_35px_rgba(16,185,129,0.1)]">
              <div className="absolute top-0 right-0 w-96 h-96 bg-cyber-neon/5 rounded-full blur-3xl -z-0"></div>
              <div className="relative z-10 max-w-3xl space-y-4">
                <div className="flex items-center space-x-3">
                  <span className="bg-cyber-neon/10 border border-cyber-neon/40 text-cyber-neon font-mono text-xs px-3 py-1 rounded-full uppercase tracking-widest font-bold flex items-center space-x-1.5">
                    <Sparkles className="w-3.5 h-3.5" />
                    <span>TRACEAI RESEARCH PLATFORM</span>
                  </span>
                  <span className="text-xs text-cyber-muted font-mono">IEEE / College Conference Benchmark</span>
                </div>
                
                <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight leading-tight">
                  Transformation-Aware Image Provenance Graph Reconstruction
                </h1>
                <p className="text-cyber-muted text-xs sm:text-sm font-mono leading-relaxed max-w-2xl">
                  Reconstructs multi-generation parent-child image derivation graphs using Minimum Spanning Trees (MST) and Breadth-First Search (BFS) over 72-dimensional visual and spectral forensic representations.
                </p>

                {/* Counter Badges */}
                <div className="grid grid-cols-3 gap-4 py-3 border-y border-cyber-border/60 max-w-lg font-mono">
                  <div>
                    <div className="text-lg font-bold text-cyber-neon">100</div>
                    <div className="text-[10px] text-cyber-muted uppercase">Provenance Graphs</div>
                  </div>
                  <div>
                    <div className="text-lg font-bold text-cyber-blue">795</div>
                    <div className="text-[10px] text-cyber-muted uppercase">Benchmark Images</div>
                  </div>
                  <div>
                    <div className="text-lg font-bold text-purple-400">12</div>
                    <div className="text-[10px] text-cyber-muted uppercase">Edit Classes</div>
                  </div>
                </div>

                <div className="flex items-center space-x-4 pt-2">
                  <button
                    onClick={() => setActiveTab('analysis')}
                    className="bg-cyber-neon text-black font-semibold px-6 py-3 rounded-xl text-xs flex items-center space-x-2 shadow-[0_0_20px_rgba(16,185,129,0.4)] hover:bg-cyber-neon/90 transition-all"
                  >
                    <Upload className="w-4 h-4" />
                    <span>Launch New Provenance Analysis</span>
                  </button>
                  <button
                    onClick={() => setActiveTab('dataset')}
                    className="bg-cyber-dark text-white hover:bg-cyber-border px-5 py-3 rounded-xl text-xs font-mono flex items-center space-x-2 border border-cyber-border transition-all"
                  >
                    <Database className="w-4 h-4 text-cyber-blue" />
                    <span>Explore Research Dataset</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Core Pipeline Stage Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div
                onClick={() => setActiveTab('analysis')}
                className="bg-cyber-card p-6 rounded-2xl border border-cyber-border hover:border-cyber-neon transition-all cursor-pointer group space-y-3"
              >
                <div className="w-10 h-10 rounded-xl bg-cyber-neon/10 border border-cyber-neon/40 text-cyber-neon flex items-center justify-center group-hover:scale-110 transition-transform">
                  <Upload className="w-5 h-5" />
                </div>
                <h3 className="text-base font-bold text-white group-hover:text-cyber-neon transition-colors">1. Multi-Image Upload</h3>
                <p className="text-xs text-cyber-muted font-mono leading-relaxed">
                  Upload multiple suspect images. Computes 72-dimensional visual embeddings and 17-dimensional spectral forensic vectors.
                </p>
                <div className="flex items-center space-x-1 text-xs text-cyber-neon font-mono font-bold pt-2">
                  <span>Start Upload</span>
                  <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
                </div>
              </div>

              <div
                onClick={() => setActiveTab('graph')}
                className="bg-cyber-card p-6 rounded-2xl border border-cyber-border hover:border-cyber-blue transition-all cursor-pointer group space-y-3"
              >
                <div className="w-10 h-10 rounded-xl bg-cyber-blue/10 border border-cyber-blue/40 text-cyber-blue flex items-center justify-center group-hover:scale-110 transition-transform">
                  <GitFork className="w-5 h-5" />
                </div>
                <h3 className="text-base font-bold text-white group-hover:text-cyber-blue transition-colors">2. MST & BFS Graph Engine</h3>
                <p className="text-xs text-cyber-muted font-mono leading-relaxed">
                  Applies Minimum Spanning Tree (MST) over candidate pairs, orients edges, detects root original, and computes BFS generation levels.
                </p>
                <div className="flex items-center space-x-1 text-xs text-cyber-blue font-mono font-bold pt-2">
                  <span>View Graph Reconstruction</span>
                  <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
                </div>
              </div>

              <div
                onClick={() => setActiveTab('xai')}
                className="bg-cyber-card p-6 rounded-2xl border border-cyber-border hover:border-purple-400 transition-all cursor-pointer group space-y-3"
              >
                <div className="w-10 h-10 rounded-xl bg-purple-500/10 border border-purple-400/40 text-purple-400 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <FileSearch className="w-5 h-5" />
                </div>
                <h3 className="text-base font-bold text-white group-hover:text-purple-400 transition-colors">3. XAI Evidence & Explanations</h3>
                <p className="text-xs text-cyber-muted font-mono leading-relaxed">
                  Generates human-readable model explanations detailing root evidence, relationship confidence, and top forensic feature deltas.
                </p>
                <div className="flex items-center space-x-1 text-xs text-purple-400 font-mono font-bold pt-2">
                  <span>Inspect XAI Evidence</span>
                  <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
                </div>
              </div>
            </div>

            {/* Research Quote & Design Card (Inspired by Template UI) */}
            <div className="bg-cyber-card p-6 rounded-2xl border border-cyber-border flex flex-col md:flex-row items-center justify-between gap-6">
              <div className="flex items-start space-x-4">
                <Quote className="w-8 h-8 text-cyber-neon flex-shrink-0 mt-1 opacity-80" />
                <div className="space-y-1">
                  <h4 className="text-sm font-bold text-white uppercase tracking-wider font-mono">Image Provenance Principles</h4>
                  <p className="text-xs text-cyber-muted italic leading-relaxed max-w-3xl">
                    "Determining the lineage of digital assets requires uncovering structural relationships and physical edit footprints across image generations without prior knowledge of root identity."
                  </p>
                  <p className="text-[11px] text-cyber-neon font-mono font-semibold pt-1">— TRACEAI Image Provenance Research Benchmark</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'analysis' && (
          <UploadZone onAnalysisComplete={handleAnalysisComplete} setActiveTab={setActiveTab} />
        )}

        {activeTab === 'graph' && (
          <ProvenanceGraphViewer analysisData={analysisData} setActiveTab={setActiveTab} />
        )}

        {activeTab === 'xai' && (
          <XaiPanel analysisData={analysisData} />
        )}

        {activeTab === 'dataset' && (
          <DatasetExplorer onAnalysisComplete={handleAnalysisComplete} setActiveTab={setActiveTab} />
        )}

        {activeTab === 'history' && (
          <HistoryView onAnalysisComplete={handleAnalysisComplete} setActiveTab={setActiveTab} />
        )}

        {activeTab === 'health' && (
          <SystemHealth health={health} />
        )}
      </main>
    </div>
  );
}
