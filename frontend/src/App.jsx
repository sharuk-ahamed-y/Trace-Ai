import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import UploadZone from './components/UploadZone';
import ProvenanceGraphViewer from './components/ProvenanceGraphViewer';
import XaiPanel from './components/XaiPanel';
import DatasetExplorer from './components/DatasetExplorer';
import HistoryView from './components/HistoryView';
import SystemHealth from './components/SystemHealth';
import { Network, Upload, FileSearch, Database, Sparkles, ShieldCheck, ArrowRight, GitFork, Cpu } from 'lucide-react';
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
    <div className="min-h-screen bg-cyber-dark text-slate-100 flex flex-col">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} health={health} />

      <main className="flex-1 p-6 max-w-7xl w-full mx-auto">
        {activeTab === 'dashboard' && (
          <div className="space-y-8">
            {/* Hero Banner */}
            <div className="bg-gradient-to-r from-cyber-card via-[#0F172A] to-cyber-card p-8 rounded-3xl border border-cyber-border relative overflow-hidden">
              <div className="absolute top-0 right-0 w-96 h-96 bg-cyber-neon/5 rounded-full blur-3xl -z-0"></div>
              <div className="relative z-10 max-w-2xl space-y-4">
                <span className="bg-cyber-neon/10 border border-cyber-neon/40 text-cyber-neon font-mono text-xs px-3 py-1 rounded-full uppercase tracking-widest font-bold">
                  PROVEX AI FORENSIC PLATFORM
                </span>
                <h1 className="text-3xl font-extrabold text-white tracking-tight sm:text-4xl">
                  Image Provenance Analysis & Graph Reconstruction
                </h1>
                <p className="text-cyber-muted text-sm font-mono leading-relaxed">
                  Identify original root images, reconstruct multi-level parent-child manipulation trees, predict 12 edit transformations using ViT-B/16 & Minimum Spanning Trees (MST).
                </p>
                <div className="flex items-center space-x-4 pt-2">
                  <button
                    onClick={() => setActiveTab('analysis')}
                    className="bg-cyber-neon text-black font-semibold px-5 py-3 rounded-xl text-xs flex items-center space-x-2 shadow-[0_0_15px_rgba(0,255,136,0.3)] hover:bg-cyber-neon/90 transition-all"
                  >
                    <Upload className="w-4 h-4" />
                    <span>Launch New Analysis</span>
                  </button>
                  <button
                    onClick={() => setActiveTab('dataset')}
                    className="bg-cyber-dark text-white hover:bg-cyber-border px-5 py-3 rounded-xl text-xs font-mono flex items-center space-x-2 border border-cyber-border transition-all"
                  >
                    <Database className="w-4 h-4 text-cyber-blue" />
                    <span>Explore Benchmark Dataset</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Core Feature Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div
                onClick={() => setActiveTab('analysis')}
                className="bg-cyber-card p-6 rounded-2xl border border-cyber-border hover:border-cyber-neon transition-all cursor-pointer group space-y-3"
              >
                <div className="w-10 h-10 rounded-xl bg-cyber-neon/10 border border-cyber-neon/40 text-cyber-neon flex items-center justify-center group-hover:scale-110 transition-transform">
                  <Upload className="w-5 h-5" />
                </div>
                <h3 className="text-base font-bold text-white group-hover:text-cyber-neon transition-colors">1. Suspect Image Upload</h3>
                <p className="text-xs text-cyber-muted font-mono leading-relaxed">
                  Drag & drop 2+ related images. Computes SHA256/MD5 hashes, EXIF metadata, and 15-dim forensic noise feature vectors.
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
                <h3 className="text-base font-bold text-white group-hover:text-cyber-blue transition-colors">2. MST Graph Reconstruction</h3>
                <p className="text-xs text-cyber-muted font-mono leading-relaxed">
                  Kruskal / Edmonds directed arborescence tree algorithm identifies candidate root node and edge dependencies.
                </p>
                <div className="flex items-center space-x-1 text-xs text-cyber-blue font-mono font-bold pt-2">
                  <span>View Graph Engine</span>
                  <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
                </div>
              </div>

              <div
                onClick={() => setActiveTab('xai')}
                className="bg-cyber-card p-6 rounded-2xl border border-cyber-border hover:border-cyber-purple transition-all cursor-pointer group space-y-3"
              >
                <div className="w-10 h-10 rounded-xl bg-cyber-purple/10 border border-cyber-purple/40 text-cyber-purple flex items-center justify-center group-hover:scale-110 transition-transform">
                  <FileSearch className="w-5 h-5" />
                </div>
                <h3 className="text-base font-bold text-white group-hover:text-cyber-purple transition-colors">3. XAI Heatmaps & Evidence</h3>
                <p className="text-xs text-cyber-muted font-mono leading-relaxed">
                  Renders spatial noise residual diff heatmaps and empirical text explanations detailing JPEG/Laplacian deltas.
                </p>
                <div className="flex items-center space-x-1 text-xs text-cyber-purple font-mono font-bold pt-2">
                  <span>Inspect Evidence</span>
                  <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
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
