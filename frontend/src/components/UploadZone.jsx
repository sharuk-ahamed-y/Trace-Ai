import React, { useState, useRef } from 'react';
import { Upload, FileImage, Play, Sparkles, CheckCircle2, AlertCircle, Sliders, Hash } from 'lucide-react';
import axios from 'axios';

export default function UploadZone({ onAnalysisComplete, setActiveTab }) {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [progressStep, setProgressStep] = useState('');
  const [simThreshold, setSimThreshold] = useState(0.65);
  const [confFilter, setConfFilter] = useState(0.50);
  const [errorMsg, setErrorMsg] = useState(null);
  
  const fileInputRef = useRef(null);

  const handleFileSelect = (e) => {
    const selected = Array.from(e.target.files);
    setFiles(prev => [...prev, ...selected]);
    setErrorMsg(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const dropped = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
    setFiles(prev => [...prev, ...dropped]);
    setErrorMsg(null);
  };

  const handleRemoveFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleRunAnalysis = async () => {
    if (files.length < 2) {
      setErrorMsg("Please select at least 2 images for provenance analysis.");
      return;
    }

    try {
      setUploading(true);
      setErrorMsg(null);
      setProgressStep("Uploading images to secure forensic workspace...");

      const formData = new FormData();
      files.forEach(f => formData.append('files', f));

      const uploadRes = await axios.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const { analysis_id } = uploadRes.data;
      setUploading(false);
      setAnalyzing(true);

      setProgressStep("Extracting ViT-B/16 CLS tokens & forensic residual features...");
      await new Promise(r => setTimeout(r, 600));

      setProgressStep("Computing pairwise directional probabilities & transformation classifications...");
      await new Promise(r => setTimeout(r, 600));

      setProgressStep("Running Edmonds/Kruskal Minimum Spanning Tree (MST) graph reconstruction...");

      const analyzeRes = await axios.post('/api/analyze', {
        analysis_id,
        similarity_threshold: simThreshold,
        confidence_filter: confFilter
      });

      setProgressStep("Generating empirical XAI heatmaps & forensic evidence report...");
      await new Promise(r => setTimeout(r, 400));

      setAnalyzing(false);
      onAnalysisComplete(analyzeRes.data);
      setActiveTab('graph');

    } catch (err) {
      console.error(err);
      setUploading(false);
      setAnalyzing(false);
      setErrorMsg(err.response?.data?.detail || "Error during provenance analysis execution.");
    }
  };

  const handleRunDemo = async () => {
    try {
      setAnalyzing(true);
      setErrorMsg(null);
      setProgressStep("Staging benchmark DAG (graph_001)...");
      await new Promise(r => setTimeout(r, 500));

      setProgressStep("Running ViT-B/16 ML Pairwise & MST Provenance Inference...");

      const demoRes = await axios.post('/api/demo?graph_id=graph_0001');

      setAnalyzing(false);
      onAnalysisComplete(demoRes.data);
      setActiveTab('graph');
    } catch (err) {
      console.error(err);
      setAnalyzing(false);
      setErrorMsg(err.response?.data?.detail || "Failed to launch benchmark demo.");
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Title Card */}
      <div className="bg-cyber-card p-6 rounded-2xl border border-cyber-border flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <span>Image Provenance Analysis & Graph Reconstruction</span>
          </h2>
          <p className="text-xs text-cyber-muted mt-1 font-mono">
            Upload multiple suspect images to reconstruct parent-child edit DAGs, identify root original, and predict transformations.
          </p>
        </div>
        <button
          onClick={handleRunDemo}
          disabled={uploading || analyzing}
          className="flex items-center space-x-2 bg-cyber-blue/20 text-cyber-blue hover:bg-cyber-blue hover:text-black font-semibold px-4 py-2.5 rounded-xl border border-cyber-blue/40 transition-all text-xs shadow-[0_0_12px_rgba(0,229,255,0.2)] disabled:opacity-50"
        >
          <Sparkles className="w-4 h-4" />
          <span>Load Instant Benchmark Demo</span>
        </button>
      </div>

      {errorMsg && (
        <div className="bg-red-500/10 border border-red-500/40 p-4 rounded-xl text-red-400 text-xs flex items-center space-x-3">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Main Upload Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Drop Area & Settings */}
        <div className="lg:col-span-1 space-y-6">
          {/* Dropzone */}
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-cyber-border hover:border-cyber-neon bg-cyber-dark/60 p-8 rounded-2xl text-center cursor-pointer transition-all hover:bg-cyber-neon/5 group"
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              multiple
              accept="image/*"
              className="hidden"
            />
            <div className="w-12 h-12 rounded-xl bg-cyber-neon/10 border border-cyber-neon/40 text-cyber-neon mx-auto flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
              <Upload className="w-6 h-6" />
            </div>
            <p className="text-sm font-semibold text-white">Drag & drop image set here</p>
            <p className="text-xs text-cyber-muted mt-1 font-mono">Supports PNG, JPG, JPEG, WEBP</p>
            <button className="mt-4 text-xs font-mono text-cyber-neon border border-cyber-neon/40 bg-cyber-neon/10 px-3 py-1.5 rounded-lg group-hover:bg-cyber-neon group-hover:text-black transition-colors">
              Browse Files
            </button>
          </div>

          {/* Hyperparameters Card */}
          <div className="bg-cyber-card p-5 rounded-2xl border border-cyber-border space-y-4">
            <div className="flex items-center space-x-2 text-xs font-semibold text-white border-b border-cyber-border pb-2">
              <Sliders className="w-4 h-4 text-cyber-neon" />
              <span>Inference Parameters</span>
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-cyber-muted">Similarity Threshold:</span>
                <span className="text-cyber-neon font-bold">{simThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.40"
                max="0.95"
                step="0.05"
                value={simThreshold}
                onChange={(e) => setSimThreshold(parseFloat(e.target.value))}
                className="w-full accent-cyber-neon cursor-pointer bg-cyber-dark rounded-lg h-2"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-cyber-muted">MST Edge Confidence Filter:</span>
                <span className="text-cyber-neon font-bold">{confFilter.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.30"
                max="0.90"
                step="0.05"
                value={confFilter}
                onChange={(e) => setConfFilter(parseFloat(e.target.value))}
                className="w-full accent-cyber-neon cursor-pointer bg-cyber-dark rounded-lg h-2"
              />
            </div>
          </div>
        </div>

        {/* Right Column: File List & Action Button */}
        <div className="lg:col-span-2 bg-cyber-card p-6 rounded-2xl border border-cyber-border flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b border-cyber-border pb-3 mb-4">
              <div className="flex items-center space-x-2">
                <FileImage className="w-4 h-4 text-cyber-neon" />
                <h3 className="text-sm font-semibold text-white">Staged Suspect Images ({files.length})</h3>
              </div>
              {files.length > 0 && (
                <button
                  onClick={() => setFiles([])}
                  className="text-xs text-red-400 hover:underline font-mono"
                >
                  Clear All
                </button>
              )}
            </div>

            {files.length === 0 ? (
              <div className="py-16 text-center text-cyber-muted font-mono text-xs border border-dashed border-cyber-border/40 rounded-xl bg-cyber-dark/30">
                No images staged. Add 2 or more images to launch graph reconstruction.
              </div>
            ) : (
              <div className="max-h-[320px] overflow-y-auto space-y-2 pr-1">
                {files.map((file, idx) => (
                  <div key={idx} className="flex items-center justify-between p-3 rounded-xl bg-cyber-dark/80 border border-cyber-border text-xs font-mono">
                    <div className="flex items-center space-x-3 truncate">
                      <span className="w-6 h-6 rounded bg-cyber-neon/10 border border-cyber-neon/30 text-cyber-neon flex items-center justify-center font-bold text-[10px]">
                        {idx + 1}
                      </span>
                      <span className="text-white truncate max-w-[200px]">{file.name}</span>
                    </div>
                    <div className="flex items-center space-x-4">
                      <span className="text-cyber-muted">{(file.size / 1024).toFixed(1)} KB</span>
                      <button
                        onClick={() => handleRemoveFile(idx)}
                        className="text-cyber-muted hover:text-red-400 font-bold px-1"
                      >
                        ×
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Progress / Submit Bar */}
          <div className="mt-6 pt-4 border-t border-cyber-border space-y-3">
            {(uploading || analyzing) && (
              <div className="p-3 bg-cyber-neon/10 border border-cyber-neon/30 rounded-xl space-y-2">
                <div className="flex items-center justify-between text-xs font-mono text-cyber-neon">
                  <span className="flex items-center space-x-2">
                    <span className="w-2 h-2 rounded-full bg-cyber-neon animate-ping"></span>
                    <span>{progressStep}</span>
                  </span>
                </div>
                <div className="w-full bg-cyber-dark h-1.5 rounded-full overflow-hidden">
                  <div className="bg-cyber-neon h-full animate-pulse rounded-full w-3/4"></div>
                </div>
              </div>
            )}

            <button
              onClick={handleRunAnalysis}
              disabled={files.length < 2 || uploading || analyzing}
              className="w-full py-3 rounded-xl font-semibold text-xs flex items-center justify-center space-x-2 transition-all bg-cyber-neon text-black hover:bg-cyber-neon/90 shadow-[0_0_15px_rgba(0,255,136,0.3)] disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Play className="w-4 h-4 fill-black" />
              <span>{analyzing ? 'Analyzing Image Provenance Graph...' : 'Execute Provenance Graph Analysis'}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
