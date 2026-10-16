import React, { useState, useRef, useEffect } from 'react';
import { 
  UploadCloud, 
  FileImage, 
  Play, 
  Sparkles, 
  CheckCircle2, 
  AlertCircle, 
  Sliders, 
  Trash2, 
  Image as ImageIcon,
  Layers,
  Cpu,
  ArrowRight
} from 'lucide-react';
import axios from 'axios';

export default function UploadZone({ onAnalysisComplete, setActiveTab }) {
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [progressStep, setProgressStep] = useState('');
  const [progressPercent, setProgressPercent] = useState(0);
  const [simThreshold, setSimThreshold] = useState(0.50);
  const [confFilter, setConfFilter] = useState(0.35);
  const [errorMsg, setErrorMsg] = useState(null);
  
  const fileInputRef = useRef(null);

  // Synchronize previews whenever files change
  useEffect(() => {
    const urls = files.map(file => URL.createObjectURL(file));
    setPreviews(urls);
    return () => {
      urls.forEach(url => URL.revokeObjectURL(url));
    };
  }, [files]);

  const handleFileSelect = (e) => {
    const selected = Array.from(e.target.files).filter(f => f.type.startsWith('image/'));
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
      setErrorMsg("Please select at least 2 suspect images for provenance analysis.");
      return;
    }

    try {
      setUploading(true);
      setErrorMsg(null);
      setProgressPercent(15);
      setProgressStep("Staging images to secure local forensic workspace...");

      const formData = new FormData();
      files.forEach(f => formData.append('files', f));

      const uploadRes = await axios.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const { analysis_id } = uploadRes.data;
      setUploading(false);
      setAnalyzing(true);
      setProgressPercent(35);

      setProgressStep("Extracting ViT-B/16 CLS tokens (768-D) & forensic residual features...");
      await new Promise(r => setTimeout(r, 600));
      setProgressPercent(60);

      setProgressStep("Evaluating 115-D pairwise probabilities & transformation classification...");
      await new Promise(r => setTimeout(r, 600));
      setProgressPercent(85);

      setProgressStep("Constructing Kruskal Minimum Spanning Forest & orienting DAG edges...");

      const analyzeRes = await axios.post('/api/analyze', {
        analysis_id,
        similarity_threshold: simThreshold,
        confidence_filter: confFilter
      });

      setProgressPercent(100);
      setProgressStep("Graph reconstruction complete. Preparing forensic views...");
      await new Promise(r => setTimeout(r, 400));

      setAnalyzing(false);
      onAnalysisComplete(analyzeRes.data);
      setActiveTab('graph');

    } catch (err) {
      console.error(err);
      setUploading(false);
      setAnalyzing(false);
      setProgressPercent(0);
      setErrorMsg(err.response?.data?.detail || "Error during provenance analysis execution.");
    }
  };

  const handleRunDemo = async () => {
    try {
      setAnalyzing(true);
      setErrorMsg(null);
      setProgressPercent(30);
      setProgressStep("Staging benchmark DAG (graph_0001)...");
      await new Promise(r => setTimeout(r, 500));
      setProgressPercent(70);

      setProgressStep("Running ViT-B/16 ML Pairwise & MST Provenance Inference...");

      const demoRes = await axios.post('/api/demo?graph_id=graph_0001');

      setProgressPercent(100);
      setProgressStep("Reconstruction validated. Opening graph canvas...");
      await new Promise(r => setTimeout(r, 300));

      setAnalyzing(false);
      onAnalysisComplete(demoRes.data);
      setActiveTab('graph');
    } catch (err) {
      console.error(err);
      setAnalyzing(false);
      setProgressPercent(0);
      setErrorMsg(err.response?.data?.detail || "Failed to launch benchmark demo.");
    }
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Title Card */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-[#F4F7F5] tracking-tight flex items-center space-x-2 font-sans">
            <UploadCloud className="w-6 h-6 text-[#39FF88]" />
            <span>Multi-Image Provenance Ingestion & Reconstruction</span>
          </h1>
          <p className="text-xs text-[#9AA8A0] font-mono pt-1">
            Ingest suspect image collections, extract ViT deep CLS features, and reconstruct parent-child edit trees.
          </p>
        </div>

        <button
          onClick={handleRunDemo}
          disabled={uploading || analyzing}
          className="bg-[#111916] hover:bg-[#1A2822] text-[#35D9FF] hover:text-white font-mono text-xs px-4 py-2.5 rounded-xl border border-[#35D9FF]/40 flex items-center space-x-2 transition-all shadow-[0_0_15px_rgba(53,217,255,0.15)] disabled:opacity-50 cursor-pointer"
        >
          <Sparkles className="w-4 h-4" />
          <span>Load Benchmark Demo (DAG #0001)</span>
        </button>
      </div>

      {errorMsg && (
        <div className="bg-red-500/10 border border-red-500/40 p-4 rounded-2xl text-red-400 text-xs flex items-center space-x-3 font-mono">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Dropzone & Parameters */}
        <div className="lg:col-span-1 space-y-6">
          {/* Dropzone */}
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-[#1A2822] hover:border-[#39FF88] bg-[#111916]/80 p-8 rounded-3xl text-center cursor-pointer transition-all hover:bg-[#39FF88]/5 group space-y-3"
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              multiple
              accept="image/*"
              className="hidden"
            />
            <div className="w-14 h-14 rounded-2xl bg-[#070909] border border-[#39FF88]/40 text-[#39FF88] mx-auto flex items-center justify-center group-hover:scale-110 transition-transform shadow-[0_0_20px_rgba(57,255,136,0.15)]">
              <UploadCloud className="w-7 h-7" />
            </div>
            <div>
              <p className="text-sm font-bold text-white font-sans">Drag & drop suspect images</p>
              <p className="text-xs text-[#9AA8A0] font-mono mt-1">PNG, JPG, JPEG, WEBP</p>
            </div>
            <button className="text-xs font-mono text-black font-bold bg-[#39FF88] px-4 py-2 rounded-xl group-hover:bg-[#D7FF32] transition-colors">
              Browse Local Files
            </button>
          </div>

          {/* Hyperparameters Card */}
          <div className="bg-[#111916] p-5 rounded-2xl border border-[#1A2822] space-y-4 font-mono">
            <div className="flex items-center space-x-2 text-xs font-bold text-white border-b border-[#1A2822] pb-2 font-sans">
              <Sliders className="w-4 h-4 text-[#39FF88]" />
              <span>Inference Hyperparameters</span>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-[#9AA8A0]">Pair Similarity Threshold (&tau;):</span>
                <span className="text-[#39FF88] font-bold">{simThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.30"
                max="0.95"
                step="0.05"
                value={simThreshold}
                onChange={(e) => setSimThreshold(parseFloat(e.target.value))}
                className="w-full accent-[#39FF88] cursor-pointer bg-[#070909] rounded-lg h-2"
              />
              <span className="text-[10px] text-[#5F7067] block mt-1">
                Lower &tau; admits weaker candidate links; higher &tau; retains only high-confidence matches.
              </span>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-[#9AA8A0]">Kruskal Edge Confidence Filter:</span>
                <span className="text-[#39FF88] font-bold">{confFilter.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.20"
                max="0.90"
                step="0.05"
                value={confFilter}
                onChange={(e) => setConfFilter(parseFloat(e.target.value))}
                className="w-full accent-[#39FF88] cursor-pointer bg-[#070909] rounded-lg h-2"
              />
              <span className="text-[10px] text-[#5F7067] block mt-1">
                Minimum relationship probability P(related) required for spanning forest inclusion.
              </span>
            </div>
          </div>
        </div>

        {/* Right Column: Staged Image Cards & Action */}
        <div className="lg:col-span-2 bg-[#111916] p-6 rounded-3xl border border-[#1A2822] flex flex-col justify-between space-y-6">
          <div>
            <div className="flex items-center justify-between border-b border-[#1A2822] pb-3 mb-4">
              <div className="flex items-center space-x-2">
                <FileImage className="w-4 h-4 text-[#39FF88]" />
                <h3 className="text-sm font-bold text-white font-sans">
                  Staged Suspect Images ({files.length})
                </h3>
              </div>
              {files.length > 0 && (
                <button
                  onClick={() => setFiles([])}
                  className="text-xs text-red-400 hover:text-red-300 font-mono transition-colors flex items-center space-x-1"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Clear All</span>
                </button>
              )}
            </div>

            {files.length === 0 ? (
              <div className="py-24 text-center text-[#9AA8A0] font-mono text-xs border border-dashed border-[#1A2822] rounded-2xl bg-[#070909]/60 flex flex-col items-center justify-center space-y-2">
                <ImageIcon className="w-8 h-8 text-[#5F7067] opacity-60" />
                <p>No images staged in workspace.</p>
                <p className="text-[11px] text-[#5F7067]">Add 2 or more suspect images to launch graph reconstruction.</p>
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 max-h-[380px] overflow-y-auto pr-1">
                {files.map((file, idx) => (
                  <div 
                    key={idx} 
                    className="bg-[#070909] border border-[#1A2822] rounded-xl p-2 relative group hover:border-[#39FF88]/50 transition-all flex flex-col justify-between"
                  >
                    {/* Thumbnail preview */}
                    <div className="h-24 bg-[#111916] rounded-lg overflow-hidden flex items-center justify-center mb-2">
                      <img 
                        src={previews[idx]} 
                        alt={file.name} 
                        className="h-full w-full object-cover group-hover:scale-105 transition-transform" 
                      />
                    </div>
                    {/* Metadata */}
                    <div className="text-[10px] font-mono">
                      <div className="text-white font-semibold truncate" title={file.name}>
                        {file.name}
                      </div>
                      <div className="flex justify-between text-[#5F7067] mt-0.5">
                        <span>#0{idx + 1}</span>
                        <span>{(file.size / 1024).toFixed(0)} KB</span>
                      </div>
                    </div>
                    {/* Remove button */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRemoveFile(idx);
                      }}
                      className="absolute top-3 right-3 bg-red-500/80 hover:bg-red-500 text-white w-5 h-5 rounded-full flex items-center justify-center text-xs opacity-0 group-hover:opacity-100 transition-opacity shadow-md"
                      title="Remove image"
                    >
                      &times;
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Execution Progress & Submit Bar */}
          <div className="pt-4 border-t border-[#1A2822] space-y-3 font-mono">
            {(uploading || analyzing) && (
              <div className="p-4 bg-[#070909] border border-[#39FF88]/30 rounded-2xl space-y-2.5">
                <div className="flex items-center justify-between text-xs text-[#39FF88]">
                  <span className="flex items-center space-x-2">
                    <span className="w-2 h-2 rounded-full bg-[#39FF88] animate-ping" />
                    <span>{progressStep}</span>
                  </span>
                  <span className="font-bold">{progressPercent}%</span>
                </div>
                <div className="w-full bg-[#111916] h-2 rounded-full overflow-hidden border border-[#1A2822]">
                  <div 
                    className="bg-gradient-to-r from-[#39FF88] to-[#D7FF32] h-full rounded-full transition-all duration-300" 
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
              </div>
            )}

            <button
              onClick={handleRunAnalysis}
              disabled={files.length < 2 || uploading || analyzing}
              className="w-full py-3.5 rounded-full font-bold text-xs flex items-center justify-center space-x-2 transition-all bg-[#39FF88] text-black hover:bg-[#D7FF32] shadow-[0_0_20px_rgba(57,255,136,0.35)] disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer uppercase tracking-wider"
            >
              <Play className="w-4 h-4 fill-black" />
              <span>{analyzing ? 'Reconstructing Provenance Graph...' : 'Reconstruct Image Provenance Graph'}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
