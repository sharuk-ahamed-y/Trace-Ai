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
  ArrowRight,
  X,
  Compass,
  Check,
  RotateCw,
  Info
} from 'lucide-react';
import axios from 'axios';

export default function UploadZone({ onAnalysisComplete, setActiveTab }) {
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(1);
  const [progressPercent, setProgressPercent] = useState(0);
  const [timeRemaining, setTimeRemaining] = useState(8);
  const [simThreshold, setSimThreshold] = useState(0.45);
  const [confFilter, setConfFilter] = useState(0.35);
  const [errorMsg, setErrorMsg] = useState(null);

  // Settings toggles
  const [detectTransformations, setDetectTransformations] = useState(true);
  const [buildGraph, setBuildGraph] = useState(true);
  const [autoRootDetection, setAutoRootDetection] = useState(true);

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

  const handleClearAll = () => {
    setFiles([]);
    setErrorMsg(null);
  };

  // 6 Processing steps matching Screen 3 in reference showcase
  const processingSteps = [
    { title: '1. Image Preprocessing', duration: '2.1s' },
    { title: '2. Feature Extraction', duration: '3.1s' },
    { title: '3. Pairwise Analysis', duration: '4.2s' },
    { title: '4. Graph Construction', duration: 'Pending' },
    { title: '5. Root Detection', duration: 'Pending' },
    { title: '6. Results Generation', duration: 'Pending' },
  ];

  const handleRunAnalysis = async () => {
    if (files.length < 2) {
      setErrorMsg("Please select at least 2 suspect images for provenance analysis.");
      return;
    }

    try {
      setUploading(true);
      setAnalyzing(true);
      setErrorMsg(null);
      setProgressPercent(15);
      setCurrentStepIndex(0);
      setTimeRemaining(10);

      const formData = new FormData();
      files.forEach(f => formData.append('files', f));

      const uploadRes = await axios.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const { analysis_id } = uploadRes.data;
      setUploading(false);

      // Step 2: Feature Extraction
      setCurrentStepIndex(1);
      setProgressPercent(35);
      setTimeRemaining(7);
      await new Promise(r => setTimeout(r, 700));

      // Step 3: Pairwise Analysis
      setCurrentStepIndex(2);
      setProgressPercent(60);
      setTimeRemaining(5);
      await new Promise(r => setTimeout(r, 700));

      // Step 4 & 5: Graph Construction & Root Detection
      setCurrentStepIndex(3);
      setProgressPercent(80);
      setTimeRemaining(2);

      const analyzeRes = await axios.post('/api/analyze', {
        analysis_id,
        similarity_threshold: simThreshold,
        confidence_filter: confFilter
      });

      // Step 6: Results Generation
      setCurrentStepIndex(5);
      setProgressPercent(100);
      setTimeRemaining(0);
      await new Promise(r => setTimeout(r, 400));

      setAnalyzing(false);
      onAnalysisComplete(analyzeRes.data);
      setActiveTab('analysis');

    } catch (err) {
      console.error(err);
      setUploading(false);
      setAnalyzing(false);
      setProgressPercent(0);
      setErrorMsg(err.response?.data?.detail || "Error during provenance analysis execution.");
    }
  };

  // =========================================================================
  // VIEW 1: "ANALYSIS IN PROGRESS" SCREEN (Screen 3 in Reference Showcase)
  // =========================================================================
  if (analyzing) {
    return (
      <div className="max-w-6xl mx-auto space-y-8 py-6">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white font-display">
            Analysis in Progress
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 font-mono pt-1">
            TRACE AI is preprocessing, extracting ViT features, and building the lineage graph...
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          {/* Left Column: 6-Step Checklist */}
          <div className="lg:col-span-7 glass-panel rounded-3xl p-6 sm:p-8 border border-gold-border/30 space-y-4">
            {processingSteps.map((step, idx) => {
              const isCompleted = idx < currentStepIndex;
              const isCurrent = idx === currentStepIndex;
              const isPending = idx > currentStepIndex;

              return (
                <div 
                  key={idx}
                  className={`flex items-center justify-between p-3.5 rounded-2xl transition-all ${
                    isCurrent 
                      ? 'bg-cyan-electric/10 border border-cyan-electric/40 shadow-cyan-glow' 
                      : isCompleted
                      ? 'bg-obsidian-800/60 border border-gold-border/20'
                      : 'bg-obsidian-900/40 border border-white/5 opacity-50'
                  }`}
                >
                  <div className="flex items-center space-x-3.5">
                    {/* Status Circle */}
                    <div className="w-8 h-8 rounded-full flex items-center justify-center shrink-0">
                      {isCompleted ? (
                        <div className="w-7 h-7 rounded-full bg-emerald-500/20 border border-emerald-400 flex items-center justify-center text-emerald-400">
                          <Check className="w-4 h-4 stroke-[3]" />
                        </div>
                      ) : isCurrent ? (
                        <div className="w-7 h-7 rounded-full bg-cyan-electric/20 border border-cyan-electric flex items-center justify-center text-cyan-electric animate-spin">
                          <RotateCw className="w-4 h-4" />
                        </div>
                      ) : (
                        <div className="w-7 h-7 rounded-full border border-slate-600 flex items-center justify-center text-slate-500 text-xs font-mono">
                          {idx + 1}
                        </div>
                      )}
                    </div>

                    <span className={`text-xs sm:text-sm font-medium ${isCurrent ? 'text-white font-bold' : isCompleted ? 'text-slate-200' : 'text-slate-400'}`}>
                      {step.title}
                    </span>
                  </div>

                  <span className={`text-xs font-mono ${isCompleted ? 'text-emerald-400' : isCurrent ? 'text-cyan-electric font-bold' : 'text-slate-400'}`}>
                    {isCompleted ? `Completed (${step.duration})` : isCurrent ? 'Processing...' : 'Pending'}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Right Column: 3D Holographic Container + Radial Progress */}
          <div className="lg:col-span-5 flex flex-col items-center justify-center space-y-6">
            <div className="relative w-64 h-64 rounded-3xl glass-panel-elevated p-6 border border-gold-radiant/50 flex flex-col items-center justify-center shadow-gold-glow">
              {/* Concentric Golden Orbital Glow */}
              <div className="absolute inset-2 rounded-2xl border border-dashed border-cyan-electric/30 animate-orbital pointer-events-none" />

              {/* Radial Progress Gauge SVG */}
              <div className="relative w-36 h-36 flex items-center justify-center">
                <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
                  <circle
                    cx="50" cy="50" r="42"
                    fill="none"
                    stroke="#141C2E"
                    strokeWidth="8"
                  />
                  <circle
                    cx="50" cy="50" r="42"
                    fill="none"
                    stroke="url(#progressGradient)"
                    strokeWidth="8"
                    strokeDasharray="264"
                    strokeDashoffset={264 - (264 * progressPercent) / 100}
                    strokeLinecap="round"
                    className="transition-all duration-500"
                  />
                  <defs>
                    <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#00F2FE" />
                      <stop offset="100%" stopColor="#D4AF37" />
                    </linearGradient>
                  </defs>
                </svg>

                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-3xl font-extrabold text-white font-display">
                    {progressPercent}%
                  </span>
                </div>
              </div>

              <div className="text-center pt-4 space-y-1">
                <div className="text-xs font-mono font-semibold text-cyan-electric animate-pulse">
                  Processing...
                </div>
                <div className="text-[11px] font-mono text-slate-400">
                  {timeRemaining > 0 ? `~${timeRemaining} seconds remaining` : 'Finalizing Graph...'}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // =========================================================================
  // VIEW 2: "UPLOAD & ANALYSIS" SCREEN (Screen 2 in Reference Showcase)
  // =========================================================================
  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-12">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white font-display">
          Upload & Analysis
        </h1>
        <p className="text-xs sm:text-sm text-slate-400 font-mono pt-1">
          Upload your images and let TRACE AI analyze their relationships.
        </p>
      </div>

      {errorMsg && (
        <div className="p-4 rounded-2xl bg-red-500/10 border border-red-500/30 text-red-300 text-xs font-mono flex items-center space-x-3">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Drag & Drop Zone + Uploaded Images List */}
        <div className="lg:col-span-8 space-y-6">
          {/* Large Drag & Drop Box with Glowing Cyan Cloud */}
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="relative rounded-3xl glass-panel p-10 sm:p-14 border-2 border-dashed border-cyan-electric/40 hover:border-cyan-electric transition-all flex flex-col items-center justify-center cursor-pointer text-center group shadow-card-dark"
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              multiple
              accept="image/*"
              className="hidden"
            />

            {/* Glowing Cloud Icon */}
            <div className="w-20 h-20 rounded-2xl bg-cyan-electric/10 border border-cyan-electric/50 flex items-center justify-center text-cyan-electric shadow-cyan-glow group-hover:scale-105 transition-transform mb-4">
              <UploadCloud className="w-10 h-10 stroke-[1.8]" />
            </div>

            <h3 className="text-base sm:text-lg font-bold text-white font-display">
              Drag & drop images here
            </h3>
            <p className="text-xs font-mono text-cyan-electric pt-1">
              or click to browse
            </p>
            <p className="text-[11px] font-mono text-slate-400 pt-3">
              Supports JPG, PNG, WEBP (Max 10MB per file)
            </p>
          </div>

          {/* Uploaded Images Section */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-mono font-bold text-white uppercase tracking-wider">
                Uploaded Images ({files.length})
              </h4>
              {files.length > 0 && (
                <button
                  onClick={handleClearAll}
                  className="text-xs font-mono text-red-400 hover:text-red-300 hover:underline cursor-pointer"
                >
                  Clear All
                </button>
              )}
            </div>

            {files.length > 0 ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                {files.map((file, idx) => (
                  <div
                    key={idx}
                    className="relative rounded-2xl glass-card p-2 border border-gold-border/30 group overflow-hidden shadow-card-dark"
                  >
                    <div className="w-full h-28 rounded-xl overflow-hidden bg-obsidian-900 relative">
                      <img
                        src={previews[idx]}
                        alt={file.name}
                        className="w-full h-full object-cover"
                      />
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRemoveFile(idx);
                        }}
                        className="absolute top-1.5 right-1.5 w-6 h-6 rounded-full bg-obsidian-950/80 hover:bg-red-500 text-white flex items-center justify-center transition-colors cursor-pointer"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    <div className="p-2 space-y-0.5">
                      <p className="text-[11px] font-mono text-slate-200 font-semibold truncate">
                        {file.name}
                      </p>
                      <p className="text-[10px] font-mono text-slate-400">
                        {(file.size / 1024).toFixed(0)} KB
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-8 text-center text-xs font-mono text-slate-400 glass-card rounded-2xl border border-white/5">
                No images staged yet. Drag and drop at least 2 images above.
              </div>
            )}
          </div>

          {/* Wide Glowing Champagne Gold Action Button */}
          <div>
            <button
              onClick={handleRunAnalysis}
              disabled={files.length < 2 || uploading}
              className="w-full bg-gold-gradient hover:opacity-95 text-black font-bold py-4 rounded-2xl text-sm font-mono flex items-center justify-center space-x-2.5 shadow-gold-glow hover:shadow-gold-glow-lg transition-all cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Play className="w-4 h-4 fill-black text-black" />
              <span>Run Provenance Analysis</span>
            </button>
          </div>
        </div>

        {/* Right Column: Analysis Settings */}
        <div className="lg:col-span-4 space-y-5">
          {/* Analysis Settings Card */}
          <div className="glass-panel rounded-3xl p-6 border border-gold-border/30 space-y-5">
            <h3 className="text-sm font-bold text-white font-display uppercase tracking-wider">
              Analysis Settings
            </h3>

            <div className="space-y-4">
              {/* Toggle 1: Detect Transformations */}
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs font-medium text-slate-200">Detect Transformations</div>
                  <div className="text-[10px] font-mono text-slate-400">Crop, blur, resize, photometric</div>
                </div>
                <button
                  type="button"
                  onClick={() => setDetectTransformations(!detectTransformations)}
                  className={`w-11 h-6 rounded-full transition-colors p-1 cursor-pointer ${
                    detectTransformations ? 'bg-gold-primary' : 'bg-obsidian-800'
                  }`}
                >
                  <div className={`w-4 h-4 rounded-full bg-white transition-transform ${
                    detectTransformations ? 'translate-x-5' : 'translate-x-0'
                  }`} />
                </button>
              </div>

              {/* Toggle 2: Build Graph */}
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs font-medium text-slate-200">Build Graph</div>
                  <div className="text-[10px] font-mono text-slate-400">Kruskal MST reconstruction</div>
                </div>
                <button
                  type="button"
                  onClick={() => setBuildGraph(!buildGraph)}
                  className={`w-11 h-6 rounded-full transition-colors p-1 cursor-pointer ${
                    buildGraph ? 'bg-gold-primary' : 'bg-obsidian-800'
                  }`}
                >
                  <div className={`w-4 h-4 rounded-full bg-white transition-transform ${
                    buildGraph ? 'translate-x-5' : 'translate-x-0'
                  }`} />
                </button>
              </div>

              {/* Toggle 3: Auto Root Detection */}
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs font-medium text-slate-200">Auto Root Detection</div>
                  <div className="text-[10px] font-mono text-slate-400">Identify true original image</div>
                </div>
                <button
                  type="button"
                  onClick={() => setAutoRootDetection(!autoRootDetection)}
                  className={`w-11 h-6 rounded-full transition-colors p-1 cursor-pointer ${
                    autoRootDetection ? 'bg-gold-primary' : 'bg-obsidian-800'
                  }`}
                >
                  <div className={`w-4 h-4 rounded-full bg-white transition-transform ${
                    autoRootDetection ? 'translate-x-5' : 'translate-x-0'
                  }`} />
                </button>
              </div>
            </div>
          </div>

          {/* Feature Callout Card matching reference */}
          <div className="glass-card p-5 rounded-2xl border border-gold-border/30 flex items-start space-x-3.5">
            <div className="w-8 h-8 rounded-xl bg-gold-primary/10 border border-gold-primary/30 flex items-center justify-center text-gold-radiant shrink-0">
              <Sparkles className="w-4 h-4" />
            </div>
            <p className="text-xs font-sans text-slate-300 leading-relaxed">
              AI will detect crops, edits, and build the image lineage graph automatically.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
