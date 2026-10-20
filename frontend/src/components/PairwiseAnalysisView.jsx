import React, { useState, useEffect } from 'react';
import { 
  GitCompare, 
  ArrowRight, 
  ShieldCheck, 
  AlertTriangle, 
  CheckCircle2, 
  Sparkles, 
  Sliders, 
  Layers, 
  Zap,
  Info
} from 'lucide-react';
import axios from 'axios';

export default function PairwiseAnalysisView({ initialImages }) {
  const [availableImages, setAvailableImages] = useState([]);
  const [selectedA, setSelectedA] = useState('');
  const [selectedB, setSelectedB] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  // Fetch available images
  useEffect(() => {
    axios.get('/api/images')
      .then(res => {
        const imgs = res.data.images || [];
        setAvailableImages(imgs);
        if (imgs.length >= 2) {
          setSelectedA(imgs[0].file_path || imgs[0].id);
          setSelectedB(imgs[1].file_path || imgs[1].id);
        }
      })
      .catch(err => console.error("Error loading images list:", err));
  }, []);

  const handlePredictPair = async () => {
    if (!selectedA || !selectedB) {
      setErrorMsg("Please select both Image A and Image B.");
      return;
    }
    if (selectedA === selectedB) {
      setErrorMsg("Please select two distinct images for comparison.");
      return;
    }

    try {
      setLoading(true);
      setErrorMsg(null);
      const res = await axios.post('/api/predict-pair', {
        image_a_path: selectedA,
        image_b_path: selectedB
      });
      setResult(res.data);
      setLoading(false);
    } catch (err) {
      console.error(err);
      setLoading(false);
      setErrorMsg(err.response?.data?.detail || "Pairwise prediction failed.");
    }
  };

  const imgObjA = availableImages.find(i => (i.file_path === selectedA || i.id === selectedA));
  const imgObjB = availableImages.find(i => (i.file_path === selectedB || i.id === selectedB));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-[#F4F7F5] tracking-tight flex items-center space-x-2">
            <GitCompare className="w-6 h-6 text-[#39FF88]" />
            <span>Pairwise Image Relationship & Transformation Analysis</span>
          </h1>
          <p className="text-xs text-[#9AA8A0] font-mono pt-1">
            Evaluates pair relatedness, directional derivation hypothesis, and transformation classification.
          </p>
        </div>

        {/* Action Button */}
        <button
          onClick={handlePredictPair}
          disabled={loading || !selectedA || !selectedB}
          className="bg-gold-gradient hover:opacity-95 text-black font-bold px-6 py-2.5 rounded-full text-xs font-mono flex items-center space-x-2 shadow-gold-glow disabled:opacity-50 transition-all cursor-pointer"
        >
          {loading ? (
            <>
              <div className="w-3.5 h-3.5 border-2 border-black border-t-transparent rounded-full animate-spin" />
              <span>Computing 115-D Inference...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4 fill-black text-black" />
              <span>Run Pairwise Analysis</span>
            </>
          )}
        </button>
      </div>

      {errorMsg && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 p-4 rounded-xl text-xs font-mono flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Image Selection & Side-by-Side Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Card Image A */}
        <div className="bg-[#111916] border border-[#1A2822] rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="w-6 h-6 rounded-full bg-[#39FF88] text-black font-bold text-xs flex items-center justify-center">A</span>
              <h3 className="text-sm font-bold text-[#F4F7F5]">Image Candidate A</h3>
            </div>
            <select
              value={selectedA}
              onChange={(e) => setSelectedA(e.target.value)}
              className="bg-[#070909] border border-[#1A2822] rounded-lg px-3 py-1.5 text-xs text-[#F4F7F5] font-mono focus:border-[#39FF88] focus:outline-none"
            >
              {availableImages.map(img => (
                <option key={img.file_path || img.id} value={img.file_path || img.id}>
                  {img.filename || img.id}
                </option>
              ))}
            </select>
          </div>

          {/* Preview Box */}
          <div className="w-full h-64 bg-[#070909] rounded-xl border border-[#1A2822] flex items-center justify-center overflow-hidden p-2 relative group">
            {imgObjA?.url ? (
              <img 
                src={imgObjA.url} 
                alt="Image A" 
                className="max-h-full max-w-full object-contain rounded-lg"
              />
            ) : (
              <div className="text-xs font-mono text-[#5F7067]">No preview available</div>
            )}
            <div className="absolute bottom-2 left-2 bg-[#070909]/90 px-2 py-1 rounded text-[10px] font-mono text-[#9AA8A0] border border-[#1A2822]">
              {imgObjA?.filename || 'Image A'}
            </div>
          </div>

          {/* Metadata */}
          <div className="grid grid-cols-2 gap-2 text-[11px] font-mono pt-1">
            <div className="bg-[#070909] p-2 rounded-lg border border-[#1A2822]">
              <span className="text-[#5F7067]">File Size:</span>
              <span className="text-[#F4F7F5] ml-1 font-semibold">{imgObjA?.size ? (imgObjA.size / 1024).toFixed(1) + ' KB' : 'N/A'}</span>
            </div>
            <div className="bg-[#070909] p-2 rounded-lg border border-[#1A2822]">
              <span className="text-[#5F7067]">Role:</span>
              <span className="text-[#39FF88] ml-1 font-semibold">Source Candidate</span>
            </div>
          </div>
        </div>

        {/* Card Image B */}
        <div className="bg-[#111916] border border-[#1A2822] rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="w-6 h-6 rounded-full bg-[#35D9FF] text-black font-bold text-xs flex items-center justify-center">B</span>
              <h3 className="text-sm font-bold text-[#F4F7F5]">Image Candidate B</h3>
            </div>
            <select
              value={selectedB}
              onChange={(e) => setSelectedB(e.target.value)}
              className="bg-[#070909] border border-[#1A2822] rounded-lg px-3 py-1.5 text-xs text-[#F4F7F5] font-mono focus:border-[#35D9FF] focus:outline-none"
            >
              {availableImages.map(img => (
                <option key={img.file_path || img.id} value={img.file_path || img.id}>
                  {img.filename || img.id}
                </option>
              ))}
            </select>
          </div>

          {/* Preview Box */}
          <div className="w-full h-64 bg-[#070909] rounded-xl border border-[#1A2822] flex items-center justify-center overflow-hidden p-2 relative group">
            {imgObjB?.url ? (
              <img 
                src={imgObjB.url} 
                alt="Image B" 
                className="max-h-full max-w-full object-contain rounded-lg"
              />
            ) : (
              <div className="text-xs font-mono text-[#5F7067]">No preview available</div>
            )}
            <div className="absolute bottom-2 left-2 bg-[#070909]/90 px-2 py-1 rounded text-[10px] font-mono text-[#9AA8A0] border border-[#1A2822]">
              {imgObjB?.filename || 'Image B'}
            </div>
          </div>

          {/* Metadata */}
          <div className="grid grid-cols-2 gap-2 text-[11px] font-mono pt-1">
            <div className="bg-[#070909] p-2 rounded-lg border border-[#1A2822]">
              <span className="text-[#5F7067]">File Size:</span>
              <span className="text-[#F4F7F5] ml-1 font-semibold">{imgObjB?.size ? (imgObjB.size / 1024).toFixed(1) + ' KB' : 'N/A'}</span>
            </div>
            <div className="bg-[#070909] p-2 rounded-lg border border-[#1A2822]">
              <span className="text-[#5F7067]">Role:</span>
              <span className="text-[#35D9FF] ml-1 font-semibold">Target Candidate</span>
            </div>
          </div>
        </div>

      </div>

      {/* Prediction Output Panel */}
      {result && (
        <div className="bg-[#111916] border border-[#39FF88]/40 rounded-2xl p-6 space-y-6 shadow-[0_0_25px_rgba(57,255,136,0.1)]">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#1A2822] pb-4">
            <div>
              <div className="text-xs font-mono text-[#9AA8A0]">Inference Engine</div>
              <div className="text-sm font-bold text-[#39FF88] font-mono">{result.prediction.active_model}</div>
            </div>
            <div className="flex items-center space-x-3">
              <span className={`px-4 py-1.5 rounded-full text-xs font-mono font-bold uppercase tracking-wider ${
                result.prediction.related 
                  ? 'bg-[#39FF88]/20 text-[#39FF88] border border-[#39FF88]/40 shadow-[0_0_15px_rgba(57,255,136,0.2)]'
                  : 'bg-red-500/20 text-red-400 border border-red-500/40'
              }`}>
                {result.prediction.related ? 'PROVENANCE RELATED' : 'UNRELATED IMAGES'}
              </span>
            </div>
          </div>

          {/* 4 Metric Columns */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-[#070909] p-4 rounded-xl border border-[#1A2822] space-y-1">
              <div className="text-[11px] font-mono text-[#9AA8A0]">Relationship Probability</div>
              <div className="text-2xl font-extrabold text-[#F4F7F5] font-sans">
                {(result.prediction.related_confidence * 100).toFixed(1)}%
              </div>
              <div className="w-full bg-[#111916] h-1.5 rounded-full overflow-hidden">
                <div 
                  className={`h-full rounded-full ${result.prediction.related ? 'bg-[#39FF88]' : 'bg-red-400'}`} 
                  style={{ width: `${result.prediction.related_confidence * 100}%` }} 
                />
              </div>
            </div>

            <div className="bg-[#070909] p-4 rounded-xl border border-[#1A2822] space-y-1">
              <div className="text-[11px] font-mono text-[#9AA8A0]">Predicted Direction</div>
              <div className="text-2xl font-extrabold text-[#35D9FF] font-sans flex items-center space-x-2">
                <span>{result.prediction.direction}</span>
              </div>
              <div className="text-[10px] text-[#5F7067] font-mono">
                Confidence: {(result.prediction.direction_confidence * 100).toFixed(1)}%
              </div>
            </div>

            <div className="bg-[#070909] p-4 rounded-xl border border-[#1A2822] space-y-1">
              <div className="text-[11px] font-mono text-[#9AA8A0]">Transformation Type</div>
              <div className="text-2xl font-extrabold text-[#D7FF32] font-sans uppercase">
                {result.prediction.transformation}
              </div>
              <div className="text-[10px] text-[#5F7067] font-mono">
                Confidence: {(result.prediction.transformation_confidence * 100).toFixed(1)}%
              </div>
            </div>

            <div className="bg-[#070909] p-4 rounded-xl border border-[#1A2822] space-y-1">
              <div className="text-[11px] font-mono text-[#9AA8A0]">Cosine Similarity (ViT)</div>
              <div className="text-2xl font-extrabold text-[#F4F7F5] font-sans">
                {result.prediction.cosine_similarity.toFixed(4)}
              </div>
              <div className="text-[10px] text-[#5F7067] font-mono">
                768-D CLS Semantic Align
              </div>
            </div>
          </div>

          {/* Feature Sensitivity Group Breakdown */}
          {result.feature_sensitivity && (
            <div className="bg-[#070909] p-4 rounded-xl border border-[#1A2822] space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-[#F4F7F5] font-bold">Feature Sensitivity Contributions</span>
                <span className="text-[10px] font-mono text-[#5F7067]">Empirical representation variance</span>
              </div>
              <div className="space-y-2">
                {result.feature_sensitivity.map((group, idx) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="text-[#9AA8A0]">{group.group}</span>
                      <span className="text-[#39FF88] font-bold">{(group.normalized_weight * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-[#111916] h-1.5 rounded-full overflow-hidden">
                      <div 
                        className="bg-gradient-to-r from-[#39FF88] to-[#35D9FF] h-full rounded-full" 
                        style={{ width: `${group.normalized_weight * 100}%` }} 
                      />
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-xs text-[#9AA8A0] font-mono pt-2 border-t border-[#1A2822]">
                {result.interpretation}
              </p>
            </div>
          )}

          {/* Forensic Integrity Disclaimer */}
          <div className="bg-[#070909] p-3.5 rounded-xl border border-amber-500/20 text-amber-300/80 text-[11px] font-mono flex items-start space-x-2">
            <Info className="w-4 h-4 shrink-0 text-amber-400 mt-0.5" />
            <span>
              <strong>Forensic Notice:</strong> Model-predicted relationship is a statistical inference derived from 
              ViT visual latents and physical/spectral deltas. High visual similarity alone does not establish confirmed forensic provenance.
            </span>
          </div>

        </div>
      )}
    </div>
  );
}
