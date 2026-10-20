import React, { useState, useEffect } from 'react';
import { Database, Image as ImageIcon, GitFork, CheckCircle, ShieldCheck, Play, ArrowRight, Sparkles, Layers } from 'lucide-react';
import axios from 'axios';

export default function DatasetExplorer({ onAnalysisComplete, setActiveTab }) {
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [activeTabSub, setActiveTabSub] = useState('benchmark'); // 'benchmark' | 'samples'
  const [loading, setLoading] = useState(true);
  const [runningDag, setRunningDag] = useState(null);

  useEffect(() => {
    axios.get('/api/dataset')
      .then(res => {
        setDatasetInfo(res.data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  const handleLaunchDagDemo = async (dagId) => {
    try {
      setRunningDag(dagId);
      const demoRes = await axios.post(`/api/demo?graph_id=${dagId}`);
      setRunningDag(null);
      onAnalysisComplete(demoRes.data);
      setActiveTab('analysis');
    } catch (err) {
      console.error(err);
      setRunningDag(null);
    }
  };

  const dags = [
    { id: 'graph_0001', name: 'Graph 0001', images: 5, edges: 4, transforms: 'Crop, Blur, JPEG', thumb: '/data/uploads/anl_20261020_140601_558b3c/WhatsApp_Image_2026-10-18_at_10.28.56_PM_(1).jpeg' },
    { id: 'graph_0002', name: 'Graph 0002', images: 6, edges: 5, transforms: 'Crop, Rescale, Noise', thumb: '/data/uploads/anl_20261020_140601_558b3c/WhatsApp_Image_2026-10-18_at_10.28.57_PM_(1).jpeg' },
    { id: 'graph_0003', name: 'Graph 0003', images: 4, edges: 3, transforms: 'Rotation, Perspective', thumb: '/data/uploads/anl_20261020_140601_558b3c/WhatsApp_Image_2026-10-18_at_10.28.57_PM_(2).jpeg' },
    { id: 'graph_0004', name: 'Graph 0004', images: 5, edges: 4, transforms: 'Brightness, Contrast', thumb: 'https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=500&auto=format&fit=crop&q=60' },
    { id: 'graph_0005', name: 'Graph 0005', images: 7, edges: 6, transforms: 'Multi-Generation Edit', thumb: 'https://images.unsplash.com/photo-1511884642898-4c92249e20b6?w=500&auto=format&fit=crop&q=60' },
    { id: 'graph_0006', name: 'Graph 0006', images: 4, edges: 3, transforms: 'WhatsApp Compression', thumb: 'https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=500&auto=format&fit=crop&q=60' },
  ];

  return (
    <div className="max-w-7xl mx-auto space-y-8 pb-12">
      {/* Header matching Screen 8 in Reference Showcase */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white font-display">
            Dataset
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 font-mono pt-1">
            Explore benchmark and sample datasets.
          </p>
        </div>

        {/* Tab Pills: Benchmark / Samples matching Screen 8 */}
        <div className="flex items-center space-x-1 glass-card p-1 rounded-full border border-gold-border/30 self-start sm:self-auto">
          <button
            onClick={() => setActiveTabSub('benchmark')}
            className={`px-4 py-1.5 rounded-full text-xs font-mono font-semibold transition-all cursor-pointer ${
              activeTabSub === 'benchmark'
                ? 'bg-gold-gradient text-black shadow-gold-glow'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Benchmark
          </button>
          <button
            onClick={() => setActiveTabSub('samples')}
            className={`px-4 py-1.5 rounded-full text-xs font-mono font-semibold transition-all cursor-pointer ${
              activeTabSub === 'samples'
                ? 'bg-gold-gradient text-black shadow-gold-glow'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Samples
          </button>
        </div>
      </div>

      {/* Grid of Benchmark DAG Cards matching Screen 8 */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {dags.map((dag) => (
          <div
            key={dag.id}
            className="glass-card glass-card-hover rounded-3xl p-4 border border-gold-border/30 overflow-hidden shadow-card-dark space-y-3.5 group flex flex-col justify-between"
          >
            <div className="space-y-3">
              {/* Image Preview Container with Glowing Border */}
              <div className="w-full h-44 rounded-2xl overflow-hidden bg-obsidian-900 relative">
                <img
                  src={dag.thumb}
                  alt={dag.name}
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.src = 'https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=500&auto=format&fit=crop&q=60';
                  }}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                />
                <div className="absolute top-2.5 left-2.5 bg-obsidian-950/80 backdrop-blur-md px-2.5 py-0.5 rounded-full text-[10px] font-mono text-gold-light border border-gold-border/30">
                  {dag.images} images • {dag.edges} edges
                </div>
              </div>

              {/* Title & Metadata */}
              <div>
                <h3 className="text-base font-bold text-white font-display group-hover:text-gold-light transition-colors">
                  {dag.name}
                </h3>
                <p className="text-xs font-mono text-slate-400 pt-0.5">
                  Transformations: <span className="text-cyan-electric">{dag.transforms}</span>
                </p>
              </div>
            </div>

            {/* Launch Action Button */}
            <button
              onClick={() => handleLaunchDagDemo(dag.id)}
              disabled={runningDag === dag.id}
              className="w-full bg-obsidian-850 hover:bg-gold-gradient hover:text-black text-slate-200 border border-gold-border/30 py-2.5 rounded-xl text-xs font-mono font-semibold flex items-center justify-center space-x-2 transition-all cursor-pointer group/btn"
            >
              <Play className="w-3.5 h-3.5 text-gold-radiant group-hover/btn:text-black group-hover/btn:fill-black" />
              <span>{runningDag === dag.id ? 'Running ViT Inference...' : 'Launch DAG Demo'}</span>
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
