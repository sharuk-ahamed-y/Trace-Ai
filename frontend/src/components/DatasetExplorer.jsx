import React, { useState, useEffect } from 'react';
import { Database, Image as ImageIcon, GitFork, CheckCircle, ShieldCheck, Play, ArrowRight, Sparkles } from 'lucide-react';
import axios from 'axios';

export default function DatasetExplorer({ onAnalysisComplete, setActiveTab }) {
  const [datasetInfo, setDatasetInfo] = useState(null);
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
      setActiveTab('graph');
    } catch (err) {
      console.error(err);
      setRunningDag(null);
    }
  };

  if (loading) {
    return (
      <div className="py-24 text-center text-[#9AA8A0] font-mono text-xs">
        Loading benchmark dataset catalog from image_provenance_dataset...
      </div>
    );
  }

  const dags = datasetInfo?.available_dag_ids || [
    'graph_0001', 'graph_0002', 'graph_0003', 'graph_0004', 'graph_0005',
    'graph_0006', 'graph_0007', 'graph_0008', 'graph_0009', 'graph_0010'
  ];

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-[#F4F7F5] tracking-tight flex items-center space-x-2 font-sans">
            <Database className="w-6 h-6 text-[#39FF88]" />
            <span>Benchmark Dataset Explorer (SIPGD)</span>
          </h1>
          <p className="text-xs text-[#9AA8A0] font-mono pt-1">
            Synthesized Image Provenance Graph Dataset with 100 DAGs, 795 benchmark images, and 12 transformation classes.
          </p>
        </div>

        <div className="flex items-center space-x-2 bg-[#111916] border border-[#39FF88]/30 text-[#39FF88] px-3.5 py-1.5 rounded-full text-xs font-mono font-bold">
          <ShieldCheck className="w-4 h-4 text-[#39FF88]" />
          <span>795 Benchmark Images Loaded</span>
        </div>
      </div>

      {/* Dataset Metric KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 font-mono">
        <div className="bg-[#111916] p-5 rounded-2xl border border-[#1A2822]">
          <span className="text-[10px] text-[#9AA8A0] uppercase tracking-wider block">Root Originals</span>
          <p className="text-2xl font-bold text-[#39FF88] mt-1">{datasetInfo?.total_provenance_dags || 100}</p>
          <span className="text-[10px] text-[#5F7067] mt-0.5 block">Unique Root Images</span>
        </div>

        <div className="bg-[#111916] p-5 rounded-2xl border border-[#1A2822]">
          <span className="text-[10px] text-[#9AA8A0] uppercase tracking-wider block">Total Test/Train/Val Images</span>
          <p className="text-2xl font-bold text-white mt-1">{datasetInfo?.total_benchmark_images || 795}</p>
          <span className="text-[10px] text-[#5F7067] mt-0.5 block">Forensic Variants</span>
        </div>

        <div className="bg-[#111916] p-5 rounded-2xl border border-[#1A2822]">
          <span className="text-[10px] text-[#9AA8A0] uppercase tracking-wider block">Ground Truth DAGs</span>
          <p className="text-2xl font-bold text-[#35D9FF] mt-1">{datasetInfo?.total_provenance_dags || 100}</p>
          <span className="text-[10px] text-[#5F7067] mt-0.5 block">Verified Lineage Graphs</span>
        </div>

        <div className="bg-[#111916] p-5 rounded-2xl border border-[#1A2822]">
          <span className="text-[10px] text-[#9AA8A0] uppercase tracking-wider block">Transformation Operations</span>
          <p className="text-2xl font-bold text-[#D7FF32] mt-1">12 Classes</p>
          <span className="text-[10px] text-[#5F7067] mt-0.5 block">Crop, Blur, Rescale, etc.</span>
        </div>
      </div>

      {/* DAG Catalog Grid */}
      <div className="bg-[#111916] p-6 rounded-3xl border border-[#1A2822] space-y-5">
        <div className="flex items-center justify-between border-b border-[#1A2822] pb-3">
          <h3 className="text-sm font-bold text-white font-sans flex items-center space-x-2">
            <GitFork className="w-4 h-4 text-[#39FF88]" />
            <span>Benchmark Provenance DAG Instances ({dags.length})</span>
          </h3>
          <span className="text-xs font-mono text-[#9AA8A0]">Ready for instant model evaluation</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {dags.map((dagId, idx) => (
            <div
              key={dagId}
              className="bg-[#070909] p-4 rounded-2xl border border-[#1A2822] hover:border-[#39FF88]/50 transition-all flex flex-col justify-between group space-y-3.5"
            >
              <div>
                <div className="flex items-center justify-between font-mono text-xs">
                  <span className="font-bold text-white group-hover:text-[#39FF88] transition-colors">{dagId}</span>
                  <span className="text-[10px] bg-[#111916] px-2 py-0.5 rounded border border-[#1A2822] text-[#9AA8A0]">
                    DAG #{idx + 1}
                  </span>
                </div>
                <p className="text-[11px] text-[#9AA8A0] mt-1.5 font-mono leading-relaxed">
                  Ground truth multi-generation lineage tree derived from benchmark collection.
                </p>
              </div>

              <button
                onClick={() => handleLaunchDagDemo(dagId)}
                disabled={runningDag === dagId}
                className="w-full bg-[#111916] hover:bg-[#39FF88] text-[#39FF88] hover:text-black border border-[#39FF88]/30 font-mono text-xs py-2 rounded-xl font-bold transition-all flex items-center justify-center space-x-1.5 cursor-pointer shadow-[0_0_12px_rgba(57,255,136,0.1)] disabled:opacity-50"
              >
                {runningDag === dagId ? (
                  <>
                    <div className="w-3 h-3 border-2 border-black border-t-transparent rounded-full animate-spin" />
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5" />
                    <span>Run Inference</span>
                  </>
                )}
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
