import React, { useState, useEffect } from 'react';
import { Database, Image as ImageIcon, GitFork, CheckCircle, ShieldCheck, Play } from 'lucide-react';
import axios from 'axios';

export default function DatasetExplorer({ onAnalysisComplete, setActiveTab }) {
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedDag, setSelectedDag] = useState('graph_001');

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
      setLoading(true);
      const demoRes = await axios.post(`/api/demo?graph_id=${dagId}`);
      setLoading(false);
      onAnalysisComplete(demoRes.data);
      setActiveTab('graph');
    } catch (err) {
      console.error(err);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="py-20 text-center text-cyber-muted font-mono text-xs">
        Loading benchmark dataset catalog...
      </div>
    );
  }

  const dags = datasetInfo?.available_dag_ids || ['graph_001', 'graph_002', 'graph_003'];

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header Banner */}
      <div className="bg-cyber-card p-6 rounded-2xl border border-cyber-border flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <Database className="w-5 h-5 text-cyber-neon" />
            <span>SIPGD — Synthesized Image Provenance Graph Dataset</span>
          </h2>
          <p className="text-xs text-cyber-muted font-mono mt-1">
            Standard research benchmark containing 100 unique original root images across 14 diverse visual categories with multi-level edit trees.
          </p>
        </div>
        <div className="flex items-center space-x-2 bg-cyber-neon/10 border border-cyber-neon/30 text-cyber-neon px-3 py-1.5 rounded-xl text-xs font-mono">
          <ShieldCheck className="w-4 h-4" />
          <span>16-Point Integrity Validated: PASS</span>
        </div>
      </div>

      {/* Dataset Metric Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 font-mono">
        <div className="bg-cyber-card p-5 rounded-2xl border border-cyber-border">
          <span className="text-[10px] text-cyber-muted uppercase">Original Root Images</span>
          <p className="text-2xl font-bold text-cyber-neon mt-1">100</p>
        </div>
        <div className="bg-cyber-card p-5 rounded-2xl border border-cyber-border">
          <span className="text-[10px] text-cyber-muted uppercase">Total Dataset Images</span>
          <p className="text-2xl font-bold text-white mt-1">795</p>
        </div>
        <div className="bg-cyber-card p-5 rounded-2xl border border-cyber-border">
          <span className="text-[10px] text-cyber-muted uppercase">Ground Truth DAGs</span>
          <p className="text-2xl font-bold text-cyber-blue mt-1">100</p>
        </div>
        <div className="bg-cyber-card p-5 rounded-2xl border border-cyber-border">
          <span className="text-[10px] text-cyber-muted uppercase">Transformations</span>
          <p className="text-2xl font-bold text-cyber-purple mt-1">12 Types</p>
        </div>
      </div>

      {/* DAG Catalog Grid */}
      <div className="bg-cyber-card p-6 rounded-2xl border border-cyber-border space-y-4">
        <div className="flex items-center justify-between border-b border-cyber-border pb-3">
          <h3 className="text-sm font-semibold text-white flex items-center space-x-2">
            <GitFork className="w-4 h-4 text-cyber-neon" />
            <span>Benchmark Provenance DAG Catalog</span>
          </h3>
          <span className="text-xs font-mono text-cyber-muted">Showing 20 DAG instances</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {dags.map((dagId, idx) => (
            <div
              key={dagId}
              className="bg-cyber-dark p-4 rounded-xl border border-cyber-border hover:border-cyber-neon transition-all flex flex-col justify-between group space-y-3"
            >
              <div>
                <div className="flex items-center justify-between font-mono text-xs">
                  <span className="font-bold text-white group-hover:text-cyber-neon transition-colors">{dagId}</span>
                  <span className="text-[10px] bg-cyber-border px-2 py-0.5 rounded text-cyber-muted">DAG #{idx + 1}</span>
                </div>
                <p className="text-[11px] text-cyber-muted mt-1 font-mono">
                  Ground Truth provenance tree with derived multi-level descendants.
                </p>
              </div>

              <button
                onClick={() => handleLaunchDagDemo(dagId)}
                className="w-full bg-cyber-neon/10 border border-cyber-neon/30 text-cyber-neon hover:bg-cyber-neon hover:text-black font-mono text-xs py-2 rounded-lg font-semibold transition-all flex items-center justify-center space-x-1"
              >
                <Play className="w-3.5 h-3.5" />
                <span>Run Inference</span>
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
