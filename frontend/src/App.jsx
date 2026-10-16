import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Sidebar from './components/Sidebar';
import DashboardView from './components/DashboardView';
import UploadZone from './components/UploadZone';
import PairwiseAnalysisView from './components/PairwiseAnalysisView';
import ProvenanceGraphViewer from './components/ProvenanceGraphViewer';
import ForestAnalysisView from './components/ForestAnalysisView';
import XaiPanel from './components/XaiPanel';
import InvestigationReportView from './components/InvestigationReportView';
import DatasetExplorer from './components/DatasetExplorer';
import ModelMetricsView from './components/ModelMetricsView';
import HistoryView from './components/HistoryView';
import { 
  Sparkles, 
  ShieldCheck, 
  AlertCircle, 
  Terminal, 
  Layers, 
  Zap, 
  CheckCircle2,
  Cpu
} from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTabState] = useState(() => {
    try {
      return localStorage.getItem('trace_ai_active_tab') || 'dashboard';
    } catch {
      return 'dashboard';
    }
  });
  const [analysisData, setAnalysisData] = useState(null);
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [history, setHistory] = useState([]);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [globalLoading, setGlobalLoading] = useState(false);
  const [statusNotice, setStatusNotice] = useState(null);

  const setActiveTab = (tab) => {
    setActiveTabState(tab);
    try {
      localStorage.setItem('trace_ai_active_tab', tab);
    } catch {}
  };

  // Restore active analysis on page refresh
  useEffect(() => {
    try {
      const savedId = localStorage.getItem('trace_ai_active_analysis_id');
      if (savedId) {
        axios.get(`/api/analysis/${savedId}`)
          .then(res => {
            if (res.data && res.data.analysis_id) {
              setAnalysisData(res.data);
            }
          })
          .catch(err => {
            console.warn("Could not restore saved analysis session:", err);
          });
      }
    } catch (e) {
      console.warn("LocalStorage access error:", e);
    }
  }, []);

  // Poll system health and load model metadata
  useEffect(() => {
    const fetchHealthAndInfo = () => {
      axios.get('/api/health')
        .then(res => setHealth(res.data))
        .catch(() => setHealth({ status: 'OFFLINE' }));

      axios.get('/api/model-info')
        .then(res => setModelInfo(res.data))
        .catch(err => console.error("Model info fetch failed:", err));

      axios.get('/api/history')
        .then(res => setHistory(res.data || []))
        .catch(err => console.error("History fetch failed:", err));

      axios.get('/api/dataset')
        .then(res => setDatasetInfo(res.data))
        .catch(err => console.error("Dataset info fetch failed:", err));
    };

    fetchHealthAndInfo();
    const interval = setInterval(fetchHealthAndInfo, 12000);
    return () => clearInterval(interval);
  }, []);

  const handleAnalysisComplete = (data) => {
    setAnalysisData(data);
    try {
      if (data?.analysis_id) {
        localStorage.setItem('trace_ai_active_analysis_id', data.analysis_id);
      }
    } catch {}
    // Refresh history
    axios.get('/api/history')
      .then(res => setHistory(res.data || []))
      .catch(() => {});
  };

  const handleRunDemo = async () => {
    try {
      setGlobalLoading(true);
      setStatusNotice("Initializing ViT inference over benchmark DAG graph_0001...");
      const res = await axios.post('/api/demo?graph_id=graph_0001');
      setAnalysisData(res.data);
      try {
        if (res.data?.analysis_id) {
          localStorage.setItem('trace_ai_active_analysis_id', res.data.analysis_id);
        }
      } catch {}
      setGlobalLoading(false);
      setStatusNotice(null);
      setActiveTab('graph');
    } catch (err) {
      console.error("Demo run error:", err);
      setGlobalLoading(false);
      setStatusNotice("Failed to launch benchmark demo.");
    }
  };

  return (
    <div className="min-h-screen bg-[#070909] text-[#F4F7F5] flex font-sans antialiased selection:bg-[#39FF88] selection:text-black">
      {/* Sleek Vertical Navigation Sidebar (Matching Reference Image 1) */}
      <Sidebar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        health={health} 
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 h-screen overflow-y-auto">
        {/* Sticky Top Header Bar */}
        <header className="sticky top-0 z-40 bg-[#070909]/90 backdrop-blur-md border-b border-[#1A2822] px-8 py-3.5 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <span className="text-[11px] font-mono text-[#5F7067] uppercase tracking-wider">
              WORKSPACE
            </span>
            <span className="text-[#5F7067] font-mono text-xs">/</span>
            <span className="text-xs font-mono font-bold text-white capitalize">
              {activeTab === 'dashboard' && 'Forensic Intelligence Dashboard'}
              {activeTab === 'upload' && 'Multi-Image Ingestion & Reconstruction'}
              {activeTab === 'pairwise' && 'Image Relationship & Transformation Analysis'}
              {activeTab === 'graph' && 'Interactive Provenance Graph Canvas'}
              {activeTab === 'forest' && 'Candidate Graph vs. Minimum Spanning Forest'}
              {activeTab === 'xai' && 'Explainable AI Forensic Evidence'}
              {activeTab === 'metrics' && 'Model Performance Metrics & Evaluation'}
              {activeTab === 'history' && 'Analysis History & Saved Investigations'}
              {activeTab === 'report' && 'Forensic Investigation Dossier'}
              {activeTab === 'dataset' && 'Benchmark Dataset Catalog'}
            </span>

            {analysisData && (
              <span className="bg-[#111916] border border-[#39FF88]/30 text-[#39FF88] text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full flex items-center space-x-1 shadow-[0_0_10px_rgba(57,255,136,0.15)]">
                <span className="w-1.5 h-1.5 rounded-full bg-[#39FF88] animate-pulse" />
                <span>ACTIVE: {analysisData.analysis_id}</span>
              </span>
            )}
          </div>

          {/* Quick Actions & System Pills */}
          <div className="flex items-center space-x-3">
            <button
              onClick={handleRunDemo}
              disabled={globalLoading}
              className="bg-[#111916] hover:bg-[#1A2822] text-[#39FF88] hover:text-white border border-[#39FF88]/30 px-3.5 py-1.5 rounded-full text-xs font-mono font-semibold flex items-center space-x-1.5 shadow-[0_0_12px_rgba(57,255,136,0.15)] transition-all cursor-pointer disabled:opacity-50"
            >
              <Sparkles className="w-3.5 h-3.5 text-[#39FF88]" />
              <span>{globalLoading ? 'Inference Running...' : 'Instant Benchmark Demo'}</span>
            </button>

            <div className="hidden sm:flex items-center space-x-2 bg-[#111916] border border-[#1A2822] px-3 py-1 rounded-full text-[11px] font-mono">
              <span className={`w-2 h-2 rounded-full ${health?.status === 'ONLINE' ? 'bg-[#39FF88]' : 'bg-red-500'}`} />
              <span className="text-[#9AA8A0]">BACKEND:</span>
              <span className={health?.status === 'ONLINE' ? 'text-[#39FF88] font-bold' : 'text-red-400'}>
                {health?.status || 'INIT'}
              </span>
            </div>
          </div>
        </header>

        {/* Global Banner Notification (if any) */}
        {statusNotice && (
          <div className="bg-[#111916] border-b border-[#39FF88]/30 px-8 py-2 text-xs font-mono text-[#39FF88] flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-[#39FF88] animate-ping" />
            <span>{statusNotice}</span>
          </div>
        )}

        {/* Main Routed Page Container */}
        <main className="p-8 max-w-[1600px] w-full mx-auto flex-1">
          {activeTab === 'dashboard' && (
            <DashboardView
              onRunDemo={handleRunDemo}
              setActiveTab={setActiveTab}
              history={history}
              health={health}
              modelInfo={modelInfo}
              datasetInfo={datasetInfo}
              analysisData={analysisData}
            />
          )}

          {activeTab === 'upload' && (
            <UploadZone
              onAnalysisComplete={handleAnalysisComplete}
              setActiveTab={setActiveTab}
            />
          )}

          {activeTab === 'pairwise' && (
            <PairwiseAnalysisView
              initialImages={analysisData?.nodes}
            />
          )}

          {activeTab === 'graph' && (
            <ProvenanceGraphViewer
              analysisData={analysisData}
              setActiveTab={setActiveTab}
            />
          )}

          {activeTab === 'forest' && (
            <ForestAnalysisView
              analysisData={analysisData}
              setActiveTab={setActiveTab}
            />
          )}

          {activeTab === 'xai' && (
            <XaiPanel
              analysisData={analysisData}
              setActiveTab={setActiveTab}
            />
          )}

          {activeTab === 'metrics' && (
            <ModelMetricsView
              modelInfo={modelInfo}
              health={health}
            />
          )}

          {activeTab === 'history' && (
            <HistoryView
              onAnalysisComplete={handleAnalysisComplete}
              setActiveTab={setActiveTab}
            />
          )}

          {activeTab === 'report' && (
            <InvestigationReportView
              analysisData={analysisData}
              setActiveTab={setActiveTab}
            />
          )}

          {activeTab === 'dataset' && (
            <DatasetExplorer
              onAnalysisComplete={handleAnalysisComplete}
              setActiveTab={setActiveTab}
            />
          )}
        </main>
      </div>
    </div>
  );
}
