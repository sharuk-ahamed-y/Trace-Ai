import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Sidebar from './components/Sidebar';
import DashboardView from './components/DashboardView';
import UploadZone from './components/UploadZone';
import AnalysisResultsView from './components/AnalysisResultsView';
import ProvenanceGraphViewer from './components/ProvenanceGraphViewer';
import ImageDetailsView from './components/ImageDetailsView';
import SettingsView from './components/SettingsView';
import HistoryView from './components/HistoryView';
import PairwiseAnalysisView from './components/PairwiseAnalysisView';
import ForestAnalysisView from './components/ForestAnalysisView';
import XaiPanel from './components/XaiPanel';
import ModelMetricsView from './components/ModelMetricsView';
import InvestigationReportView from './components/InvestigationReportView';
import ErrorBoundary from './components/ErrorBoundary';
import { 
  Sparkles, 
  Play, 
  ShieldCheck, 
  Cpu, 
  AlertCircle 
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

  const setActiveTab = (tab) => {
    setActiveTabState(tab);
    try {
      localStorage.setItem('trace_ai_active_tab', tab);
    } catch {}
  };

  // Restore saved analysis from localStorage on initial load
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
      console.warn("LocalStorage error:", e);
    }
  }, []);

  // Poll system health and metadata
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
    // Refresh history records
    axios.get('/api/history')
      .then(res => setHistory(res.data || []))
      .catch(() => {});
  };

  const handleRunDemo = async () => {
    try {
      setGlobalLoading(true);
      const res = await axios.post('/api/demo?graph_id=graph_0001');
      setAnalysisData(res.data);
      try {
        if (res.data?.analysis_id) {
          localStorage.setItem('trace_ai_active_analysis_id', res.data.analysis_id);
        }
      } catch {}
      setGlobalLoading(false);
      setActiveTab('analysis');
    } catch (err) {
      console.error("Demo run error:", err);
      setGlobalLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#05070B] text-slate-100 flex font-sans antialiased selection:bg-gold-primary selection:text-black">
      {/* Sleek Vertical Navigation Sidebar (Matching Reference Showcase) */}
      <Sidebar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        health={health} 
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 h-screen overflow-y-auto">
        {/* Top Header Bar */}
        <header className="sticky top-0 z-40 bg-[#05070B]/90 backdrop-blur-xl border-b border-gold-border/30 px-8 py-3.5 flex items-center justify-between shadow-[0_4px_30px_rgba(0,0,0,0.5)]">
          {/* Platform Breadcrumb & Current Context Title */}
          <div className="flex items-center space-x-3">
            <span className="text-[11px] font-mono tracking-widest uppercase text-gold-light/60 font-semibold">
              Trace AI Forensic Laboratory
            </span>
            <span className="text-slate-600">/</span>
            <span className="text-xs sm:text-sm font-semibold text-white tracking-wide capitalize">
              {activeTab === 'xai' ? 'Explainable AI' : activeTab === 'report' ? 'Investigation Report' : activeTab}
            </span>
          </div>

          {/* Right Header Status: Real Engine Health */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 bg-obsidian-850/80 border border-gold-border/25 px-3 py-1.5 rounded-full shadow-card-dark">
              <span className="relative flex h-2 w-2">
                <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                  health?.status === 'ONLINE' ? 'bg-cyan-electric' : 'bg-amber-400'
                }`} />
                <span className={`relative inline-flex rounded-full h-2 w-2 ${
                  health?.status === 'ONLINE' ? 'bg-cyan-electric' : 'bg-amber-400'
                }`} />
              </span>
              <span className="text-[11px] font-mono font-medium text-slate-300">
                {health?.status === 'ONLINE' ? 'Engine Online' : 'Connecting Engine...'}
              </span>
            </div>
          </div>
        </header>

        {/* Main Routed Page Container */}
        <main className="p-6 md:p-8 max-w-[1600px] w-full mx-auto flex-1">
          <ErrorBoundary>
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

            {activeTab === 'analysis' && (
              <AnalysisResultsView
                analysisData={analysisData}
                setActiveTab={setActiveTab}
              />
            )}

            {activeTab === 'graph' && (
              <ProvenanceGraphViewer
                analysisData={analysisData}
                setActiveTab={setActiveTab}
              />
            )}

            {activeTab === 'settings' && (
              <SettingsView
                health={health}
                modelInfo={modelInfo}
              />
            )}

            {/* Secondary Forensic Screens */}
            {activeTab === 'history' && (
              <HistoryView
                onAnalysisComplete={handleAnalysisComplete}
                setActiveTab={setActiveTab}
              />
            )}

            {activeTab === 'pairwise' && (
              <PairwiseAnalysisView
                initialImages={analysisData?.nodes}
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

            {activeTab === 'report' && (
              <InvestigationReportView
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
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
}
