import React, { useState, useEffect } from 'react';
import { History, Download, ExternalLink, RefreshCw, FolderSearch } from 'lucide-react';
import axios from 'axios';

export default function HistoryView({ onAnalysisComplete, setActiveTab }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState(null);

  const fetchHistory = () => {
    setLoading(true);
    setErrorMsg(null);
    axios.get('/api/history')
      .then(res => {
        setHistory(res.data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setErrorMsg("Failed to query SQLite history database.");
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleOpenAnalysis = async (analysisId) => {
    try {
      const res = await axios.get(`/api/analysis/${analysisId}`);
      const data = res.data;
      const formatted = {
        analysis_id: data.analysis_id || data.analysis?.id || analysisId,
        status: data.status || data.analysis?.status || 'COMPLETED',
        root_image_id: data.root_image_id || data.analysis?.root_image_id,
        root_confidence: data.root_confidence !== undefined ? data.root_confidence : data.analysis?.root_confidence,
        execution_time_ms: data.execution_time_ms !== undefined ? data.execution_time_ms : data.analysis?.execution_time_ms,
        similarity_threshold: data.similarity_threshold || data.analysis?.similarity_threshold,
        confidence_filter: data.confidence_filter || data.analysis?.confidence_filter,
        nodes: data.nodes || data.images || [],
        images: data.images || data.nodes || [],
        edges: data.edges || [],
        xai_results: data.xai_results || []
      };
      onAnalysisComplete(formatted);
      setActiveTab('graph');
    } catch (err) {
      console.error("Open analysis error:", err);
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-[#F4F7F5] tracking-tight flex items-center space-x-2 font-sans">
            <History className="w-6 h-6 text-[#39FF88]" />
            <span>Analysis History & Saved Investigations</span>
          </h1>
          <p className="text-xs text-[#9AA8A0] font-mono pt-1">
            Persisted analysis records stored in SQLite database (<code className="text-[#39FF88]">provenance_app.db</code>).
          </p>
        </div>

        <button
          onClick={fetchHistory}
          className="bg-[#111916] hover:bg-[#1A2822] text-[#39FF88] hover:text-white px-4 py-2 rounded-xl border border-[#39FF88]/30 transition-all text-xs font-mono flex items-center space-x-1.5 cursor-pointer shadow-[0_0_12px_rgba(57,255,136,0.1)]"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh Database</span>
        </button>
      </div>

      {errorMsg && (
        <div className="bg-red-500/10 border border-red-500/30 p-4 rounded-2xl text-red-400 font-mono text-xs">
          {errorMsg}
        </div>
      )}

      {/* Main Table Card */}
      <div className="bg-[#111916] rounded-3xl border border-[#1A2822] overflow-hidden">
        {loading ? (
          <div className="py-24 text-center text-[#9AA8A0] font-mono text-xs flex flex-col items-center space-y-2">
            <RefreshCw className="w-6 h-6 text-[#5F7067] animate-spin" />
            <span>Querying provenance history table...</span>
          </div>
        ) : history.length === 0 ? (
          <div className="py-24 text-center text-[#9AA8A0] font-mono text-xs flex flex-col items-center space-y-3">
            <FolderSearch className="w-8 h-8 text-[#5F7067] opacity-60" />
            <p>No analysis sessions saved yet in SQLite database.</p>
            <p className="text-[11px] text-[#5F7067]">Run your first provenance reconstruction from the Upload Zone or launch a benchmark demo.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead className="bg-[#070909] text-[#9AA8A0] uppercase text-[10px] border-b border-[#1A2822]">
                <tr>
                  <th className="p-4">Analysis ID</th>
                  <th className="p-4">Timestamp</th>
                  <th className="p-4">Images</th>
                  <th className="p-4">Root Image ID</th>
                  <th className="p-4">Execution Time</th>
                  <th className="p-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1A2822] text-[#F4F7F5]">
                {history.map((row) => (
                  <tr key={row.id} className="hover:bg-[#1A2822]/40 transition-colors">
                    <td className="p-4 font-bold text-[#39FF88]">{row.id}</td>
                    <td className="p-4 text-[#9AA8A0]">{row.timestamp ? row.timestamp.replace('T', ' ').substring(0, 19) : 'N/A'}</td>
                    <td className="p-4">{row.image_count} images</td>
                    <td className="p-4 text-[#35D9FF] font-bold">{row.root_image_id || 'N/A'}</td>
                    <td className="p-4">{row.execution_time_ms ? `${row.execution_time_ms.toFixed(0)} ms` : 'N/A'}</td>
                    <td className="p-4 text-right space-x-2">
                      <button
                        onClick={() => handleOpenAnalysis(row.id)}
                        className="bg-[#39FF88]/10 border border-[#39FF88]/30 text-[#39FF88] hover:bg-[#39FF88] hover:text-black px-3 py-1.5 rounded-lg text-xs font-semibold transition-all inline-flex items-center space-x-1 cursor-pointer"
                      >
                        <ExternalLink className="w-3.5 h-3.5" />
                        <span>Open Graph</span>
                      </button>
                      <a
                        href={`/api/download/${row.id}/html`}
                        target="_blank"
                        rel="noreferrer"
                        className="bg-[#070909] border border-[#1A2822] text-[#9AA8A0] hover:text-white px-3 py-1.5 rounded-lg text-xs transition-all inline-flex items-center space-x-1"
                      >
                        <Download className="w-3.5 h-3.5" />
                        <span>Report</span>
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
