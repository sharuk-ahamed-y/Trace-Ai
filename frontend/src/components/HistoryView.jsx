import React, { useState, useEffect } from 'react';
import { History, Download, ExternalLink, RefreshCw } from 'lucide-react';
import axios from 'axios';

export default function HistoryView({ onAnalysisComplete, setActiveTab }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchHistory = () => {
    setLoading(true);
    axios.get('/api/history')
      .then(res => {
        setHistory(res.data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
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
        analysis_id: data.analysis.id,
        status: data.analysis.status,
        root_image_id: data.analysis.root_image_id,
        root_confidence: data.analysis.root_confidence,
        execution_time_ms: data.analysis.execution_time_ms,
        nodes: data.images,
        edges: data.edges,
        xai_results: data.xai_results
      };
      onAnalysisComplete(formatted);
      setActiveTab('graph');
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="bg-cyber-card p-6 rounded-2xl border border-cyber-border flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <History className="w-5 h-5 text-cyber-neon" />
            <span>Analysis History & Saved Sessions</span>
          </h2>
          <p className="text-xs text-cyber-muted font-mono mt-1">
            Persisted analysis records stored in local SQLite database (<code className="text-cyber-neon">provenance_app.db</code>).
          </p>
        </div>
        <button
          onClick={fetchHistory}
          className="flex items-center space-x-2 bg-cyber-dark hover:bg-cyber-border text-cyber-muted hover:text-white px-3 py-2 rounded-xl border border-cyber-border transition-all text-xs font-mono"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Refresh</span>
        </button>
      </div>

      <div className="bg-cyber-card rounded-2xl border border-cyber-border overflow-hidden">
        {loading ? (
          <div className="py-16 text-center text-cyber-muted font-mono text-xs">
            Loading history records...
          </div>
        ) : history.length === 0 ? (
          <div className="py-16 text-center text-cyber-muted font-mono text-xs">
            No analysis sessions saved yet. Run your first graph analysis from the Upload Zone.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead className="bg-cyber-dark text-cyber-muted uppercase text-[10px] border-b border-cyber-border">
                <tr>
                  <th className="p-4">Analysis ID</th>
                  <th className="p-4">Timestamp</th>
                  <th className="p-4">Images</th>
                  <th className="p-4">Root Image ID</th>
                  <th className="p-4">Execution Time</th>
                  <th className="p-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-cyber-border text-slate-200">
                {history.map((row) => (
                  <tr key={row.id} className="hover:bg-cyber-dark/40 transition-colors">
                    <td className="p-4 font-bold text-cyber-neon">{row.id}</td>
                    <td className="p-4 text-cyber-muted">{row.timestamp.replace('T', ' ').substring(0, 19)}</td>
                    <td className="p-4">{row.image_count} images</td>
                    <td className="p-4 text-cyber-blue font-bold">{row.root_image_id || 'N/A'}</td>
                    <td className="p-4">{row.execution_time_ms ? `${row.execution_time_ms.toFixed(0)} ms` : 'N/A'}</td>
                    <td className="p-4 text-right space-x-2">
                      <button
                        onClick={() => handleOpenAnalysis(row.id)}
                        className="bg-cyber-neon/10 border border-cyber-neon/30 text-cyber-neon hover:bg-cyber-neon hover:text-black px-3 py-1.5 rounded-lg text-xs font-semibold transition-all inline-flex items-center space-x-1"
                      >
                        <ExternalLink className="w-3.5 h-3.5" />
                        <span>Open Graph</span>
                      </button>
                      <a
                        href={`/api/download/${row.id}/html`}
                        target="_blank"
                        rel="noreferrer"
                        className="bg-cyber-dark border border-cyber-border text-cyber-muted hover:text-white px-3 py-1.5 rounded-lg text-xs transition-all inline-flex items-center space-x-1"
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
