import React from 'react';
import { 
  LayoutDashboard, 
  UploadCloud, 
  GitCompare, 
  Network, 
  GitFork, 
  FileSearch, 
  FileText, 
  Database, 
  BarChart3,
  History,
  Activity, 
  Zap, 
  ShieldCheck,
  Cpu
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab, health }) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'upload', label: 'Upload & Analyze', icon: UploadCloud },
    { id: 'pairwise', label: 'Image Relationship', icon: GitCompare },
    { id: 'graph', label: 'Provenance Graph', icon: Network },
    { id: 'forest', label: 'Graph & MST Forest', icon: GitFork },
    { id: 'xai', label: 'Explainable AI', icon: FileSearch },
    { id: 'metrics', label: 'Model Metrics', icon: BarChart3 },
    { id: 'history', label: 'Analysis History', icon: History },
    { id: 'report', label: 'Investigation Report', icon: FileText },
    { id: 'dataset', label: 'Benchmark Dataset', icon: Database },
  ];

  return (
    <aside className="w-64 bg-[#070909] border-r border-[#1A2822] flex flex-col justify-between p-4 min-h-screen shrink-0 select-none">
      {/* Brand & Logo */}
      <div className="space-y-6">
        <div className="flex items-center space-x-3 px-2 py-2">
          <div className="w-9 h-9 rounded-xl bg-[#111916] border border-[#39FF88]/40 flex items-center justify-center shadow-[0_0_15px_rgba(57,255,136,0.2)]">
            <Zap className="w-5 h-5 text-[#39FF88] fill-[#39FF88]" />
          </div>
          <div>
            <div className="flex items-center space-x-1.5">
              <span className="font-extrabold text-base tracking-tight text-white font-sans">TRACE</span>
              <span className="font-extrabold text-base tracking-tight text-[#39FF88] font-sans">AI</span>
            </div>
            <div className="text-[10px] text-[#9AA8A0] tracking-wider uppercase font-mono font-medium">Forensic Intelligence</div>
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="space-y-1.5">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-xs font-medium transition-all duration-200 ${
                  isActive
                    ? 'bg-[#D7FF32] text-black font-bold shadow-[0_0_20px_rgba(215,255,50,0.35)] translate-x-1'
                    : 'text-[#9AA8A0] hover:text-white hover:bg-[#111916] hover:translate-x-0.5'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-black stroke-[2.5]' : 'text-[#9AA8A0]'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Engine & Compute Status Widget (Matching Reference Image 1) */}
      <div className="space-y-3 pt-6 border-t border-[#1A2822]">
        <div className="bg-[#111916] p-3.5 rounded-2xl border border-[#1A2822] space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-mono text-[#9AA8A0] font-medium">Model Engine</span>
            <span className={`inline-flex items-center px-1.5 py-0.5 rounded-full text-[9px] font-mono font-bold ${
              health?.status === 'ONLINE' ? 'bg-[#39FF88]/10 text-[#39FF88] border border-[#39FF88]/30' : 'bg-red-500/10 text-red-400'
            }`}>
              ● {health?.status || 'INIT'}
            </span>
          </div>

          <div className="space-y-1">
            <div className="flex justify-between text-[10px] font-mono">
              <span className="text-[#5F7067]">ViT-B/16 Backbone</span>
              <span className="text-[#F4F7F5] font-semibold">{health?.vit_backbone_ready ? '768-D CLS' : 'READY'}</span>
            </div>
            <div className="flex justify-between text-[10px] font-mono">
              <span className="text-[#5F7067]">Device</span>
              <span className="text-[#35D9FF] font-semibold uppercase">{health?.device || 'CPU'}</span>
            </div>
          </div>

          {/* Progress bar */}
          <div className="w-full bg-[#070909] rounded-full h-1.5 overflow-hidden border border-[#1A2822]">
            <div 
              className="bg-gradient-to-r from-[#39FF88] to-[#D7FF32] h-full rounded-full transition-all duration-500" 
              style={{ width: `${health?.memory_usage_percent || 42}%` }}
            />
          </div>
          <div className="flex justify-between items-center text-[9px] font-mono text-[#5F7067]">
            <span>RAM: {health?.memory_usage_percent?.toFixed(1) || '40.0'}%</span>
            <span>CPU: {health?.cpu_usage_percent?.toFixed(1) || '15.0'}%</span>
          </div>
        </div>

        <div className="px-2 text-[10px] text-[#5F7067] font-mono text-center">
          IEEE Provenance Architecture v2.0
        </div>
      </div>
    </aside>
  );
}
