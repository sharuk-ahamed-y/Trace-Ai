import React, { useState } from 'react';
import { 
  Home, 
  UploadCloud, 
  Network, 
  BrainCircuit, 
  Settings, 
  FileSearch, 
  History, 
  GitCompare, 
  GitFork, 
  FileText, 
  Compass, 
  ChevronDown, 
  ChevronRight,
  ShieldCheck,
  Cpu,
  Layers
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab, health }) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Primary navigation matching the core platform workflow
  const primaryNav = [
    { id: 'dashboard', label: 'Home', icon: Home },
    { id: 'upload', label: 'Upload', icon: UploadCloud },
    { id: 'analysis', label: 'Analysis', icon: FileSearch },
    { id: 'xai', label: 'Explainable AI', icon: BrainCircuit },
    { id: 'graph', label: 'Graph', icon: Network },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  // Secondary forensic intelligence modules
  const secondaryNav = [
    { id: 'history', label: 'History Dossier', icon: History },
    { id: 'pairwise', label: 'Pairwise Classifier', icon: GitCompare },
    { id: 'forest', label: 'Forest Comparison', icon: GitFork },
    { id: 'report', label: 'Investigation Report', icon: FileText },
  ];

  return (
    <aside className="w-64 bg-[#05070B] border-r border-gold-border/40 flex flex-col justify-between p-4 min-h-screen shrink-0 select-none z-30 relative shadow-[4px_0_30px_rgba(0,0,0,0.5)]">
      {/* Brand & Logo matching reference */}
      <div className="space-y-6">
        <div 
          onClick={() => setActiveTab('dashboard')} 
          className="flex items-center space-x-3.5 px-2 py-2 cursor-pointer group"
        >
          {/* Golden Gyroscope / Compass Icon Container */}
          <div className="relative w-10 h-10 rounded-xl bg-obsidian-800 border border-gold-primary/50 flex items-center justify-center shadow-[0_0_20px_rgba(212,175,55,0.25)] group-hover:shadow-[0_0_28px_rgba(229,193,88,0.45)] group-hover:border-gold-radiant transition-all">
            <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-gold-radiant/20 to-transparent pointer-events-none" />
            <Compass className="w-5 h-5 text-gold-radiant stroke-[2.2] animate-pulse-gold" />
          </div>
          <div>
            <div className="flex items-center space-x-1.5 leading-none">
              <span className="font-extrabold text-lg tracking-tight text-white font-sans">TRACE</span>
              <span className="font-extrabold text-lg tracking-tight text-gold-radiant font-sans">AI</span>
            </div>
            <div className="text-[10px] text-slate-400 tracking-wider uppercase font-mono font-medium pt-1">
              Forensic Image Provenance
            </div>
          </div>
        </div>

        {/* Primary Navigation Links */}
        <nav className="space-y-1.5 pt-2">
          {primaryNav.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center space-x-3.5 px-4 py-2.5 rounded-xl text-xs font-medium transition-all duration-200 cursor-pointer ${
                  isActive
                    ? 'bg-gradient-to-r from-gold-radiant/25 via-gold-primary/15 to-transparent text-white font-semibold border-l-2 border-gold-radiant shadow-[0_0_20px_rgba(212,175,55,0.18)] translate-x-1'
                    : 'text-slate-400 hover:text-white hover:bg-obsidian-800/60 hover:translate-x-0.5'
                }`}
              >
                <Icon className={`w-4 h-4 transition-colors ${isActive ? 'text-gold-radiant stroke-[2.3]' : 'text-slate-400'}`} />
                <span className="tracking-wide">{item.label}</span>
                {isActive && (
                  <span className="ml-auto w-1.5 h-1.5 rounded-full bg-gold-radiant shadow-[0_0_8px_#D4AF37]" />
                )}
              </button>
            );
          })}

          {/* Collapsible Forensic Toolset Dropdown */}
          <div className="pt-3">
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="w-full flex items-center justify-between px-3 py-1.5 text-[11px] font-mono text-slate-400 hover:text-gold-light tracking-wider uppercase transition-colors"
            >
              <span>Forensic Toolset</span>
              {showAdvanced ? (
                <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
              ) : (
                <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
              )}
            </button>

            {showAdvanced && (
              <div className="space-y-1 pt-1.5 pl-2 border-l border-gold-border/20 ml-3">
                {secondaryNav.map((item) => {
                  const Icon = item.icon;
                  const isActive = activeTab === item.id;
                  return (
                    <button
                      key={item.id}
                      onClick={() => setActiveTab(item.id)}
                      className={`w-full flex items-center space-x-2.5 px-3 py-2 rounded-lg text-[11px] font-medium transition-all duration-150 cursor-pointer ${
                        isActive
                          ? 'bg-gold-primary/15 text-gold-light font-semibold border-l-2 border-gold-primary'
                          : 'text-slate-400 hover:text-slate-200 hover:bg-obsidian-800/40'
                      }`}
                    >
                      <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-gold-radiant' : 'text-slate-400'}`} />
                      <span>{item.label}</span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </nav>
      </div>

      {/* System Status Widget matching reference bottom-left */}
      <div className="pt-4 border-t border-gold-border/30 space-y-3">
        <div className="glass-card p-3 rounded-xl border border-gold-border/30 space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="relative flex h-2 w-2">
                <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                  health?.status === 'ONLINE' ? 'bg-cyan-electric' : 'bg-red-400'
                }`} />
                <span className={`relative inline-flex rounded-full h-2 w-2 ${
                  health?.status === 'ONLINE' ? 'bg-cyan-electric' : 'bg-red-500'
                }`} />
              </span>
              <span className="text-[11px] font-mono text-white font-medium">System Online</span>
            </div>
            <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-cyan-electric/10 text-cyan-electric border border-cyan-electric/30">
              {health?.device || 'CPU'}
            </span>
          </div>

          <div className="space-y-0.5 text-[10px] font-mono text-slate-400">
            <div className="flex justify-between">
              <span>Backend:</span>
              <span className="text-slate-300">127.0.0.1:8000</span>
            </div>
            <div className="flex justify-between">
              <span>Frontend:</span>
              <span className="text-slate-300">127.0.0.1:5173</span>
            </div>
          </div>
        </div>

        <div className="text-center text-[9px] font-mono text-slate-400 uppercase tracking-widest">
          TRACE AI • v1.0.0 Production
        </div>
      </div>
    </aside>
  );
}
