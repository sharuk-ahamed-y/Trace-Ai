import React from 'react';
import { Network, Upload, FileSearch, Database, History, Activity, ShieldCheck, Download } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, health }) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Activity },
    { id: 'analysis', label: 'New Analysis', icon: Upload },
    { id: 'graph', label: 'Graph Reconstruction', icon: Network },
    { id: 'xai', label: 'XAI Forensic Evidence', icon: FileSearch },
    { id: 'dataset', label: 'Benchmark Dataset', icon: Database },
    { id: 'history', label: 'Analysis History', icon: History },
    { id: 'health', label: 'System Health', icon: ShieldCheck },
  ];

  return (
    <header className="bg-cyber-card border-b border-cyber-border sticky top-0 z-50 px-6 py-3">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('dashboard')}>
          <div className="w-9 h-9 rounded-lg bg-cyber-neon/10 border border-cyber-neon flex items-center justify-center text-cyber-neon shadow-[0_0_10px_rgba(0,255,136,0.3)]">
            <Network className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-lg tracking-wider text-white">TRACE<span className="text-cyber-neon">AI</span></span>
              <span className="text-[10px] uppercase tracking-widest bg-cyber-neon/20 text-cyber-neon px-2 py-0.5 rounded border border-cyber-neon/40 font-mono">LAB v1.0</span>
            </div>
            <p className="text-[11px] text-cyber-muted font-mono">Image Provenance & MST Graph Analysis</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center space-x-1 bg-cyber-dark/80 p-1 rounded-xl border border-cyber-border">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                  isActive
                    ? 'bg-cyber-neon text-black font-semibold shadow-[0_0_12px_rgba(0,255,136,0.4)]'
                    : 'text-cyber-muted hover:text-slate-200 hover:bg-cyber-border/40'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* System Status Pill */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2 text-xs font-mono bg-cyber-dark px-3 py-1.5 rounded-lg border border-cyber-border">
            <span className={`w-2 h-2 rounded-full ${health?.status === 'ONLINE' ? 'bg-cyber-neon animate-pulse' : 'bg-red-500'}`}></span>
            <span className="text-cyber-muted">BACKEND:</span>
            <span className={health?.status === 'ONLINE' ? 'text-cyber-neon font-bold' : 'text-red-400'}>{health?.status || 'CONNECTING...'}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
