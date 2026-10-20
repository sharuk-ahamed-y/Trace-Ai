import React, { useState } from 'react';
import { 
  Settings, 
  Bell, 
  ShieldCheck, 
  Info,
  Sliders
} from 'lucide-react';

export default function SettingsView({ health, modelInfo }) {
  const [graphAnimation, setGraphAnimation] = useState(true);
  const [autoSave, setAutoSave] = useState(true);
  const [notifications, setNotifications] = useState(false);

  return (
    <div className="max-w-4xl mx-auto space-y-8 pb-12">
      {/* Header matching Reference Showcase */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white font-display">
          Settings
        </h1>
        <p className="text-xs sm:text-sm text-slate-400 font-mono pt-1">
          Customize your TRACE AI experience and inspect engine parameters.
        </p>
      </div>

      <div className="space-y-6">
        {/* Appearance & Behaviour */}
        <div className="glass-panel rounded-3xl p-6 sm:p-8 border border-gold-border/30 space-y-5 shadow-card-dark">
          <h3 className="text-sm font-bold text-white font-display uppercase tracking-wider flex items-center space-x-2">
            <Sliders className="w-4 h-4 text-gold-radiant" />
            <span>Appearance</span>
          </h3>

          <div className="space-y-4">
            {/* Toggle 1: Graph Animation */}
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-medium text-slate-200">Graph Animation</div>
                <div className="text-xs font-mono text-slate-400">Smooth node layout transitions and easing</div>
              </div>
              <button
                type="button"
                onClick={() => setGraphAnimation(!graphAnimation)}
                className={`w-11 h-6 rounded-full transition-colors p-1 cursor-pointer ${
                  graphAnimation ? 'bg-gold-primary' : 'bg-obsidian-800'
                }`}
              >
                <div className={`w-4 h-4 rounded-full bg-white transition-transform ${
                  graphAnimation ? 'translate-x-5' : 'translate-x-0'
                }`} />
              </button>
            </div>

            {/* Toggle 2: Auto Save */}
            <div className="flex items-center justify-between border-t border-gold-border/10 pt-4">
              <div>
                <div className="text-sm font-medium text-slate-200">Auto Save</div>
                <div className="text-xs font-mono text-slate-400">Persist completed dossiers automatically to SQLite</div>
              </div>
              <button
                type="button"
                onClick={() => setAutoSave(!autoSave)}
                className={`w-11 h-6 rounded-full transition-colors p-1 cursor-pointer ${
                  autoSave ? 'bg-gold-primary' : 'bg-obsidian-800'
                }`}
              >
                <div className={`w-4 h-4 rounded-full bg-white transition-transform ${
                  autoSave ? 'translate-x-5' : 'translate-x-0'
                }`} />
              </button>
            </div>

            {/* Toggle 3: Notifications */}
            <div className="flex items-center justify-between border-t border-gold-border/10 pt-4">
              <div>
                <div className="text-sm font-medium text-slate-200">Notifications</div>
                <div className="text-xs font-mono text-slate-400">Display desktop alerts on analysis completion</div>
              </div>
              <button
                type="button"
                onClick={() => setNotifications(!notifications)}
                className={`w-11 h-6 rounded-full transition-colors p-1 cursor-pointer ${
                  notifications ? 'bg-gold-primary' : 'bg-obsidian-800'
                }`}
              >
                <div className={`w-4 h-4 rounded-full bg-white transition-transform ${
                  notifications ? 'translate-x-5' : 'translate-x-0'
                }`} />
              </button>
            </div>
          </div>
        </div>

        {/* Section 4: System Information */}
        <div className="glass-panel rounded-3xl p-6 sm:p-8 border border-gold-border/30 space-y-4 shadow-card-dark">
          <h3 className="text-sm font-bold text-white font-display uppercase tracking-wider flex items-center space-x-2">
            <Info className="w-4 h-4 text-cyan-electric" />
            <span>System Information</span>
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-mono">
            <div className="p-3.5 rounded-xl bg-obsidian-900 border border-gold-border/20 space-y-1">
              <span className="text-slate-400">Version</span>
              <p className="text-white font-bold">1.0.0</p>
            </div>
            <div className="p-3.5 rounded-xl bg-obsidian-900 border border-gold-border/20 space-y-1">
              <span className="text-slate-400">Build</span>
              <p className="text-white font-bold">2026-10-20</p>
            </div>
            <div className="p-3.5 rounded-xl bg-obsidian-900 border border-gold-border/20 space-y-1">
              <span className="text-slate-400">Environment</span>
              <p className="text-emerald-400 font-bold">Production</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
