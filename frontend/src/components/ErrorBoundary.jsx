import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-8 max-w-2xl mx-auto my-12 glass-panel rounded-3xl border border-gold-border/40 text-center space-y-4">
          <div className="w-12 h-12 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-gold-radiant flex items-center justify-center mx-auto">
            <AlertTriangle className="w-6 h-6 stroke-[2]" />
          </div>
          <h2 className="text-xl font-bold text-white font-display">Module Render Interrupted</h2>
          <p className="text-xs font-mono text-slate-300 break-words">
            {this.state.error?.message || "An unexpected error occurred in this view."}
          </p>
          <button
            onClick={() => {
              this.setState({ hasError: false, error: null });
              window.location.reload();
            }}
            className="bg-gold-gradient text-black font-bold px-6 py-2 rounded-full text-xs font-mono inline-flex items-center space-x-2 shadow-gold-glow cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Reload Module</span>
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
