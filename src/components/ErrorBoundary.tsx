import React from 'react';
import { AlertTriangle, RefreshCw, ShieldCheck } from 'lucide-react';

interface ErrorBoundaryProps {
  children: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  message: string;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, message: '' };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, message: error.message || 'Unexpected application error' };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[MedVision] Unhandled error:', error, info);
  }

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (!this.state.hasError) return this.props.children;

    return (
      <div className="min-h-screen flex items-center justify-center p-6 bg-[var(--bg)]">
        <div className="surface-card max-w-md w-full p-8 text-center animate-scale-in">
          <div className="w-14 h-14 mx-auto rounded-2xl bg-rose-500/10 border border-rose-500/25 flex items-center justify-center text-rose-500 mb-4">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <h1 className="text-lg font-black text-slate-900 dark:text-white">Something went wrong</h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-2 leading-relaxed">
            MedVision AI hit an unexpected error. Your data is safe — reload to continue.
          </p>
          <p className="mt-3 p-2 rounded-lg bg-slate-100 dark:bg-slate-800 text-[10px] font-mono text-rose-500 break-words">
            {this.state.message}
          </p>
          <div className="flex items-center justify-center gap-3 mt-6">
            <button
              onClick={this.handleReload}
              className="btn-gradient text-white text-xs font-bold px-5 py-2.5 rounded-xl flex items-center gap-2 cursor-pointer"
            >
              <RefreshCw className="w-4 h-4" /> Reload Application
            </button>
          </div>
          <p className="mt-5 flex items-center justify-center gap-1.5 text-[10px] text-slate-400">
            <ShieldCheck className="w-3 h-3 text-emerald-500" /> Research & educational use only
          </p>
        </div>
      </div>
    );
  }
}
