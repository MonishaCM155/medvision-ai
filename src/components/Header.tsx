import React, { useState } from 'react';
import { Menu, Sun, Moon, Bell, Search, ShieldAlert, Activity } from 'lucide-react';
import { useApp } from '../contexts/AppContext';
import { NotificationsPanel } from './NotificationsPanel';
import { cn } from '../utils/cn';

export type NavTab =
  | 'dashboard'
  | 'inference'
  | 'batch'
  | 'history'
  | 'models'
  | 'mlops'
  | 'analytics'
  | 'knowledge'
  | 'doctor'
  | 'docs'
  | 'patients'
  | 'modelhub'
  | 'training'
  | 'datasets'
  | 'settings';

interface HeaderProps {
  activeTab: NavTab;
  setActiveTab: (tab: NavTab) => void;
  onToggleSidebar?: () => void;
  onToggleCopilot?: () => void;
  isCopilotOpen?: boolean;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab, onToggleSidebar, onToggleCopilot, isCopilotOpen }) => {
  const { theme, toggleTheme, unreadCount, engine } = useApp();
  const [notifOpen, setNotifOpen] = useState(false);
  const [query, setQuery] = useState('');

  const handleGlobalSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    // Global search navigates to the most relevant tab based on keywords
    const lower = q.toLowerCase();
    if (/(patient|pat-)/.test(lower)) setActiveTab('patients');
    else if (/(model|hub|architecture|benchmark)/.test(lower)) setActiveTab('modelhub');
    else if (/(train|epoch|loss)/.test(lower)) setActiveTab('training');
    else if (/(dataset|nih|chexpert|mimic|rsna)/.test(lower)) setActiveTab('datasets');
    else if (/(xray|x-ray|scan|upload|inference)/.test(lower)) setActiveTab('inference');
    else if (/(history|audit|log)/.test(lower)) setActiveTab('history');
    else if (/(mlops|calibrat|mlflow)/.test(lower)) setActiveTab('mlops');
    else if (/(knowledge|disease|glossary)/.test(lower)) setActiveTab('knowledge');
    else setActiveTab('dashboard');
    setQuery('');
  };

  const tabTitle: Record<NavTab, string> = {
    dashboard: 'Enterprise Dashboard',
    inference: 'Inference',
    batch: 'Batch',
    history: 'History',
    models: 'Models',
    mlops: 'MLOps',
    analytics: 'Analytics & GPU',
    knowledge: 'Knowledge Hub',
    doctor: 'DICOM Studio',
    docs: 'Docs',
    patients: 'Patients',
    modelhub: 'Model Hub',
    training: 'Training',
    datasets: 'Datasets',
    settings: 'Settings',
  };

  return (
    <header className="sticky top-0 z-40 glass-strong border-b border-slate-200/50 dark:border-slate-800/60">
      {/* Ultra-compact disclaimer banner */}
      <div className="bg-gradient-to-r from-amber-600/90 via-orange-600/85 to-amber-600/90 px-4 py-1 text-[10px] text-white flex items-center justify-between">
        <div className="flex items-center gap-1.5 max-w-7xl mx-auto w-full">
          <ShieldAlert className="w-3 h-3 shrink-0" />
          <span className="font-medium truncate">
            <strong>RESEARCH ONLY:</strong> Not for clinical diagnosis. All AI outputs require radiologist validation.
          </span>
        </div>
        <span className="hidden md:inline-block text-[9px] bg-white/15 px-1.5 py-0.5 rounded border border-white/25 font-mono shrink-0">
          v2.7.0 · PUBLIC RESEARCH MODE
        </span>
      </div>

      {/* Main bar */}
      <div className="max-w-[1600px] mx-auto px-3 sm:px-5">
        <div className="flex items-center justify-between gap-3 h-14">
          {/* Left: burger + brand */}
          <div className="flex items-center gap-2 min-w-0">
            {onToggleSidebar && (
              <button
                onClick={onToggleSidebar}
                className="p-2 rounded-lg text-slate-500 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer lg:hidden"
                title="Toggle navigation"
              >
                <Menu className="w-4.5 h-4.5" />
              </button>
            )}
            <button
              onClick={() => setActiveTab('dashboard')}
              className="flex items-center gap-2.5 cursor-pointer group shrink-0"
            >
              <div className="relative w-8 h-8 rounded-xl bg-gradient-to-br from-indigo-600 via-violet-600 to-sky-500 flex items-center justify-center text-white font-black text-sm shadow-lg shadow-indigo-500/30 group-hover:scale-105 transition-transform">
                M
                <span className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-emerald-400 border-2 border-white dark:border-slate-900" />
              </div>
              <div className="hidden sm:block text-left">
                <h1 className="text-sm font-extrabold tracking-tight text-slate-900 dark:text-white leading-none">
                  MedVision <span className="text-gradient">AI</span>
                </h1>
                <p className="text-[9px] text-slate-400 font-mono mt-0.5 tracking-wide">EXPLAINABLE MEDICAL SUITE</p>
              </div>
            </button>
          </div>

          {/* Center: global search */}
          <form onSubmit={handleGlobalSearch} className="hidden md:block flex-1 max-w-md mx-auto">
            <div className="relative group">
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 transition-colors group-focus-within:text-primary-500" />
              <input
                id="global-search-input"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search patients, models, datasets, scans…  (/)"
                className="w-full bg-slate-100/80 dark:bg-slate-800/70 border border-transparent focus:border-primary-400/50 focus:bg-white dark:focus:bg-slate-800 text-xs rounded-full pl-9 pr-4 py-2 outline-none transition-all text-slate-700 dark:text-slate-200 placeholder:text-slate-400 dark:placeholder:text-slate-500"
              />
              <kbd className="hidden lg:inline absolute right-3 top-1/2 -translate-y-1/2 text-[9px] font-mono text-slate-400 border border-slate-200 dark:border-slate-700 rounded px-1.5 py-0.5">
                ↵
              </kbd>
            </div>
          </form>

          {/* Right: actions */}
          <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
            <span
              className={cn(
                'hidden xl:inline-flex items-center gap-1.5 text-[10px] font-mono rounded-full px-2.5 py-1 border transition-colors',
                engine?.status === 'ready' && String(engine.source).startsWith('pytorch')
                  ? 'text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 border-emerald-500/25'
                  : 'text-amber-600 dark:text-amber-400 bg-amber-500/10 border-amber-500/25'
              )}
              title={engine?.note || (engine ? `Engine: ${engine.source}${engine.checkpointFile ? ` · ${engine.checkpointFile}` : ''}` : 'Probing inference engine…')}
            >
              <span className={cn('w-1.5 h-1.5 rounded-full', engine?.status === 'ready' && String(engine.source).startsWith('pytorch') ? 'live-dot bg-emerald-500' : 'bg-amber-500')} />
              {!engine
                ? 'AI CORE …'
                : engine.source === 'pytorch-checkpoint'
                  ? `REAL MODEL · ${String(engine.device).startsWith('cuda') ? 'CUDA' : 'CPU'}`
                  : engine.status === 'ready' && engine.source === 'pytorch-backbone'
                    ? `BACKBONE · ${String(engine.device).startsWith('cuda') ? 'CUDA' : 'CPU'}`
                    : 'DEMO ENGINE'}
            </span>

            {/* Copilot quick toggle */}
            {onToggleCopilot && (
              <button
                onClick={onToggleCopilot}
                className={`hidden sm:inline-flex items-center gap-1.5 text-[11px] font-semibold px-3 py-1.5 rounded-full border transition-colors cursor-pointer ${
                  isCopilotOpen
                    ? 'bg-indigo-600 text-white border-indigo-600 shadow-md shadow-indigo-500/30'
                    : 'text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:border-indigo-400'
                }`}
              >
                <Activity className="w-3.5 h-3.5" />
                Copilot
              </button>
            )}

            {/* Theme toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 rounded-full text-slate-500 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-amber-500 dark:hover:text-amber-400 transition-all cursor-pointer"
              title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {theme === 'dark' ? <Sun className="w-4.5 h-4.5" /> : <Moon className="w-4.5 h-4.5" />}
            </button>

            {/* Notifications */}
            <div className="relative">
              <button
                onClick={() => setNotifOpen((o) => !o)}
                className="relative p-2 rounded-full text-slate-500 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer"
                title="Notifications"
              >
                <Bell className="w-4.5 h-4.5" />
                {unreadCount > 0 && (
                  <span className="absolute -top-0.5 -right-0.5 min-w-[1.05rem] h-[1.05rem] px-0.5 rounded-full bg-rose-500 text-white text-[9px] font-bold flex items-center justify-center border-2 border-white dark:border-slate-900">
                    {unreadCount}
                  </span>
                )}
              </button>
              <NotificationsPanel open={notifOpen} onClose={() => setNotifOpen(false)} />
            </div>

            <span className="hidden sm:block w-px h-6 bg-slate-200 dark:bg-slate-700 mx-0.5" />

            {/* Public research mode badge — no accounts, no login */}
            <span
              className="hidden lg:inline-flex items-center gap-1.5 text-[10px] font-mono font-bold px-2.5 py-1.5 rounded-full border border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400"
              title="MedVision AI is a public research & education platform. No login, accounts, or roles are required."
            >
              <span className="w-1.5 h-1.5 rounded-full bg-slate-400" />
              PUBLIC MODE
            </span>
          </div>
        </div>

        {/* Mobile active tab indicator */}
        <div className="lg:hidden pb-2 -mt-1">
          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">
            {tabTitle[activeTab]}
          </span>
        </div>
      </div>
    </header>
  );
};
