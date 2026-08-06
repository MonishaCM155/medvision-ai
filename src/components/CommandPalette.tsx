import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Search, LayoutDashboard, UploadCloud, Layers, UserCheck, Users, BookOpen, History,
  Cpu, GraduationCap, Database, BarChart3, Activity, FileText, Settings, Bot, Sun, Moon,
  ScanLine, CornerDownLeft, Command, UserRound,
} from 'lucide-react';
import { NavTab } from './Header';
import { useApp } from '../contexts/AppContext';
import { PATIENTS, HUB_MODELS, DATASETS } from '../data/mockEnterprise';
import { cn } from '../utils/cn';

interface CommandPaletteProps {
  onNavigate: (tab: NavTab) => void;
  onNewScan: () => void;
}

interface PaletteItem {
  id: string;
  label: string;
  hint?: string;
  icon: React.ReactNode;
  group: string;
  keywords: string;
  execute: () => void;
}

const TAB_ITEMS: { tab: NavTab; label: string; icon: React.ReactNode; keywords: string }[] = [
  { tab: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard className="w-4 h-4" />, keywords: 'home overview kpi' },
  { tab: 'inference', label: 'Inference', icon: <UploadCloud className="w-4 h-4" />, keywords: 'scan xray upload predict analyze' },
  { tab: 'batch', label: 'Batch Processing', icon: <Layers className="w-4 h-4" />, keywords: 'bulk cohort queue' },
  { tab: 'doctor', label: 'DICOM Studio', icon: <UserCheck className="w-4 h-4" />, keywords: 'viewer workspace radiologist window level' },
  { tab: 'patients', label: 'Patients', icon: <Users className="w-4 h-4" />, keywords: 'ehr records profile' },
  { tab: 'knowledge', label: 'Knowledge Hub', icon: <BookOpen className="w-4 h-4" />, keywords: 'disease glossary education' },
  { tab: 'history', label: 'History & Audit', icon: <History className="w-4 h-4" />, keywords: 'logs past reports' },
  { tab: 'modelhub', label: 'Model Hub', icon: <Cpu className="w-4 h-4" />, keywords: 'architectures benchmark compare' },
  { tab: 'training', label: 'Training', icon: <GraduationCap className="w-4 h-4" />, keywords: 'experiment loss epochs hyperparams' },
  { tab: 'datasets', label: 'Datasets', icon: <Database className="w-4 h-4" />, keywords: 'nih chexpert mimic rsna vindr' },
  { tab: 'models', label: 'Benchmarks', icon: <BarChart3 className="w-4 h-4" />, keywords: 'comparison auroc latency' },
  { tab: 'mlops', label: 'MLOps', icon: <Activity className="w-4 h-4" />, keywords: 'calibration mlflow registry deploy' },
  { tab: 'analytics', label: 'Analytics & GPU', icon: <BarChart3 className="w-4 h-4" />, keywords: 'charts telemetry gpu' },
  { tab: 'docs', label: 'Documentation', icon: <FileText className="w-4 h-4" />, keywords: 'docs architecture help' },
  { tab: 'settings', label: 'Settings', icon: <Settings className="w-4 h-4" />, keywords: 'preferences theme profile' },
];

export const CommandPalette: React.FC<CommandPaletteProps> = ({ onNavigate, onNewScan }) => {
  const { theme, toggleTheme, isCopilotAvailable, toggleCopilot } = usePaletteActions();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Toggle on Ctrl/Cmd+K, open on demand (e.g. '/' fallback on small screens)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setOpen((o) => {
          const next = !o;
          if (next) setQuery('');
          return next;
        });
      }
    };
    const openHandler = () => {
      setQuery('');
      setOpen(true);
    };
    window.addEventListener('keydown', handler);
    window.addEventListener('medvision:open-palette', openHandler);
    return () => {
      window.removeEventListener('keydown', handler);
      window.removeEventListener('medvision:open-palette', openHandler);
    };
  }, []);

  // Focus input when opened
  useEffect(() => {
    if (open) {
      setActiveIndex(0);
      setTimeout(() => inputRef.current?.focus(), 20);
    }
  }, [open]);

  // ESC closes
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [open]);

  const items = useMemo<PaletteItem[]>(() => {
    const pages: PaletteItem[] = TAB_ITEMS.map((t) => ({
      id: `tab-${t.tab}`,
      label: t.label,
      icon: t.icon,
      group: 'Navigate',
      keywords: t.keywords,
      execute: () => onNavigate(t.tab),
    }));

    const patients: PaletteItem[] = PATIENTS.slice(0, 6).map((p) => ({
      id: `patient-${p.id}`,
      label: p.name,
      hint: `${p.id} · ${p.topDiagnosis ?? '—'}`,
      icon: <span className="w-4 h-4 rounded-full text-[9px] font-bold text-white flex items-center justify-center" style={{ background: `hsl(${p.avatarHue} 70% 55%)` }}>{p.name[0]}</span>,
      group: 'Patients',
      keywords: `${p.name} ${p.id} ${p.topDiagnosis ?? ''}`,
      execute: () => onNavigate('patients'),
    }));

    const models: PaletteItem[] = HUB_MODELS.slice(0, 8).map((m) => ({
      id: `model-${m.id}`,
      label: m.name,
      hint: `${m.task} · AUROC ${m.auroc.toFixed(3)}`,
      icon: <span className="w-4 h-4 rounded-md flex items-center justify-center text-[9px] font-black text-white" style={{ background: m.color }}>{m.name[0]}</span>,
      group: 'Models',
      keywords: `${m.name} ${m.task}`,
      execute: () => onNavigate('modelhub'),
    }));

    const datasets: PaletteItem[] = DATASETS.slice(0, 5).map((d) => ({
      id: `ds-${d.id}`,
      label: d.name,
      hint: `${d.images.toLocaleString()} images`,
      icon: <Database className="w-4 h-4" />,
      group: 'Datasets',
      keywords: d.name,
      execute: () => onNavigate('datasets'),
    }));

    const actions: PaletteItem[] = [
      {
        id: 'act-new-scan',
        label: 'New Scan',
        hint: 'Open inference workspace',
        icon: <ScanLine className="w-4 h-4" />,
        group: 'Actions',
        keywords: 'scan upload new',
        execute: () => {
          onNewScan();
          onNavigate('inference');
        },
      },
      {
        id: 'act-copilot',
        label: isCopilotAvailable ? 'Toggle AI Copilot' : 'Toggle AI Copilot',
        hint: 'Show or hide the assistant panel',
        icon: <Bot className="w-4 h-4" />,
        group: 'Actions',
        keywords: 'copilot chat assistant',
        execute: toggleCopilot,
      },
      {
        id: 'act-theme',
        label: theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode',
        icon: theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />,
        group: 'Actions',
        keywords: 'theme dark light appearance',
        execute: toggleTheme,
      },
      {
        id: 'act-profile',
        label: 'Switch Role',
        hint: 'Admin · Radiologist · Doctor · Researcher · Student',
        icon: <UserRound className="w-4 h-4" />,
        group: 'Actions',
        keywords: 'role profile user account',
        execute: () => onNavigate('settings'),
      },
    ];

    return [...pages, ...patients, ...models, ...datasets, ...actions];
  }, [onNavigate, onNewScan, theme, isCopilotAvailable, toggleCopilot, toggleTheme]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items.slice(0, 30);
    const scored = items.map((item) => {
      const haystack = `${item.label} ${item.hint ?? ''} ${item.keywords}`.toLowerCase();
      const idx = haystack.indexOf(q);
      return { item, score: idx === -1 ? -1 : idx + (item.group === 'Navigate' ? 0 : 100) };
    });
    return scored
      .filter((s) => s.score !== -1)
      .sort((a, b) => a.score - b.score)
      .slice(0, 30)
      .map((s) => s.item);
  }, [items, query]);

  // Keyboard navigation
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const item = filtered[activeIndex];
      if (item) {
        item.execute();
        setOpen(false);
      }
    }
  };

  // Scroll active item into view
  useEffect(() => {
    const el = listRef.current?.querySelector('[data-active="true"]');
    el?.scrollIntoView({ block: 'nearest' });
  }, [activeIndex]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[70] bg-slate-950/50 backdrop-blur-sm flex items-start justify-center pt-[12vh] px-4 animate-fade-in" onClick={() => setOpen(false)}>
      <div
        className="w-full max-w-xl glass-strong rounded-2xl overflow-hidden shadow-2xl animate-scale-in"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
      >
        {/* Input */}
        <div className="flex items-center gap-3 px-4 py-3.5 border-b border-slate-200/60 dark:border-slate-700/60">
          <Command className="w-4.5 h-4.5 text-primary-500 shrink-0" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setActiveIndex(0);
            }}
            onKeyDown={onKeyDown}
            placeholder="Search pages, patients, models, datasets, actions…"
            className="flex-1 bg-transparent text-sm text-slate-800 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500 focus:outline-none"
          />
          <kbd className="text-[9px] font-mono text-slate-400 border border-slate-200 dark:border-slate-700 rounded px-1.5 py-0.5">ESC</kbd>
        </div>

        {/* Results */}
        <div ref={listRef} className="max-h-[46vh] overflow-y-auto p-2">
          {filtered.length === 0 && (
            <div className="py-10 text-center text-xs text-slate-400">
              No results for <strong className="font-mono">“{query}”</strong>
            </div>
          )}
          {filtered.map((item, i) => {
            const active = i === activeIndex;
            return (
              <button
                key={item.id}
                data-active={active}
                onClick={() => {
                  item.execute();
                  setOpen(false);
                }}
                onMouseEnter={() => setActiveIndex(i)}
                className={cn(
                  'w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-left transition-colors cursor-pointer',
                  active ? 'bg-primary-500/10 text-primary-700 dark:text-primary-300' : 'text-slate-700 dark:text-slate-200'
                )}
              >
                <span className={cn('w-7 h-7 rounded-lg flex items-center justify-center shrink-0', active ? 'bg-primary-500/15 text-primary-600 dark:text-primary-400' : 'bg-slate-100 dark:bg-slate-800 text-slate-500')}>
                  {item.icon}
                </span>
                <span className="flex-1 min-w-0">
                  <span className="block text-xs font-semibold truncate">{item.label}</span>
                  {item.hint && <span className="block text-[10px] text-slate-400 truncate">{item.hint}</span>}
                </span>
                <span className="text-[9px] font-mono uppercase tracking-widest text-slate-300 dark:text-slate-600">{item.group}</span>
                {active && <CornerDownLeft className="w-3.5 h-3.5 text-primary-400 shrink-0" />}
              </button>
            );
          })}
        </div>

        {/* Footer hints */}
        <div className="px-4 py-2.5 border-t border-slate-200/60 dark:border-slate-700/60 flex items-center gap-3 text-[10px] text-slate-400">
          <span className="flex items-center gap-1"><kbd className="border border-slate-300 dark:border-slate-600 rounded px-1 font-mono">↑↓</kbd> navigate</span>
          <span className="flex items-center gap-1"><kbd className="border border-slate-300 dark:border-slate-600 rounded px-1 font-mono">↵</kbd> select</span>
          <span className="ml-auto font-mono">MedVision AI · Command</span>
        </div>
      </div>
    </div>
  );
};

function usePaletteActions() {
  const ctx = useApp();
  return {
    theme: ctx.theme,
    toggleTheme: ctx.toggleTheme,
    isCopilotAvailable: true,
    toggleCopilot: () => {
      // Copilot toggling is wired by the App-level handler; the palette just triggers it
      window.dispatchEvent(new CustomEvent('medvision:toggle-copilot'));
    },
  };
}
