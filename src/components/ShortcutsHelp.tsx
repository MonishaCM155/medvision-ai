import React, { useEffect } from 'react';
import { Keyboard, X, Command, ScanLine, Sun, Moon, Bot, Search, HelpCircle } from 'lucide-react';

interface ShortcutsHelpProps {
  open: boolean;
  onClose: () => void;
}

const GROUPS: { title: string; shortcuts: { keys: string; label: string; icon?: React.ReactNode }[] }[] = [
  {
    title: 'General',
    shortcuts: [
      { keys: 'Ctrl K', label: 'Open command palette', icon: <Command className="w-3.5 h-3.5" /> },
      { keys: '?', label: 'Show this help', icon: <HelpCircle className="w-3.5 h-3.5" /> },
      { keys: '/', label: 'Focus global search', icon: <Search className="w-3.5 h-3.5" /> },
      { keys: 'Esc', label: 'Close overlays' },
    ],
  },
  {
    title: 'Workspace',
    shortcuts: [
      { keys: 'C', label: 'Toggle AI Copilot', icon: <Bot className="w-3.5 h-3.5" /> },
      { keys: 'T', label: 'Toggle dark / light mode', icon: <Sun className="w-3.5 h-3.5" /> },
      { keys: 'N', label: 'New scan (inference)', icon: <ScanLine className="w-3.5 h-3.5" /> },
      { keys: 'Moon', label: 'Switch to dark mode', icon: <Moon className="w-3.5 h-3.5" /> },
    ],
  },
];

const KeyCap: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <kbd className="inline-flex items-center justify-center min-w-[1.5rem] px-1.5 py-0.5 rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-[10px] font-mono font-bold text-slate-700 dark:text-slate-200 shadow-sm">
    {children}
  </kbd>
);

export const ShortcutsHelp: React.FC<ShortcutsHelpProps> = ({ open, onClose }) => {
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[70] bg-slate-950/50 backdrop-blur-sm flex items-center justify-center p-4 animate-fade-in" onClick={onClose}>
      <div
        className="w-full max-w-md glass-strong rounded-2xl overflow-hidden shadow-2xl animate-scale-in"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
      >
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-200/60 dark:border-slate-700/60">
          <div className="flex items-center gap-2.5">
            <span className="w-8 h-8 rounded-lg bg-primary-500/15 text-primary-600 dark:text-primary-400 flex items-center justify-center">
              <Keyboard className="w-4 h-4" />
            </span>
            <div>
              <h2 className="text-sm font-bold text-slate-900 dark:text-white">Keyboard Shortcuts</h2>
              <p className="text-[10px] text-slate-400">Work faster with global commands</p>
            </div>
          </div>
          <button onClick={onClose} className="p-1.5 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors cursor-pointer" aria-label="Close">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="px-5 py-4 space-y-5 max-h-[60vh] overflow-y-auto">
          {GROUPS.map((g) => (
            <div key={g.title}>
              <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400 mb-2">{g.title}</p>
              <div className="space-y-1.5">
                {g.shortcuts.map((s, i) => (
                  <div key={i} className="flex items-center justify-between py-1.5">
                    <span className="text-xs text-slate-700 dark:text-slate-200 flex items-center gap-2">
                      {s.icon}
                      {s.label}
                    </span>
                    <span className="flex items-center gap-1">
                      {s.keys.split(' ').map((k, j) => (
                        <React.Fragment key={j}>
                          {j > 0 && <span className="text-slate-400 text-[10px]">+</span>}
                          <KeyCap>{k}</KeyCap>
                        </React.Fragment>
                      ))}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ))}

          <div className="rounded-xl bg-amber-500/10 border border-amber-500/25 p-3 text-[10px] text-amber-700 dark:text-amber-300 leading-relaxed">
            <strong>Tip:</strong> shortcuts are disabled while typing in a text field, so you can safely use <KeyCap>C</KeyCap>, <KeyCap>T</KeyCap> or <KeyCap>/</KeyCap> in patient notes and chat.
          </div>
        </div>
      </div>
    </div>
  );
};
