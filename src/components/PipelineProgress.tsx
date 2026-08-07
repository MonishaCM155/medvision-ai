import React from 'react';
import {
  UploadCloud, ShieldCheck, ScanLine, Cpu, BrainCircuit, FileText,
  Check, X, Loader2,
} from 'lucide-react';
import { cn } from '../utils/cn';

export const PIPELINE_STAGES = [
  { key: 'upload', label: 'Upload', icon: UploadCloud },
  { key: 'validation', label: 'Validation', icon: ShieldCheck },
  { key: 'quality', label: 'Quality Check', icon: ScanLine },
  { key: 'analysis', label: 'AI Analysis', icon: Cpu },
  { key: 'explainability', label: 'Explainability', icon: BrainCircuit },
  { key: 'report', label: 'Report', icon: FileText },
] as const;

export type PipelineStatus = 'idle' | 'running' | 'passed' | 'failed';

interface PipelineProgressProps {
  stage: number; // 0..5 index into PIPELINE_STAGES
  status: PipelineStatus;
  message?: string;
}

export const PipelineProgress: React.FC<PipelineProgressProps> = ({ stage, status, message }) => {
  const running = status === 'running';
  const failed = status === 'failed';
  const passed = status === 'passed';

  return (
    <div
      className={cn(
        'rounded-2xl border p-4 sm:p-5 transition-colors duration-500',
        failed
          ? 'border-rose-300/60 dark:border-rose-500/40 bg-rose-50/70 dark:bg-rose-500/10'
          : passed
            ? 'border-emerald-300/60 dark:border-emerald-500/30 bg-emerald-50/60 dark:bg-emerald-500/5'
            : 'border-slate-200 dark:border-slate-700/80 bg-white/70 dark:bg-slate-900/50 backdrop-blur-sm'
      )}
    >
      {/* Top row: title + status pill */}
      <div className="flex items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-black uppercase tracking-widest text-slate-500 dark:text-slate-400">
            Analysis Pipeline
          </span>
        </div>
        <span
          className={cn(
            'inline-flex items-center gap-1.5 text-[10px] font-bold px-2.5 py-1 rounded-full border font-mono',
            failed
              ? 'text-rose-700 dark:text-rose-300 bg-rose-500/10 border-rose-500/30'
              : passed
                ? 'text-emerald-700 dark:text-emerald-300 bg-emerald-500/10 border-emerald-500/30'
                : running
                  ? 'text-indigo-700 dark:text-indigo-300 bg-indigo-500/10 border-indigo-500/30'
                  : 'text-slate-500 dark:text-slate-400 bg-slate-500/10 border-slate-500/25'
          )}
        >
          {failed ? (
            <><X className="w-3 h-3" /> BLOCKED</>
          ) : passed ? (
            <><Check className="w-3 h-3" /> COMPLETE</>
          ) : running ? (
            <><Loader2 className="w-3 h-3 animate-spin" /> PROCESSING</>
          ) : (
            'STANDBY'
          )}
        </span>
      </div>

      {/* Progress bar */}
      <div className="h-1.5 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden mb-4">
        <div
          className={cn(
            'h-full rounded-full transition-all duration-700 ease-out',
            failed
              ? 'bg-gradient-to-r from-rose-500 to-red-500'
              : passed
                ? 'bg-gradient-to-r from-emerald-500 to-teal-500'
                : 'bg-gradient-to-r from-indigo-500 to-violet-500'
          )}
          style={{ width: `${Math.round(((failed ? stage : passed ? 6 : stage + 1) / 6) * 100)}%` }}
        />
      </div>

      {/* Stage nodes */}
      <ol className="grid grid-cols-3 sm:grid-cols-6 gap-2">
        {PIPELINE_STAGES.map((s, i) => {
          const done = passed || i < stage;
          const active = running && i === stage;
          const blocked = failed && i === stage;
          const Icon = s.icon;
          return (
            <li key={s.key} className="flex flex-col items-center gap-1.5 text-center min-w-0">
              <span
                className={cn(
                  'w-8 h-8 rounded-full flex items-center justify-center border transition-all duration-300',
                  done && 'bg-emerald-500 border-emerald-500 text-white shadow-sm',
                  active && 'bg-indigo-600 border-indigo-500 text-white ring-4 ring-indigo-500/25 animate-pulse-soft',
                  blocked && 'bg-rose-500 border-rose-500 text-white ring-4 ring-rose-500/25',
                  !done && !active && !blocked && 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-400'
                )}
              >
                {done ? (
                  <Check className="w-4 h-4" />
                ) : active ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : blocked ? (
                  <X className="w-4 h-4" />
                ) : (
                  <Icon className="w-4 h-4" />
                )}
              </span>
              <span
                className={cn(
                  'text-[9px] font-bold leading-tight tracking-wide uppercase',
                  done && 'text-emerald-600 dark:text-emerald-400',
                  blocked && 'text-rose-600 dark:text-rose-400',
                  active && 'text-indigo-600 dark:text-indigo-400',
                  !done && !active && !blocked && 'text-slate-400 dark:text-slate-500'
                )}
              >
                {s.label}
              </span>
            </li>
          );
        })}
      </ol>

      {/* Status message */}
      {message && (
        <p className="mt-3 text-[11px] leading-relaxed text-slate-600 dark:text-slate-300 flex items-start gap-1.5">
          {failed ? (
            <X className="w-3.5 h-3.5 text-rose-500 shrink-0 mt-0.5" />
          ) : passed ? (
            <Check className="w-3.5 h-3.5 text-emerald-500 shrink-0 mt-0.5" />
          ) : (
            <Loader2 className="w-3.5 h-3.5 text-indigo-500 shrink-0 mt-0.5 animate-spin" />
          )}
          <span>{message}</span>
        </p>
      )}
    </div>
  );
};
