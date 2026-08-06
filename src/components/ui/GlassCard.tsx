import React from 'react';
import { cn } from '../../utils/cn';

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  gradient?: boolean;
  onClick?: () => void;
}

export const GlassCard: React.FC<GlassCardProps> = ({ children, className, hover = false, gradient = false, onClick }) => {
  return (
    <div
      onClick={onClick}
      className={cn(
        'surface-card p-5',
        hover && 'surface-card-hover cursor-pointer',
        gradient && 'ring-gradient',
        className
      )}
    >
      {children}
    </div>
  );
};

export const SectionHeader: React.FC<{
  icon?: React.ReactNode;
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
  className?: string;
}> = ({ icon, title, subtitle, right, className }) => {
  return (
    <div className={cn('flex flex-col sm:flex-row sm:items-center justify-between gap-3', className)}>
      <div className="flex items-center gap-2.5">
        {icon && <span className="text-primary-600 dark:text-primary-400 shrink-0">{icon}</span>}
        <div>
          <h2 className="text-base font-bold tracking-tight text-slate-900 dark:text-slate-100">{title}</h2>
          {subtitle && <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{subtitle}</p>}
        </div>
      </div>
      {right && <div className="flex items-center gap-2 shrink-0">{right}</div>}
    </div>
  );
};

export const LiveBadge: React.FC<{ label?: string; color?: 'emerald' | 'indigo' | 'rose' | 'amber' }> = ({
  label = 'Live',
  color = 'emerald',
}) => {
  const colors = {
    emerald: 'text-emerald-600 dark:text-emerald-400',
    indigo: 'text-indigo-600 dark:text-indigo-400',
    rose: 'text-rose-600 dark:text-rose-400',
    amber: 'text-amber-600 dark:text-amber-400',
  };
  return (
    <span className={`inline-flex items-center gap-1.5 text-[10px] font-mono font-semibold ${colors[color]}`}>
      <span className={`live-dot w-1.5 h-1.5 rounded-full ${colors[color]} bg-current`} />
      {label}
    </span>
  );
};

export const SeverityPill: React.FC<{ level: string; score?: number }> = ({ level, score }) => {
  const styles: Record<string, string> = {
    Low: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/30',
    Moderate: 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30',
    High: 'bg-orange-500/10 text-orange-600 dark:text-orange-400 border-orange-500/30',
    Critical: 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/30',
  };
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[10px] font-bold font-mono ${styles[level] ?? styles.Low}`}>
      {level}
      {typeof score === 'number' && <span className="opacity-70">· {score}/100</span>}
    </span>
  );
};
