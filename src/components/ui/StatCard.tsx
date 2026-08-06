import React from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { GlassCard } from './GlassCard';
import { cn } from '../../utils/cn';

interface StatCardProps {
  label: string;
  value: string;
  icon: React.ReactNode;
  trend?: string;
  trendDirection?: 'up' | 'down' | 'flat';
  trendPositive?: boolean;
  accent?: string; // tailwind gradient classes for icon chip
  suffix?: string;
}

export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  icon,
  trend,
  trendDirection = 'flat',
  trendPositive = true,
  accent = 'from-indigo-500 to-violet-500',
  suffix,
}) => {
  const trendColor = trendDirection === 'flat' ? 'text-slate-400' : trendPositive ? 'text-emerald-500' : 'text-rose-500';
  const TrendIcon = trendDirection === 'up' ? TrendingUp : trendDirection === 'down' ? TrendingDown : Minus;

  return (
    <GlassCard hover className="group relative overflow-hidden animate-fade-in-up" >
      <div className="absolute -top-8 -right-8 w-24 h-24 rounded-full bg-gradient-to-br from-primary-500/10 to-transparent blur-2xl transition-opacity group-hover:opacity-100 opacity-60" />
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">{label}</p>
          <p className="mt-2 text-2xl font-black tracking-tight text-slate-900 dark:text-white font-mono tabular-nums truncate">
            {value}
            {suffix && <span className="text-sm font-semibold text-slate-400 ml-0.5">{suffix}</span>}
          </p>
          {trend && (
            <p className={cn('mt-1.5 text-[11px] font-medium flex items-center gap-1', trendColor)}>
              <TrendIcon className="w-3.5 h-3.5" />
              <span>{trend}</span>
            </p>
          )}
        </div>
        <div className={cn('w-11 h-11 rounded-xl bg-gradient-to-br text-white flex items-center justify-center shadow-lg shrink-0 transition-transform group-hover:scale-110 group-hover:rotate-3', accent)}>
          {icon}
        </div>
      </div>
    </GlassCard>
  );
};
