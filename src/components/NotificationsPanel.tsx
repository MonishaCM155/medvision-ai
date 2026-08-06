import React, { useRef, useEffect } from 'react';
import { Bell, CheckCheck, X, ShieldAlert, CheckCircle2, Info, AlertTriangle } from 'lucide-react';
import { useApp } from '../contexts/AppContext';
import { AppNotification } from '../types';
import { cn } from '../utils/cn';

const KIND_META = {
  success: { icon: CheckCircle2, color: 'text-emerald-500', bg: 'bg-emerald-500/10' },
  info: { icon: Info, color: 'text-sky-500', bg: 'bg-sky-500/10' },
  warning: { icon: AlertTriangle, color: 'text-amber-500', bg: 'bg-amber-500/10' },
  critical: { icon: ShieldAlert, color: 'text-rose-500', bg: 'bg-rose-500/10' },
};

export const NotificationsPanel: React.FC<{ open: boolean; onClose: () => void }> = ({ open, onClose }) => {
  const { notifications, unreadCount, markAllRead, markRead, dismissNotification } = useApp();
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div ref={ref} className="absolute right-0 top-full mt-2 w-[22rem] max-w-[calc(100vw-2rem)] glass-strong rounded-2xl overflow-hidden z-50 animate-scale-in">
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200/60 dark:border-slate-700/60">
        <div className="flex items-center gap-2">
          <Bell className="w-4 h-4 text-primary-500" />
          <span className="text-sm font-bold text-slate-900 dark:text-white">Notifications</span>
          {unreadCount > 0 && (
            <span className="text-[10px] font-mono font-bold bg-rose-500 text-white rounded-full px-1.5 py-0.5">{unreadCount} new</span>
          )}
        </div>
        <button
          onClick={markAllRead}
          className="text-[11px] font-semibold text-primary-600 dark:text-primary-400 hover:underline flex items-center gap-1 cursor-pointer"
        >
          <CheckCheck className="w-3.5 h-3.5" /> Mark all read
        </button>
      </div>

      <div className="max-h-[22rem] overflow-y-auto divide-y divide-slate-100 dark:divide-slate-800">
        {notifications.length === 0 && (
          <div className="p-8 text-center text-xs text-slate-400">You're all caught up 🎉</div>
        )}
        {notifications.map((n: AppNotification) => {
          const meta = KIND_META[n.kind];
          const Icon = meta.icon;
          return (
            <div
              key={n.id}
              onClick={() => markRead(n.id)}
              className={cn(
                'flex gap-3 px-4 py-3 cursor-pointer transition-colors hover:bg-slate-50 dark:hover:bg-slate-800/50',
                !n.read && 'bg-primary-50/50 dark:bg-primary-900/10'
              )}
            >
              <div className={cn('w-8 h-8 rounded-lg flex items-center justify-center shrink-0', meta.bg, meta.color)}>
                <Icon className="w-4 h-4" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-start justify-between gap-2">
                  <p className={cn('text-xs font-semibold text-slate-900 dark:text-slate-100', !n.read && 'font-bold')}>{n.title}</p>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      dismissNotification(n.id);
                    }}
                    className="text-slate-300 hover:text-rose-500 transition-colors cursor-pointer shrink-0"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
                <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5 leading-snug">{n.body}</p>
                <p className="text-[10px] font-mono text-slate-400 mt-1">{n.time}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
