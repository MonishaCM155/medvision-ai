import React, { useEffect } from 'react';
import { CheckCircle2, Info, AlertTriangle, ShieldAlert, X } from 'lucide-react';
import { useApp, Toast } from '../contexts/AppContext';
import { cn } from '../utils/cn';

const KIND_META = {
  success: { icon: CheckCircle2, ring: 'border-emerald-500/40', iconCls: 'text-emerald-500', bar: 'bg-emerald-500' },
  info: { icon: Info, ring: 'border-sky-500/40', iconCls: 'text-sky-500', bar: 'bg-sky-500' },
  warning: { icon: AlertTriangle, ring: 'border-amber-500/40', iconCls: 'text-amber-500', bar: 'bg-amber-500' },
  critical: { icon: ShieldAlert, ring: 'border-rose-500/40', iconCls: 'text-rose-500', bar: 'bg-rose-500' },
};

const ToastItem: React.FC<{ toast: Toast }> = ({ toast }) => {
  const { dismissToast } = useApp();
  const meta = KIND_META[toast.kind];
  const Icon = meta.icon;

  // Auto-dismiss
  useEffect(() => {
    const t = setTimeout(() => dismissToast(toast.id), 5200);
    return () => clearTimeout(t);
  }, [toast.id, dismissToast]);

  return (
    <div className={cn('relative glass-strong rounded-xl border p-3.5 pr-9 w-80 max-w-[calc(100vw-2rem)] shadow-xl animate-slide-in-right overflow-hidden', meta.ring)} role="status">
      <div className="flex gap-2.5">
        <span className={cn('shrink-0 mt-0.5', meta.iconCls)}>
          <Icon className="w-4.5 h-4.5" />
        </span>
        <div className="min-w-0">
          <p className="text-xs font-bold text-slate-900 dark:text-white">{toast.title}</p>
          {toast.body && <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5 leading-snug">{toast.body}</p>}
        </div>
      </div>
      <button
        onClick={() => dismissToast(toast.id)}
        className="absolute top-2.5 right-2.5 p-1 text-slate-300 hover:text-slate-500 dark:hover:text-slate-200 transition-colors cursor-pointer"
        aria-label="Dismiss notification"
      >
        <X className="w-3.5 h-3.5" />
      </button>
      {/* Auto-dismiss progress */}
      <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-slate-100 dark:bg-slate-800">
        <div className={cn('h-full rounded-full animate-toast-progress', meta.bar)} />
      </div>
    </div>
  );
};

export const ToastHost: React.FC = () => {
  const { toasts } = useApp();
  if (toasts.length === 0) return null;
  return (
    <div className="fixed top-20 right-4 z-[60] flex flex-col gap-2.5 items-end pointer-events-none">
      {toasts.map((t) => (
        <div key={t.id} className="pointer-events-auto">
          <ToastItem toast={t} />
        </div>
      ))}
    </div>
  );
};
