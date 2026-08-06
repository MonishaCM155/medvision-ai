import React, { useRef, useEffect, useState } from 'react';
import { ChevronDown, ShieldCheck, UserRound, Stethoscope, GraduationCap, FlaskConical, UserCog } from 'lucide-react';
import { useApp, AVAILABLE_ROLES } from '../contexts/AppContext';
import { UserRole } from '../types';
import { cn } from '../utils/cn';

const ROLE_ICON: Record<UserRole, React.ReactNode> = {
  Admin: <UserCog className="w-3.5 h-3.5" />,
  Radiologist: <Stethoscope className="w-3.5 h-3.5" />,
  Doctor: <UserRound className="w-3.5 h-3.5" />,
  Researcher: <FlaskConical className="w-3.5 h-3.5" />,
  Student: <GraduationCap className="w-3.5 h-3.5" />,
};

export const RoleSwitcher: React.FC = () => {
  const { user, switchUser } = useApp();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 pl-1 pr-2 py-1 rounded-full border border-slate-200 dark:border-slate-700 bg-white/70 dark:bg-slate-800/70 hover:border-primary-400 transition-colors cursor-pointer"
        title={`Signed in as ${user.name} (${user.role})`}
      >
        <span
          className="w-7 h-7 rounded-full bg-gradient-to-br from-indigo-500 to-violet-600 text-white text-[11px] font-bold flex items-center justify-center shadow"
        >
          {user.initials}
        </span>
        <span className="hidden md:flex flex-col items-start leading-tight">
          <span className="text-[11px] font-bold text-slate-800 dark:text-slate-100">{user.name.split(' ').slice(-1)[0]}</span>
          <span className="text-[9px] text-slate-500 dark:text-slate-400 font-medium">{user.role}</span>
        </span>
        <ChevronDown className={cn('w-3.5 h-3.5 text-slate-400 transition-transform', open && 'rotate-180')} />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-2 w-60 glass-strong rounded-2xl overflow-hidden z-50 animate-scale-in shadow-xl">
          <div className="px-4 py-2.5 border-b border-slate-200/60 dark:border-slate-700/60">
            <p className="text-[10px] font-mono uppercase tracking-widest text-slate-400">Simulated Session</p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">Switch hospital role to see role-aware UI.</p>
          </div>
          {AVAILABLE_ROLES.map((r) => (
            <button
              key={r.name}
              onClick={() => {
                switchUser(r);
                setOpen(false);
              }}
              className={cn(
                'w-full flex items-center gap-2.5 px-4 py-2.5 text-left transition-colors hover:bg-slate-50 dark:hover:bg-slate-800/60 cursor-pointer',
                user.name === r.name && 'bg-primary-50/70 dark:bg-primary-900/15'
              )}
            >
              <span className="w-7 h-7 rounded-full bg-gradient-to-br from-slate-500 to-slate-700 text-white text-[10px] font-bold flex items-center justify-center shrink-0">
                {r.initials}
              </span>
              <span className="flex-1 min-w-0">
                <span className="block text-xs font-semibold text-slate-800 dark:text-slate-100 truncate">{r.name}</span>
                <span className="block text-[10px] text-slate-500 dark:text-slate-400">{r.department}</span>
              </span>
              <span
                className={cn(
                  'inline-flex items-center gap-1 text-[9px] font-bold px-1.5 py-0.5 rounded-full border',
                  r.role === 'Admin'
                    ? 'text-violet-600 dark:text-violet-400 border-violet-300 dark:border-violet-700'
                    : 'text-primary-600 dark:text-primary-400 border-primary-300 dark:border-primary-700'
                )}
              >
                {ROLE_ICON[r.role]}
                {r.role}
              </span>
            </button>
          ))}
          <div className="px-4 py-2 border-t border-slate-200/60 dark:border-slate-700/60 flex items-center gap-1.5 text-[10px] text-slate-400">
            <ShieldCheck className="w-3 h-3 text-emerald-500" />
            <span>HIPAA-ready access controls (demo)</span>
          </div>
        </div>
      )}
    </div>
  );
};
