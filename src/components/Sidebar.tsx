import React from 'react';
import {
  LayoutDashboard, UploadCloud, Layers, UserCheck, Users, BookOpen, History,
  Cpu, GraduationCap, Database, BarChart3, Activity, FileText, Settings,
  ChevronLeft, Bot, X, HeartPulse, Gauge,
} from 'lucide-react';
import { NavTab } from './Header';
import { useApp } from '../contexts/AppContext';
import { cn } from '../utils/cn';

interface NavItem {
  tab: NavTab;
  label: string;
  icon: React.ReactNode;
  badge?: string;
}

interface NavGroup {
  title: string;
  items: NavItem[];
}

interface SidebarProps {
  activeTab: NavTab;
  onNavigate: (tab: NavTab) => void;
  onToggleCopilot?: () => void;
  isCopilotOpen?: boolean;
  variant?: 'desktop' | 'mobile';
  onCloseMobile?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, onNavigate, onToggleCopilot, isCopilotOpen, variant = 'desktop', onCloseMobile }) => {
  const { sidebarCollapsed, toggleSidebar, user } = useApp();
  const mobile = variant === 'mobile';

  const groups: NavGroup[] = [
    {
      title: 'Clinical',
      items: [
        { tab: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard className="w-4 h-4" /> },
        { tab: 'inference', label: 'Inference', icon: <UploadCloud className="w-4 h-4" /> },
        { tab: 'batch', label: 'Batch Processing', icon: <Layers className="w-4 h-4" /> },
        { tab: 'doctor', label: 'DICOM Studio', icon: <UserCheck className="w-4 h-4" /> },
        { tab: 'patients', label: 'Patients', icon: <Users className="w-4 h-4" />, badge: '8.4K' },
      ],
    },
    {
      title: 'Intelligence',
      items: [
        { tab: 'knowledge', label: 'Knowledge Hub', icon: <BookOpen className="w-4 h-4" /> },
        { tab: 'history', label: 'History & Audit', icon: <History className="w-4 h-4" /> },
      ],
    },
    {
      title: 'AI & ML',
      items: [
        { tab: 'modelhub', label: 'Model Hub', icon: <Cpu className="w-4 h-4" /> },
        { tab: 'training', label: 'Training', icon: <GraduationCap className="w-4 h-4" /> },
        { tab: 'datasets', label: 'Datasets', icon: <Database className="w-4 h-4" /> },
        { tab: 'models', label: 'Benchmarks', icon: <BarChart3 className="w-4 h-4" /> },
        { tab: 'mlops', label: 'MLOps', icon: <Activity className="w-4 h-4" /> },
      ],
    },
    {
      title: 'System',
      items: [
        { tab: 'analytics', label: 'Analytics & GPU', icon: <Gauge className="w-4 h-4" /> },
        { tab: 'docs', label: 'Documentation', icon: <FileText className="w-4 h-4" /> },
        { tab: 'settings', label: 'Settings', icon: <Settings className="w-4 h-4" /> },
      ],
    },
  ];

  return (
    <aside
      className={cn(
        mobile
          ? 'fixed inset-y-0 left-0 z-50 w-64 flex flex-col glass-strong border-r border-slate-200/60 dark:border-slate-800/70 animate-slide-in-left lg:hidden'
          : 'hidden lg:flex fixed left-0 top-14 bottom-0 z-30 flex-col glass border-r border-slate-200/50 dark:border-slate-800/60 transition-all duration-300',
        !mobile && (sidebarCollapsed ? 'w-[4.5rem]' : 'w-60')
      )}
    >
      {mobile && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200/60 dark:border-slate-800/70">
          <span className="text-xs font-black text-slate-900 dark:text-white tracking-tight">
            MedVision <span className="text-gradient">AI</span>
          </span>
          <button onClick={onCloseMobile} className="p-1.5 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 cursor-pointer">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Collapse handle (desktop only) */}
      {!mobile && (
        <button
          onClick={toggleSidebar}
          className="absolute -right-3 top-16 w-6 h-6 rounded-full bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shadow flex items-center justify-center text-slate-400 hover:text-primary-500 transition-all cursor-pointer z-10"
          title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <ChevronLeft className={cn('w-3.5 h-3.5 transition-transform duration-300', sidebarCollapsed && 'rotate-180')} />
        </button>
      )}

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-5 scrollbar-none">
        {groups.map((group) => (
          <div key={group.title}>
            {!sidebarCollapsed && (
              <p className="px-3 mb-1.5 text-[9px] font-bold uppercase tracking-[0.18em] text-slate-400 dark:text-slate-500">
                {group.title}
              </p>
            )}
            <div className="space-y-0.5">
              {group.items.map((item) => {
                const active = activeTab === item.tab;
                return (
                  <button
                    key={item.tab}
                    onClick={() => onNavigate(item.tab)}
                    title={sidebarCollapsed ? item.label : undefined}
                    className={cn(
                      'w-full flex items-center gap-2.5 rounded-lg px-3 py-2 text-xs font-semibold transition-all cursor-pointer group relative',
                      sidebarCollapsed && !mobile && 'justify-center px-2',
                      active
                        ? 'bg-gradient-to-r from-indigo-600 to-violet-600 text-white shadow-md shadow-indigo-500/25'
                        : 'text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800/70 hover:text-slate-900 dark:hover:text-white'
                    )}
                  >
                    <span className={cn('shrink-0', active ? 'text-white' : 'text-slate-400 dark:text-slate-400 group-hover:text-primary-500 transition-colors')}>
                      {item.icon}
                    </span>
                    {(!sidebarCollapsed || mobile) && <span className="flex-1 text-left truncate">{item.label}</span>}
                    {(!sidebarCollapsed || mobile) && item.badge && (
                      <span className={cn('text-[9px] font-mono font-bold px-1.5 py-0.5 rounded-full', active ? 'bg-white/20 text-white' : 'bg-slate-100 dark:bg-slate-800 text-slate-500')}>
                        {item.badge}
                      </span>
                    )}
                    {active && (
                      <span className={cn('absolute left-0 top-1/2 -translate-y-1/2 w-1 h-5 rounded-r-full bg-white/90', sidebarCollapsed && 'left-0.5')} />
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        ))}

        {/* Copilot quick action */}
        {onToggleCopilot && (
          <div className={cn('pt-4 mt-4 border-t border-slate-200/60 dark:border-slate-800', sidebarCollapsed && 'px-0')}>
            <button
              onClick={onToggleCopilot}
              className={cn(
                'w-full flex items-center gap-2.5 rounded-lg px-3 py-2.5 text-xs font-bold transition-all cursor-pointer',
                sidebarCollapsed && !mobile && 'justify-center',
                isCopilotOpen
                  ? 'bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white shadow-md shadow-violet-500/25'
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-700'
              )}
              title={sidebarCollapsed ? 'AI Copilot' : undefined}
            >
              <Bot className="w-4 h-4 shrink-0" />
              {(!sidebarCollapsed || mobile) && <span className="flex-1 text-left">AI Copilot</span>}
              {(!sidebarCollapsed || mobile) && <HeartPulse className="w-3.5 h-3.5 opacity-60" />}
            </button>
          </div>
        )}
      </nav>

      {/* System health footer */}
      {(!sidebarCollapsed || mobile) && (
        <div className="p-3 border-t border-slate-200/60 dark:border-slate-800">
          <div className="rounded-xl bg-slate-100/70 dark:bg-slate-800/60 p-3 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold text-slate-500 dark:text-slate-400">SYSTEM HEALTH</span>
              <span className="flex items-center gap-1 text-[9px] font-mono text-emerald-500">
                <span className="live-dot w-1.5 h-1.5 rounded-full bg-emerald-500" />
                99.2%
              </span>
            </div>
            {[
              { label: 'GPU Pool', v: 72 },
              { label: 'Inference', v: 41 },
              { label: 'Storage', v: 78 },
            ].map((row) => (
              <div key={row.label} className="space-y-1">
                <div className="flex justify-between text-[9px] font-mono text-slate-400">
                  <span>{row.label}</span>
                  <span>{row.v}%</span>
                </div>
                <div className="h-1 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
                  <div
                    className={cn('h-full rounded-full transition-all duration-700', row.v > 75 ? 'bg-amber-500' : 'bg-emerald-500')}
                    style={{ width: `${row.v}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
          <p className="mt-2 px-1 text-[9px] text-slate-400 font-mono truncate">
            {user.name} · {user.role}
          </p>
        </div>
      )}
    </aside>
  );
};


