import React, { useEffect, useState } from 'react';
import {
  ResponsiveContainer, AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip,
  CartesianGrid, Cell, PieChart, Pie, Legend,
} from 'recharts';
import {
  ScanLine, FileClock, Siren, Users, Target, Zap, Cpu, Gauge, BellRing,
  ArrowRight, Bot, ShieldCheck, Database, TrendingUp, BrainCircuit,
  Activity, Server, AlertTriangle, Ban, MemoryStick, RefreshCw, FlaskConical,
} from 'lucide-react';
import { useApp } from '../contexts/AppContext';
import { api } from '../services/api';
import { DashboardStats, ActivityEvent, MonitoringSnapshot } from '../types';
import { ACTIVITY_FEED } from '../data/mockEnterprise';
import { StatCard } from '../components/ui/StatCard';
import { GlassCard, SectionHeader, LiveBadge, SeverityPill } from '../components/ui/GlassCard';
import { cn } from '../utils/cn';

const TOOLTIP_STYLE = {
  backgroundColor: '#0f172a',
  border: '1px solid #334155',
  borderRadius: '10px',
  fontSize: '11px',
  color: '#f8fafc',
};

const ACTIVITY_ICON: Record<ActivityEvent['type'], React.ReactNode> = {
  scan: <ScanLine className="w-3.5 h-3.5" />,
  report: <FileClock className="w-3.5 h-3.5" />,
  model: <Cpu className="w-3.5 h-3.5" />,
  system: <Gauge className="w-3.5 h-3.5" />,
  patient: <Users className="w-3.5 h-3.5" />,
  training: <BrainCircuit className="w-3.5 h-3.5" />,
};

export const Dashboard: React.FC<{ onNavigate: (tab: string) => void }> = ({ onNavigate }) => {
  const { user, pushNotification } = useApp();
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [activity] = useState<ActivityEvent[]>(ACTIVITY_FEED);
  const [monitor, setMonitor] = useState<MonitoringSnapshot | null>(null);

  useEffect(() => {
    let mounted = true;
    api.getDashboardStats().then((s) => {
      if (mounted) setStats(s);
    });
    return () => {
      mounted = false;
    };
  }, []);

  // Poll live server telemetry every 5s (falls back to offline snapshot when the API is down)
  useEffect(() => {
    let mounted = true;
    const tick = () => {
      api.getMonitoring().then((m) => {
        if (mounted) setMonitor(m);
      });
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, []);

  const today = new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' });

  if (!stats) {
    return (
      <div className="space-y-6 animate-fade-in">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="skeleton h-28" />
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 skeleton h-80" />
          <div className="skeleton h-80" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Greeting */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-black tracking-tight text-slate-900 dark:text-white">
            Good {new Date().getHours() < 12 ? 'morning' : new Date().getHours() < 18 ? 'afternoon' : 'evening'} 👋
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            {today} · {user.department} · KPI dashboard shows <strong className="font-semibold text-slate-600 dark:text-slate-300">simulated demo data</strong>
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onNavigate('inference')}
            className="btn-gradient text-white text-xs font-bold px-4 py-2.5 rounded-xl flex items-center gap-2 cursor-pointer"
          >
            <ScanLine className="w-4 h-4" />
            New Scan
          </button>
          <button
            onClick={() => onNavigate('patients')}
            className="text-xs font-bold px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:border-primary-400 transition-colors flex items-center gap-2 cursor-pointer bg-white/60 dark:bg-slate-800/60"
          >
            <Users className="w-4 h-4" />
            Patients
          </button>
        </div>
      </div>

      {/* AI status banner */}
      <div className="ring-gradient surface-card p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-11 h-11 rounded-2xl bg-gradient-to-br from-emerald-500 to-teal-600 text-white flex items-center justify-center shadow-lg shadow-emerald-500/30">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-slate-900 dark:text-white">MedVision AI Core</span>
              <LiveBadge label="Online" />
            </div>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              DenseNet-121 + EfficientNet-B3 + Swin-B ensemble · Gemini 3.6 Flash report synthesis · {stats.activeModels} production models
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4 text-[11px] font-mono">
          <span className="text-slate-500 dark:text-slate-400">Throughput <strong className="text-emerald-500">{stats.avgLatencyMs}ms</strong></span>
          <span className="hidden sm:inline text-slate-300 dark:text-slate-600">|</span>
          <span className="text-slate-500 dark:text-slate-400">Queue <strong className="text-amber-500">{stats.queuedPredictions}</strong></span>
          <span className="hidden sm:inline text-slate-300 dark:text-slate-600">|</span>
          <span className="text-slate-500 dark:text-slate-400">Uptime <strong className="text-sky-500">99.98%</strong></span>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Today's Scans" value={stats.todayScans.toLocaleString()} icon={<ScanLine className="w-5 h-5" />} trend="+18.4% vs yesterday" trendDirection="up" accent="from-indigo-500 to-violet-500" />
        <StatCard label="Pending Reports" value={stats.pendingReports.toLocaleString()} icon={<FileClock className="w-5 h-5" />} trend="4 need urgent review" trendDirection="down" trendPositive={false} accent="from-amber-500 to-orange-500" />
        <StatCard label="Emergency Cases" value={stats.emergencyCases.toLocaleString()} icon={<Siren className="w-5 h-5" />} trend="Critical priority" trendDirection="up" trendPositive={false} accent="from-rose-500 to-red-600" />
        <StatCard label="Total Patients" value={stats.totalPatients.toLocaleString()} icon={<Users className="w-5 h-5" />} trend="+312 this month" trendDirection="up" accent="from-emerald-500 to-teal-600" />
        <StatCard label="AI Accuracy" value={stats.aiAccuracy.toFixed(1)} suffix="%" icon={<Target className="w-5 h-5" />} trend="+0.6% calibrated" trendDirection="up" accent="from-sky-500 to-cyan-500" />
        <StatCard label="Mean Latency" value={stats.avgLatencyMs.toFixed(1)} suffix="ms" icon={<Zap className="w-5 h-5" />} trend="TensorRT FP16" trendDirection="down" accent="from-fuchsia-500 to-pink-500" />
        <StatCard label="Active Models" value={String(stats.activeModels)} icon={<Cpu className="w-5 h-5" />} trend="2 in staging" trendDirection="flat" accent="from-violet-500 to-purple-600" />
        <StatCard label="Prediction Queue" value={String(stats.queuedPredictions)} icon={<Gauge className="w-5 h-5" />} trend="Auto-scaling pool" trendDirection="flat" accent="from-slate-500 to-slate-700" />
      </div>

      {/* Charts row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Weekly trend */}
        <GlassCard className="lg:col-span-2" gradient>
          <SectionHeader
            icon={<TrendingUp className="w-4 h-4 text-primary-500" />}
            title="Weekly Clinical Throughput"
            subtitle="Scans processed vs reports generated"
            right={<LiveBadge label="7-Day" color="indigo" />}
          />
          <div className="h-64 mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={stats.weeklyTrend} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                <defs>
                  <linearGradient id="gradScans" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#6366f1" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#6366f1" stopOpacity={0.02} />
                  </linearGradient>
                  <linearGradient id="gradReports" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#06b6d4" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#06b6d4" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="day" stroke="var(--text-soft)" fontSize={11} axisLine={false} tickLine={false} />
                <YAxis stroke="var(--text-soft)" fontSize={11} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Area type="monotone" dataKey="scans" name="Scans" stroke="#6366f1" strokeWidth={2.5} fill="url(#gradScans)" />
                <Area type="monotone" dataKey="reports" name="Reports" stroke="#06b6d4" strokeWidth={2.5} fill="url(#gradReports)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </GlassCard>

        {/* Severity distribution donut */}
        <GlassCard>
          <SectionHeader
            icon={<Siren className="w-4 h-4 text-rose-500" />}
            title="Severity Distribution"
            subtitle="Radiographic severity grades"
          />
          <div className="h-52 mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={stats.severityDistribution}
                  dataKey="value"
                  nameKey="name"
                  cx="50%" cy="50%"
                  innerRadius={52}
                  outerRadius={78}
                  paddingAngle={3}
                  strokeWidth={0}
                >
                  {stats.severityDistribution.map((s) => (
                    <Cell key={s.name} fill={s.color} />
                  ))}
                </Pie>
                <Tooltip contentStyle={TOOLTIP_STYLE} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-4 gap-1.5 mt-1">
            {stats.severityDistribution.map((s) => (
              <div key={s.name} className="text-center">
                <span className="block w-2 h-2 rounded-full mx-auto" style={{ background: s.color }} />
                <span className="text-[9px] font-mono text-slate-500 dark:text-slate-400">{s.name}</span>
                <span className="text-[11px] font-bold text-slate-800 dark:text-slate-200 block">{s.value}%</span>
              </div>
            ))}
          </div>
        </GlassCard>
      </div>

      {/* Charts row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Disease stats */}
        <GlassCard className="lg:col-span-2">
          <SectionHeader
            icon={<Database className="w-4 h-4 text-sky-500" />}
            title="Disease Statistics"
            subtitle="Detections across all studies (30 days)"
            right={
              <button onClick={() => onNavigate('analytics')} className="text-[11px] font-semibold text-primary-600 dark:text-primary-400 hover:underline flex items-center gap-1 cursor-pointer">
                Full analytics <ArrowRight className="w-3 h-3" />
              </button>
            }
          />
          <div className="h-64 mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats.diseaseStats} layout="vertical" margin={{ top: 0, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                <XAxis type="number" stroke="var(--text-soft)" fontSize={11} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="disease" stroke="var(--text)" fontSize={11} width={110} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: 'rgba(99,102,241,0.06)' }} />
                <Bar dataKey="count" name="Detections" radius={[0, 6, 6, 0]} barSize={16}>
                  {stats.diseaseStats.map((d) => (
                    <Cell key={d.disease} fill={d.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </GlassCard>

        {/* Live activity feed */}
        <GlassCard className="flex flex-col">
          <SectionHeader
            icon={<BellRing className="w-4 h-4 text-amber-500" />}
            title="Live Activity Feed"
            right={<LiveBadge label="Live" />}
          />
          <div className="flex-1 space-y-1 mt-3 max-h-[19rem] overflow-y-auto pr-1">
            {activity.map((a) => (
              <div key={a.id} className="flex gap-2.5 p-2 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors group">
                <span className={cn(
                  'w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-0.5 border',
                  a.type === 'scan' && 'bg-indigo-500/10 text-indigo-500 border-indigo-500/20',
                  a.type === 'report' && 'bg-sky-500/10 text-sky-500 border-sky-500/20',
                  a.type === 'model' && 'bg-violet-500/10 text-violet-500 border-violet-500/20',
                  a.type === 'system' && 'bg-slate-500/10 text-slate-500 border-slate-500/20',
                  a.type === 'patient' && 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20',
                  a.type === 'training' && 'bg-fuchsia-500/10 text-fuchsia-500 border-fuchsia-500/20',
                )}>
                  {ACTIVITY_ICON[a.type]}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="text-[11px] leading-snug text-slate-700 dark:text-slate-200">
                    <strong className="font-semibold">{a.actor}</strong> {a.action}
                  </p>
                  <p className="text-[10px] text-slate-500 dark:text-slate-400 truncate">{a.detail}</p>
                </div>
                <div className="flex flex-col items-end gap-1 shrink-0">
                  <span className="text-[9px] font-mono text-slate-400">{a.time}</span>
                  {a.severity && <SeverityPill level={a.severity} />}
                </div>
              </div>
            ))}
          </div>
        </GlassCard>
      </div>

      {/* Charts row 3 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Monthly trend */}
        <GlassCard>
          <SectionHeader icon={<TrendingUp className="w-4 h-4 text-emerald-500" />} title="Monthly Scan Volume" subtitle="Last 6 months" />
          <div className="h-52 mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats.monthlyTrend} margin={{ top: 5, right: 5, left: -15, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="month" stroke="var(--text-soft)" fontSize={10} axisLine={false} tickLine={false} />
                <YAxis stroke="var(--text-soft)" fontSize={10} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: 'rgba(16,185,129,0.06)' }} />
                <Bar dataKey="scans" name="Scans" radius={[6, 6, 0, 0]} fill="#10b981" barSize={22} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </GlassCard>

        {/* System health */}
        <GlassCard>
          <SectionHeader icon={<Gauge className="w-4 h-4 text-cyan-500" />} title="Model & System Health" subtitle="Resource utilization" />
          <div className="space-y-3 mt-4">
            {stats.systemHealth.map((h) => (
              <div key={h.component} className="space-y-1">
                <div className="flex justify-between text-[11px]">
                  <span className="text-slate-600 dark:text-slate-300 font-medium">{h.component}</span>
                  <span className={cn('font-mono font-bold', h.status === 'critical' ? 'text-rose-500' : h.status === 'warning' ? 'text-amber-500' : 'text-emerald-500')}>
                    {h.usage}%
                  </span>
                </div>
                <div className="h-1.5 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                  <div
                    className={cn('h-full rounded-full transition-all duration-1000', h.status === 'critical' ? 'bg-rose-500' : h.status === 'warning' ? 'bg-amber-500' : 'bg-gradient-to-r from-emerald-500 to-teal-500')}
                    style={{ width: `${h.usage}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </GlassCard>

        {/* Prediction queue */}
        <GlassCard>
          <SectionHeader
            icon={<Gauge className="w-4 h-4 text-amber-500" />}
            title="Prediction Queue"
            right={
              <button
                onClick={() => pushNotification({ kind: 'success', title: 'Queue flushed', body: 'All queued predictions completed successfully. (Simulated demo action — no accounts or permissions are enforced.)' })}
                className="text-[10px] font-semibold inline-flex items-center gap-1 text-slate-400 hover:text-primary-500 cursor-pointer transition-colors"
                title="Flush the simulated inference queue (public demo action)"
              >
                Flush
              </button>
            }
          />
          <div className="space-y-2.5 mt-4">
            {stats.queue.map((q) => (
              <div key={q.id} className="rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-100 dark:border-slate-700/60 p-3 space-y-1.5 hover:border-primary-300 dark:hover:border-primary-700 transition-colors">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="font-bold text-slate-800 dark:text-slate-100 font-mono">{q.patient}</span>
                  <span className="text-[9px] font-mono text-slate-400">{q.model}</span>
                </div>
                <p className="text-[10px] text-slate-500 dark:text-slate-400 truncate">{q.study}</p>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-1 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-indigo-500 to-violet-500 rounded-full transition-all duration-700" style={{ width: `${q.progress}%` }} />
                  </div>
                  <span className="text-[9px] font-mono text-slate-400">ETA {q.etaSec}s</span>
                </div>
              </div>
            ))}
            <div className="flex items-center gap-2 text-[10px] text-slate-400 mt-2 pt-2 border-t border-slate-100 dark:border-slate-800">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-500" />
              <span>All predictions verified against model registry v2.7.0</span>
            </div>
          </div>
        </GlassCard>
      </div>

      {/* Live server monitor — real telemetry from GET /api/monitoring */}
      <GlassCard>
        <SectionHeader
          icon={<Activity className="w-4 h-4 text-emerald-500" />}
          title="Live System Monitor"
          subtitle="Real-time telemetry from the MedVision server — not simulated"
          right={
            <span className="inline-flex items-center gap-1.5 text-[10px] font-mono text-emerald-600 dark:text-emerald-400">
              <RefreshCw className="w-3 h-3 animate-spin-slow" />
              {monitor ? `polling · ${new Date(monitor.timestamp).toLocaleTimeString()}` : 'connecting…'}
            </span>
          }
        />
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-2 mt-4">
          {[
            { label: 'Service', value: monitor ? monitor.service : '—', sub: monitor ? `v${monitor.version}` : '', icon: <Server className="w-3.5 h-3.5" />, color: 'text-indigo-500' },
            { label: 'Uptime', value: monitor ? formatUptime(monitor.uptimeSec) : '—', sub: 'since boot', icon: <Activity className="w-3.5 h-3.5" />, color: 'text-emerald-500' },
            { label: 'Requests', value: monitor ? monitor.requests.toLocaleString() : '—', sub: `${monitor?.requestsPerMinute ?? 0}/min`, icon: <Zap className="w-3.5 h-3.5" />, color: 'text-sky-500' },
            { label: 'Errors', value: String(monitor?.errors ?? '—'), sub: monitor?.errors ? 'needs review' : 'none', icon: <AlertTriangle className="w-3.5 h-3.5" />, color: monitor?.errors ? 'text-rose-500' : 'text-slate-400' },
            { label: 'Rejected Images', value: String(monitor?.rejectedImages ?? '—'), sub: 'safety gate', icon: <Ban className="w-3.5 h-3.5" />, color: 'text-amber-500' },
            { label: 'Predictions', value: String(monitor?.predictions ?? '—'), sub: monitor?.avgInferenceMs ? `${monitor.avgInferenceMs.toFixed(0)}ms avg` : 'no load', icon: <BrainCircuit className="w-3.5 h-3.5" />, color: 'text-violet-500' },
            { label: 'Public Mode', value: 'No login', sub: 'open research platform', icon: <FlaskConical className="w-3.5 h-3.5" />, color: 'text-fuchsia-500' },
            { label: 'Memory', value: monitor ? `${monitor.memory.rssMb.toFixed(0)}MB` : '—', sub: `heap ${monitor?.memory.heapUsedMb.toFixed(0) ?? 0}MB`, icon: <MemoryStick className="w-3.5 h-3.5" />, color: 'text-teal-500' },
          ].map((t) => (
            <div key={t.label} className="rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-100 dark:border-slate-700/60 p-2.5 space-y-1">
              <div className="flex items-center gap-1.5 text-[9px] font-mono uppercase tracking-widest text-slate-400">
                <span className={t.color}>{t.icon}</span>
                {t.label}
              </div>
              <p className="text-sm font-black font-mono text-slate-900 dark:text-white truncate">{t.value}</p>
              <p className="text-[9px] font-mono text-slate-400 truncate">{t.sub}</p>
            </div>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 mt-3 pt-3 border-t border-slate-100 dark:border-slate-800 text-[10px] font-mono text-slate-500 dark:text-slate-400">
          <span className="inline-flex items-center gap-1.5">
            <span className={cn('w-1.5 h-1.5 rounded-full', monitor?.status === 'offline' ? 'bg-rose-500' : 'bg-emerald-500 animate-pulse')} />
            API {monitor?.status === 'offline' ? 'OFFLINE' : 'HEALTHY'}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <BrainCircuit className="w-3 h-3" />
            Engine: {monitor ? `${monitor.engine.status.toUpperCase()} · ${monitor.engine.source} · ${monitor.engine.device}` : '—'}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <Cpu className="w-3 h-3" />
            CPU {monitor ? `${(monitor.cpu.user + monitor.cpu.system).toFixed(0)}%` : '—'}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <Database className="w-3 h-3" />
            Model {monitor?.modelVersion ?? '—'} · Storage {monitor?.storage ?? '—'}
          </span>
        </div>
      </GlassCard>
    </div>
  );
};

function formatUptime(sec: number): string {
  if (sec < 60) return `${sec}s`;
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}
