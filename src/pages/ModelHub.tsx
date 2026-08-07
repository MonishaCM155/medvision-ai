import React, { useEffect, useMemo, useState } from 'react';
import {
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid,
} from 'recharts';
import { Cpu, Layers, Boxes, ScanSearch, Sigma, Combine, Rocket, GitCompareArrows, CheckCircle2, Clock, HardDrive, Cpu as CpuIcon } from 'lucide-react';
import { api } from '../services/api';
import { HubModel, ModelTask } from '../types';
import { GlassCard, SectionHeader, LiveBadge } from '../components/ui/GlassCard';
import { cn } from '../utils/cn';

const TASK_FILTERS: { id: ModelTask | 'All'; label: string; icon: React.ReactNode }[] = [
  { id: 'All', label: 'All', icon: <Layers className="w-3.5 h-3.5" /> },
  { id: 'Classification', label: 'Classification', icon: <Cpu className="w-3.5 h-3.5" /> },
  { id: 'Detection', label: 'Detection', icon: <ScanSearch className="w-3.5 h-3.5" /> },
  { id: 'Segmentation', label: 'Segmentation', icon: <Boxes className="w-3.5 h-3.5" /> },
  { id: 'Regression', label: 'Regression', icon: <Sigma className="w-3.5 h-3.5" /> },
  { id: 'Ensemble', label: 'Ensemble', icon: <Combine className="w-3.5 h-3.5" /> },
];

const TOOLTIP_STYLE = {
  backgroundColor: '#0f172a',
  border: '1px solid #334155',
  borderRadius: '10px',
  fontSize: '11px',
  color: '#f8fafc',
};

const PROVENANCE_META: Record<HubModel['source'], { label: string; cls: string; title: string }> = {
  published: {
    label: 'PUBLISHED',
    cls: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/25',
    title: 'Metrics from a peer-reviewed paper or official benchmark',
  },
  estimated: {
    label: 'ESTIMATED',
    cls: 'bg-sky-500/10 text-sky-600 dark:text-sky-400 border-sky-500/25',
    title: 'Extrapolated from comparable published models — no direct paper',
  },
  synthetic: {
    label: 'DEMO',
    cls: 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/25',
    title: 'Internal demo value — not from a published benchmark',
  },
};

export const ModelHub: React.FC = () => {
  const [models, setModels] = useState<HubModel[]>([]);
  const [task, setTask] = useState<ModelTask | 'All'>('All');
  const [selected, setSelected] = useState<Set<string>>(new Set(['densenet121', 'efficientnet_b3', 'swin_b']));

  useEffect(() => {
    api.getModelHub().then(setModels);
  }, []);

  const filtered = useMemo(() => (task === 'All' ? models : models.filter((m) => m.task === task)), [models, task]);

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const compareData = useMemo(() => {
    const sel = models.filter((m) => selected.has(m.id));
    const pct = (v: number | undefined) => (v != null ? +((v as number) * 100).toFixed(1) : 0);
    return sel.map((m) => ({
      name: m.name.split(' ')[0],
      full: m.name,
      accuracy: m.accuracy ?? 0,
      precision: pct(m.precision),
      recall: pct(m.recall),
      specificity: pct(m.specificity),
      sensitivity: pct(m.sensitivity),
      f1: pct(m.f1),
      auroc: pct(m.auroc),
      latencyMs: m.latencyMs ?? 0,
    }));
  }, [models, selected]);

  const radarData = useMemo(() => {
    const sel = models.filter((m) => selected.has(m.id));
    const keys = ['accuracy', 'precision', 'recall', 'f1', 'auroc'] as const;
    const pct = (v: number | undefined) => (v != null ? +((v as number) * 100).toFixed(1) : 0);
    return keys.map((k) => {
      const row: Record<string, string | number> = { metric: k.toUpperCase() };
      sel.forEach((m) => {
        row[m.name] = k === 'accuracy' ? (m.accuracy ?? 0) : pct(m[k] as number | undefined);
      });
      return row;
    });
  }, [models, selected]);

  const selectedModels = models.filter((m) => selected.has(m.id));

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-black tracking-tight text-slate-900 dark:text-white">AI Model Hub</h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            {models.length} architectures across classification, detection, segmentation, traditional ML & ensembles
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 text-[10px] font-mono text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 border border-emerald-500/25 rounded-full px-2.5 py-1.5">
            <Rocket className="w-3.5 h-3.5" /> 6 deployed to production
          </span>
          <span className="inline-flex items-center gap-1.5 text-[10px] font-mono text-sky-600 dark:text-sky-400 bg-sky-500/10 border border-sky-500/25 rounded-full px-2.5 py-1.5">
            <CpuIcon className="w-3.5 h-3.5" /> Registry v2.4.1
          </span>
        </div>
      </div>

      {/* Task filter */}
      <div className="flex flex-wrap gap-1.5">
        {TASK_FILTERS.map((f) => (
          <button
            key={f.id}
            onClick={() => setTask(f.id)}
            className={cn(
              'inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[11px] font-semibold border transition-all cursor-pointer',
              task === f.id
                ? 'bg-indigo-600 text-white border-indigo-600 shadow-md shadow-indigo-500/25'
                : 'bg-white/70 dark:bg-slate-800/70 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:border-indigo-400'
            )}
          >
            {f.icon}
            {f.label}
            {f.id !== 'All' && (
              <span className={cn('font-mono text-[9px]', task === f.id ? 'text-white/80' : 'text-slate-400')}>
                {models.filter((m) => m.task === f.id).length}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Provenance legend */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[10px] text-slate-500 dark:text-slate-400">
        <span className="font-semibold text-slate-600 dark:text-slate-300">Metric provenance:</span>
        {(['published', 'estimated', 'synthetic'] as const).map((s) => (
          <span key={s} className="inline-flex items-center gap-1.5">
            <span className={cn('w-2 h-2 rounded-full border', PROVENANCE_META[s].cls.split(' ')[0])} />
            {PROVENANCE_META[s].label} — {PROVENANCE_META[s].title.toLowerCase()}
          </span>
        ))}
      </div>

      {/* Model cards */}
      {models.length === 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="skeleton h-56" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {filtered.map((m) => {
            const isSel = selected.has(m.id);
            return (
              <GlassCard
                key={m.id}
                hover
                gradient={isSel}
                className={cn('relative flex flex-col animate-fade-in-up !p-4', isSel && 'ring-2 ring-primary-400/70 dark:ring-primary-500/60')}
              >
                <button
                  onClick={() => toggleSelect(m.id)}
                  className={cn(
                    'absolute top-3 right-3 w-5 h-5 rounded-md border flex items-center justify-center transition-all cursor-pointer',
                    isSel ? 'bg-primary-600 border-primary-600 text-white' : 'border-slate-300 dark:border-slate-600 text-transparent hover:border-primary-400'
                  )}
                  title={isSel ? 'Remove from comparison' : 'Add to comparison'}
                >
                  <CheckCircle2 className="w-4 h-4" />
                </button>

                <div className="flex items-center gap-2.5">
                  <span
                    className="w-9 h-9 rounded-xl flex items-center justify-center text-white text-[13px] font-black shadow-md"
                    style={{ background: `linear-gradient(135deg, ${m.color}, ${m.color}88)` }}
                  >
                    {m.name[0]}
                  </span>
                  <div className="min-w-0">
                    <h3 className="text-xs font-bold text-slate-900 dark:text-white truncate">{m.name}</h3>
                    <div className="flex items-center gap-1.5 mt-0.5">
                      <span className="text-[9px] font-mono font-bold px-1.5 py-0.5 rounded-full border border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400">
                        {m.task}
                      </span>
                      {m.deployed && (
                        <span className="text-[9px] font-mono font-bold px-1.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/25">
                          PROD
                        </span>
                      )}
                      <span
                        title={`${PROVENANCE_META[m.source].title}${m.reference ? ` — ${m.reference}` : ''}`}
                        className={cn('text-[9px] font-mono font-bold px-1.5 py-0.5 rounded-full border', PROVENANCE_META[m.source].cls)}
                      >
                        {PROVENANCE_META[m.source].label}
                      </span>
                    </div>
                  </div>
                </div>

                <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed mt-2.5 line-clamp-2">{m.description}</p>

                <div className="grid grid-cols-2 gap-1.5 mt-3 text-center">
                  {[
                    { label: 'Accuracy', value: `${m.accuracy ?? '—'}%`, color: 'text-emerald-600 dark:text-emerald-400' },
                    { label: 'F1', value: m.f1 != null ? m.f1.toFixed(3) : '—', color: 'text-indigo-600 dark:text-indigo-400' },
                    { label: 'AUROC', value: m.auroc != null ? m.auroc.toFixed(3) : '—', color: 'text-sky-600 dark:text-sky-400' },
                    { label: 'Latency', value: m.latencyMs != null ? `${m.latencyMs}ms` : '—', color: 'text-amber-600 dark:text-amber-400' },
                    { label: 'Sens', value: m.sensitivity != null ? m.sensitivity.toFixed(3) : '—', color: 'text-rose-600 dark:text-rose-400' },
                    { label: 'Spec', value: m.specificity != null ? m.specificity.toFixed(3) : '—', color: 'text-teal-600 dark:text-teal-400' },
                  ].map((s) => (
                    <div key={s.label} className="bg-slate-50 dark:bg-slate-800/60 rounded-lg py-1.5">
                      <p className="text-[8px] font-mono uppercase tracking-widest text-slate-400">{s.label}</p>
                      <p className={cn('text-[11px] font-bold font-mono', s.color)}>{s.value}</p>
                    </div>
                  ))}
                </div>

                <div className="flex items-center gap-3 mt-2.5 text-[9px] font-mono text-slate-400">
                  <span className="inline-flex items-center gap-1"><HardDrive className="w-3 h-3" />{m.size}</span>
                  <span className="inline-flex items-center gap-1"><CpuIcon className="w-3 h-3" />{m.parameters}</span>
                  <span className="inline-flex items-center gap-1 truncate"><Clock className="w-3 h-3" />{m.framework}</span>
                </div>
              </GlassCard>
            );
          })}
        </div>
      )}

      {/* Comparison workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <GlassCard gradient>
          <SectionHeader
            icon={<GitCompareArrows className="w-4 h-4 text-primary-500" />}
            title="Head-to-Head Comparison"
            subtitle={selectedModels.length ? `Comparing ${selectedModels.map((m) => m.name.split(' ')[0]).join(' vs ')}` : 'Select models using the checkboxes'}
            right={<LiveBadge label={`${selected.size} selected`} color="indigo" />}
          />
          {compareData.length < 2 ? (
            <p className="text-xs text-slate-400 py-12 text-center">Select at least 2 models to compare metrics.</p>
          ) : (
            <div className="h-72 mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={compareData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                  <XAxis dataKey="name" stroke="var(--text-soft)" fontSize={11} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 100]} stroke="var(--text-soft)" fontSize={11} axisLine={false} tickLine={false} unit="%" />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="accuracy" name="Accuracy %" radius={[4, 4, 0, 0]} fill="#6366f1" />
                  <Bar dataKey="sensitivity" name="Sensitivity %" radius={[4, 4, 0, 0]} fill="#ec4899" />
                  <Bar dataKey="specificity" name="Specificity %" radius={[4, 4, 0, 0]} fill="#14b8a6" />
                  <Bar dataKey="f1" name="F1 %" radius={[4, 4, 0, 0]} fill="#06b6d4" />
                  <Bar dataKey="auroc" name="AUROC %" radius={[4, 4, 0, 0]} fill="#10b981" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </GlassCard>

        <GlassCard>
          <SectionHeader icon={<Boxes className="w-4 h-4 text-fuchsia-500" />} title="Radar Profile" subtitle="Normalized performance across 5 metrics" />
          {radarData.length === 0 || selected.size < 2 ? (
            <p className="text-xs text-slate-400 py-12 text-center">Radar profile appears once 2+ models are selected.</p>
          ) : (
            <div className="h-72 mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData} outerRadius="68%">
                  <PolarGrid stroke="var(--border)" />
                  <PolarAngleAxis dataKey="metric" stroke="var(--text-muted)" fontSize={10} />
                  <PolarRadiusAxis domain={[0, 100]} stroke="var(--border-strong)" fontSize={8} />
                  {selectedModels.map((m, i) => (
                    <Radar key={m.id} name={m.name.split(' ')[0]} dataKey={m.name} stroke={m.color} fill={m.color} fillOpacity={0.18} strokeWidth={2} />
                  ))}
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          )}
        </GlassCard>
      </div>
    </div>
  );
};
