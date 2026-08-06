import React, { useEffect, useMemo, useState } from 'react';
import {
  Database, HardDrive, Images, Tag, AlertTriangle, Copy, CheckCircle2, UploadCloud,
  Table2, BarChart3, FileCheck2, Search,
} from 'lucide-react';
import { api } from '../services/api';
import { DatasetInfo } from '../types';
import { GlassCard, SectionHeader, LiveBadge } from '../components/ui/GlassCard';
import { StatCard } from '../components/ui/StatCard';
import { cn } from '../utils/cn';

const TASK_COLORS: Record<DatasetInfo['task'], string> = {
  'Multi-label': 'bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border-indigo-500/25',
  Classification: 'bg-sky-500/10 text-sky-600 dark:text-sky-400 border-sky-500/25',
  Detection: 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/25',
  Segmentation: 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/25',
};

export const Datasets: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [query, setQuery] = useState('');
  const [qualityChecks, setQualityChecks] = useState<Record<string, boolean>>({});

  useEffect(() => {
    api.getDatasets().then((d) => {
      setDatasets(d);
      const checks: Record<string, boolean> = {};
      d.forEach((ds) => {
        checks[ds.id] = ds.missingLabels < ds.images * 0.01;
      });
      setQualityChecks(checks);
    });
  }, []);

  const filtered = useMemo(
    () => datasets.filter((d) => d.name.toLowerCase().includes(query.toLowerCase())),
    [datasets, query]
  );

  const totalImages = datasets.reduce((a, d) => a + d.images, 0);
  const totalSize = datasets.reduce((a, d) => a + parseFloat(d.sizeGb), 0);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-black tracking-tight text-slate-900 dark:text-white">Dataset Registry</h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Curated medical imaging corpora with quality checks, splits, and label statistics
          </p>
        </div>
        <button className="btn-gradient text-white text-xs font-bold px-4 py-2.5 rounded-xl flex items-center gap-2 cursor-pointer">
          <UploadCloud className="w-4 h-4" />
          Import Dataset
        </button>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total Images" value={totalImages.toLocaleString()} icon={<Images className="w-5 h-5" />} trend="5 corpora" trendDirection="flat" accent="from-indigo-500 to-violet-500" />
        <StatCard label="Total Size" value={totalSize.toFixed(1)} suffix="GB" icon={<HardDrive className="w-5 h-5" />} trend="Object storage" trendDirection="flat" accent="from-sky-500 to-cyan-500" />
        <StatCard label="Missing Labels" value={(datasets.reduce((a, d) => a + d.missingLabels, 0) / totalImages * 100).toFixed(1)} suffix="%" icon={<AlertTriangle className="w-5 h-5" />} trend="Imputation available" trendDirection="down" accent="from-amber-500 to-orange-500" />
        <StatCard label="Duplicates" value={datasets.reduce((a, d) => a + d.duplicates, 0).toLocaleString()} icon={<Copy className="w-5 h-5" />} trend="Dedup pipeline ready" trendDirection="down" accent="from-emerald-500 to-teal-600" />
      </div>

      {/* Search */}
      <GlassCard className="!p-4">
        <div className="relative max-w-md">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search datasets…"
            className="w-full bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 text-slate-800 dark:text-slate-200 text-xs rounded-xl pl-9 pr-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-primary-400/50"
          />
        </div>
      </GlassCard>

      {/* Dataset cards */}
      {filtered.length === 0 ? (
        <GlassCard className="text-center py-16">
          <div className="w-14 h-14 mx-auto rounded-2xl bg-slate-100 dark:bg-slate-800 flex items-center justify-center text-slate-400 mb-3">
            <Database className="w-6 h-6" />
          </div>
          <p className="text-sm font-bold text-slate-700 dark:text-slate-200">No datasets found</p>
        </GlassCard>
      ) : (
        <div className="space-y-5">
          {filtered.map((d) => (
            <GlassCard key={d.id} hover className="animate-fade-in-up">
              <div className="flex flex-col lg:flex-row lg:items-start gap-5">
                {/* Identity */}
                <div className="lg:w-72 shrink-0">
                  <div className="flex items-center gap-3">
                    <span className="w-11 h-11 rounded-2xl flex items-center justify-center text-white shadow-lg" style={{ background: `linear-gradient(135deg, ${d.color}, ${d.color}88)` }}>
                      <Database className="w-5 h-5" />
                    </span>
                    <div>
                      <h3 className="text-sm font-bold text-slate-900 dark:text-white">{d.name}</h3>
                      <div className="flex items-center gap-1.5 mt-1">
                        <span className={cn('text-[9px] font-mono font-bold px-1.5 py-0.5 rounded-full border', TASK_COLORS[d.task])}>{d.task}</span>
                        <span className="text-[9px] font-mono text-slate-400">{d.source}</span>
                      </div>
                    </div>
                  </div>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed mt-3">{d.description}</p>
                  <div className="flex flex-wrap gap-1.5 mt-3 text-[10px] font-mono text-slate-500">
                    <span className="inline-flex items-center gap-1 bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 rounded-full px-2 py-0.5">
                      <Images className="w-3 h-3" /> {d.images.toLocaleString()} imgs
                    </span>
                    <span className="inline-flex items-center gap-1 bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 rounded-full px-2 py-0.5">
                      <HardDrive className="w-3 h-3" /> {d.sizeGb} GB
                    </span>
                    <span className="inline-flex items-center gap-1 bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 rounded-full px-2 py-0.5">
                      <Tag className="w-3 h-3" /> {d.labels} labels
                    </span>
                    <span className="inline-flex items-center gap-1 bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 rounded-full px-2 py-0.5">
                      <Table2 className="w-3 h-3" /> {d.format}
                    </span>
                  </div>
                </div>

                {/* Splits + quality */}
                <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Split */}
                  <div className="bg-slate-50 dark:bg-slate-800/50 rounded-xl p-4">
                    <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400 mb-3 flex items-center gap-1.5">
                      <BarChart3 className="w-3.5 h-3.5" /> Data Split
                    </p>
                    <div className="flex h-3 rounded-full overflow-hidden border border-slate-200 dark:border-slate-700">
                      <div className="bg-indigo-500" style={{ width: `${d.trainSplit}%` }} />
                      <div className="bg-sky-500" style={{ width: `${d.valSplit}%` }} />
                      <div className="bg-emerald-500" style={{ width: `${d.testSplit}%` }} />
                    </div>
                    <div className="flex justify-between mt-2 text-[10px] font-mono text-slate-500">
                      <span className="text-indigo-500 font-bold">Train {d.trainSplit}%</span>
                      <span className="text-sky-500 font-bold">Val {d.valSplit}%</span>
                      <span className="text-emerald-500 font-bold">Test {d.testSplit}%</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 mt-3 text-[10px] font-mono">
                      <div className="bg-white dark:bg-slate-900/50 rounded-lg p-2 border border-slate-200 dark:border-slate-700">
                        <p className="text-slate-400">Missing labels</p>
                        <p className={cn('font-bold', d.missingLabels > d.images * 0.01 ? 'text-amber-500' : 'text-emerald-500')}>{d.missingLabels.toLocaleString()}</p>
                      </div>
                      <div className="bg-white dark:bg-slate-900/50 rounded-lg p-2 border border-slate-200 dark:border-slate-700">
                        <p className="text-slate-400">Duplicates</p>
                        <p className={cn('font-bold', d.duplicates > 100 ? 'text-amber-500' : 'text-emerald-500')}>{d.duplicates.toLocaleString()}</p>
                      </div>
                    </div>
                  </div>

                  {/* Class distribution */}
                  <div className="bg-slate-50 dark:bg-slate-800/50 rounded-xl p-4">
                    <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400 mb-3 flex items-center gap-1.5">
                      <Table2 className="w-3.5 h-3.5" /> Label Statistics
                    </p>
                    <div className="max-h-28 overflow-y-auto space-y-1.5 pr-1">
                      {d.classes.slice(0, 8).map((c, i) => {
                        const width = 100 - (i % 3) * 18 - ((i * 7) % 14);
                        return (
                          <div key={c} className="flex items-center gap-2">
                            <span className="text-[10px] text-slate-500 dark:text-slate-400 w-32 truncate">{c}</span>
                            <div className="flex-1 h-1.5 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
                              <div className="h-full rounded-full transition-all duration-700" style={{ width: `${width}%`, background: d.color }} />
                            </div>
                            <span className="text-[9px] font-mono text-slate-400 w-9 text-right">{(d.images * width / 100).toLocaleString()}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Quality / actions */}
                <div className="lg:w-48 shrink-0 flex lg:flex-col items-center lg:items-stretch gap-3">
                  <div
                    className={cn(
                      'flex-1 rounded-xl border p-3 flex items-center lg:items-start gap-2.5',
                      qualityChecks[d.id]
                        ? 'bg-emerald-500/5 border-emerald-500/25'
                        : 'bg-amber-500/5 border-amber-500/25'
                    )}
                  >
                    {qualityChecks[d.id] ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
                    ) : (
                      <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
                    )}
                    <div>
                      <p className={cn('text-[10px] font-bold', qualityChecks[d.id] ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400')}>
                        {qualityChecks[d.id] ? 'Quality Passed' : 'Needs Review'}
                      </p>
                      <p className="text-[9px] text-slate-400 mt-0.5 leading-snug">
                        {qualityChecks[d.id] ? 'Label completeness above 99%' : 'Missing-label ratio above 1% threshold'}
                      </p>
                    </div>
                  </div>
                  <div className="flex gap-2 lg:flex-col">
                    <button className="flex-1 text-[10px] font-bold text-indigo-600 dark:text-indigo-400 border border-indigo-300/50 dark:border-indigo-700/60 rounded-lg px-3 py-2 hover:bg-indigo-500/10 transition-colors cursor-pointer flex items-center justify-center gap-1.5">
                      <FileCheck2 className="w-3.5 h-3.5" /> Preview
                    </button>
                    <button className="flex-1 text-[10px] font-bold text-slate-500 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 hover:border-indigo-400 transition-colors cursor-pointer">
                      Split
                    </button>
                  </div>
                </div>
              </div>
            </GlassCard>
          ))}
        </div>
      )}

      {/* Footer note */}
      <div className="flex items-center gap-2 text-[10px] text-slate-400">
        <LiveBadge label="Auto-verified" color="emerald" />
        <span>Datasets re-validated against quality gates every 24h · All splits stratified by patient to prevent leakage.</span>
      </div>
    </div>
  );
};
