import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend, AreaChart, Area,
} from 'recharts';
import {
  GraduationCap, Play, Pause, RotateCcw, Cpu, Gauge, MemoryStick,
  Download, Settings2, Rocket, Flame, Flag, Save,
} from 'lucide-react';
import { api } from '../services/api';
import { TrainingRun } from '../types';
import { GlassCard, SectionHeader, LiveBadge } from '../components/ui/GlassCard';
import { cn } from '../utils/cn';

const TOOLTIP_STYLE = {
  backgroundColor: '#0f172a',
  border: '1px solid #334155',
  borderRadius: '10px',
  fontSize: '11px',
  color: '#f8fafc',
};

interface SimPoint {
  epoch: number;
  loss: number;
  valLoss: number;
  acc: number;
  auroc: number;
}

function makeCurve(seed: number, target: number, start: number): SimPoint[] {
  const pts: SimPoint[] = [];
  for (let e = 0; e <= 45; e++) {
    const t = e / 45;
    const noise = Math.sin(e * 0.6 + seed) * 0.012 * (1 - t * 0.6);
    pts.push({
      epoch: e,
      loss: +(start - (start - target) * t * t + noise + 0.01 * Math.exp(-3 * t)).toFixed(3),
      valLoss: +(start + 0.04 - (start - target) * Math.pow(t, 1.8) + noise * 1.4).toFixed(3),
      acc: +(58 + 36 * (1 - Math.exp(-2.4 * t)) + Math.sin(e * 0.4 + seed * 2) * 0.8).toFixed(1),
      auroc: +(0.62 + 0.27 * (1 - Math.exp(-2.1 * t))).toFixed(3),
    });
  }
  return pts;
}

const OPTIMIZERS = ['AdamW', 'Adam', 'SGD Momentum', 'RMSProp'];
const SCHEDULERS = ['CosineAnnealingLR', 'StepLR', 'ReduceLROnPlateau', 'OneCycleLR'];

export const Training: React.FC = () => {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(32);
  const [lr, setLr] = useState('1e-4');
  const [optimizer, setOptimizer] = useState(OPTIMIZERS[0]);
  const [scheduler, setScheduler] = useState(SCHEDULERS[0]);
  const [earlyStop, setEarlyStop] = useState(true);
  const [checkpoints, setCheckpoints] = useState(true);
  const [simulating, setSimulating] = useState(false);
  const [simEpoch, setSimEpoch] = useState(0);
  const simTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    api.getTrainingRuns().then(setRuns);
    return () => {
      if (simTimer.current) clearInterval(simTimer.current);
    };
  }, []);

  const runningRun = runs.find((r) => r.status === 'RUNNING');
  const curve = useMemo(() => makeCurve(7, 0.09, 0.38), []);

  const startSimulation = () => {
    setSimulating(true);
    setSimEpoch(0);
    simTimer.current = setInterval(() => {
      setSimEpoch((e) => {
        if (e >= epochs - 1) {
          if (simTimer.current) clearInterval(simTimer.current);
          setSimulating(false);
          return e;
        }
        return e + 1;
      });
    }, 120);
  };

  const pauseSimulation = () => {
    setSimulating(false);
    if (simTimer.current) clearInterval(simTimer.current);
  };

  const resetSimulation = () => {
    pauseSimulation();
    setSimEpoch(0);
  };

  const visibleCurve = curve.filter((p) => p.epoch <= simEpoch);

  const gpuSpecs = [
    { label: 'GPU', value: 'NVIDIA H100 · 4×', icon: <Cpu className="w-3.5 h-3.5" /> },
    { label: 'VRAM', value: '78 / 80 GB', icon: <MemoryStick className="w-3.5 h-3.5" /> },
    { label: 'Utilization', value: simulating ? '92%' : '18%', icon: <Gauge className="w-3.5 h-3.5" /> },
    { label: 'Mixed Precision', value: 'FP16 + BF16', icon: <Flame className="w-3.5 h-3.5" /> },
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-black tracking-tight text-slate-900 dark:text-white">Model Training Studio</h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Configure, launch, and monitor deep learning experiments with live telemetry
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 text-[10px] font-mono text-fuchsia-600 dark:text-fuchsia-400 bg-fuchsia-500/10 border border-fuchsia-500/25 rounded-full px-2.5 py-1.5">
            <GraduationCap className="w-3.5 h-3.5" /> PyTorch 2.2 · CUDA 12.2
          </span>
        </div>
      </div>

      {/* Accelerator strip */}
      <div className="ring-gradient surface-card p-4 grid grid-cols-2 md:grid-cols-4 gap-3">
        {gpuSpecs.map((s) => (
          <div key={s.label} className="flex items-center gap-2.5">
            <span className="w-8 h-8 rounded-lg bg-slate-100 dark:bg-slate-800 text-indigo-500 flex items-center justify-center shrink-0">
              {s.icon}
            </span>
            <div>
              <p className="text-[9px] font-mono uppercase tracking-widest text-slate-400">{s.label}</p>
              <p className="text-xs font-bold text-slate-800 dark:text-slate-100 font-mono">{s.value}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Hyperparameters */}
        <GlassCard>
          <SectionHeader icon={<Settings2 className="w-4 h-4 text-indigo-500" />} title="Hyperparameters" subtitle="Experiment configuration" />
          <div className="space-y-4 mt-4">
            <div>
              <div className="flex justify-between text-[11px] font-semibold text-slate-600 dark:text-slate-300 mb-1.5">
                <span>Epochs</span>
                <span className="font-mono text-indigo-600 dark:text-indigo-400">{epochs}</span>
              </div>
              <input type="range" min={5} max={150} value={epochs} onChange={(e) => setEpochs(+e.target.value)} className="w-full accent-indigo-600 cursor-pointer" />
            </div>
            <div>
              <div className="flex justify-between text-[11px] font-semibold text-slate-600 dark:text-slate-300 mb-1.5">
                <span>Batch Size</span>
                <span className="font-mono text-indigo-600 dark:text-indigo-400">{batchSize}</span>
              </div>
              <input type="range" min={8} max={256} step={8} value={batchSize} onChange={(e) => setBatchSize(+e.target.value)} className="w-full accent-indigo-600 cursor-pointer" />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <label className="space-y-1">
                <span className="text-[10px] font-semibold text-slate-500 block">Learning Rate</span>
                <input
                  value={lr}
                  onChange={(e) => setLr(e.target.value)}
                  className="w-full bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 rounded-lg px-2.5 py-2 text-xs font-mono text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-primary-400/50"
                />
              </label>
              <label className="space-y-1">
                <span className="text-[10px] font-semibold text-slate-500 block">Optimizer</span>
                <select value={optimizer} onChange={(e) => setOptimizer(e.target.value)} className="w-full bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 rounded-lg px-2.5 py-2 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-primary-400/50">
                  {OPTIMIZERS.map((o) => <option key={o}>{o}</option>)}
                </select>
              </label>
              <label className="space-y-1 col-span-2">
                <span className="text-[10px] font-semibold text-slate-500 block">Scheduler</span>
                <select value={scheduler} onChange={(e) => setScheduler(e.target.value)} className="w-full bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 rounded-lg px-2.5 py-2 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-primary-400/50">
                  {SCHEDULERS.map((s) => <option key={s}>{s}</option>)}
                </select>
              </label>
            </div>

            <div className="space-y-2.5 bg-slate-50 dark:bg-slate-800/50 rounded-xl p-3">
              <label className="flex items-center justify-between cursor-pointer">
                <span className="text-[11px] font-semibold text-slate-700 dark:text-slate-200 flex items-center gap-1.5">
                  <Flag className="w-3.5 h-3.5 text-amber-500" /> Early Stopping (patience 7)
                </span>
                <input type="checkbox" checked={earlyStop} onChange={(e) => setEarlyStop(e.target.checked)} className="w-4 h-4 accent-indigo-600 rounded cursor-pointer" />
              </label>
              <label className="flex items-center justify-between cursor-pointer">
                <span className="text-[11px] font-semibold text-slate-700 dark:text-slate-200 flex items-center gap-1.5">
                  <Save className="w-3.5 h-3.5 text-sky-500" /> Checkpoint every 5 epochs
                </span>
                <input type="checkbox" checked={checkpoints} onChange={(e) => setCheckpoints(e.target.checked)} className="w-4 h-4 accent-indigo-600 rounded cursor-pointer" />
              </label>
            </div>

            {/* Controls */}
            <div className="grid grid-cols-3 gap-2">
              {!simulating ? (
                <button onClick={startSimulation} className="col-span-2 btn-gradient text-white text-xs font-bold py-2.5 rounded-xl flex items-center justify-center gap-1.5 cursor-pointer">
                  <Play className="w-4 h-4 fill-white" /> Start Training
                </button>
              ) : (
                <button onClick={pauseSimulation} className="col-span-2 bg-amber-500 hover:bg-amber-600 text-white text-xs font-bold py-2.5 rounded-xl flex items-center justify-center gap-1.5 cursor-pointer">
                  <Pause className="w-4 h-4" /> Pause
                </button>
              )}
              <button onClick={resetSimulation} className="text-xs font-bold py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:border-rose-400 hover:text-rose-500 flex items-center justify-center gap-1 cursor-pointer transition-colors">
                <RotateCcw className="w-3.5 h-3.5" /> Reset
              </button>
            </div>
            {simulating && (
              <div className="flex items-center justify-between text-[10px] font-mono text-indigo-600 dark:text-indigo-400">
                <span className="flex items-center gap-1.5">
                  <span className="live-dot w-1.5 h-1.5 rounded-full bg-indigo-500" />
                  Training epoch {simEpoch + 1}/{epochs}
                </span>
                <span>{((simEpoch + 1) / epochs) * 100 | 0}%</span>
              </div>
            )}
          </div>
        </GlassCard>

        {/* Live metrics */}
        <GlassCard className="lg:col-span-2" gradient>
          <SectionHeader
            icon={<Rocket className="w-4 h-4 text-emerald-500" />}
            title={simulating ? `Training ${optimizer} · ${lr} · batch ${batchSize}` : 'Live Training Telemetry'}
            subtitle={simulating ? `Epoch ${simEpoch + 1} of ${epochs}` : 'Launch a run to stream loss & accuracy curves'}
            right={<LiveBadge label={simulating ? 'Streaming' : 'Idle'} color={simulating ? 'emerald' : 'amber'} />}
          />
          {simulating ? (
            <>
              <div className="h-56 mt-4">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={visibleCurve} margin={{ top: 5, right: 10, left: -12, bottom: 0 }}>
                    <defs>
                      <linearGradient id="gradLoss" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#6366f1" stopOpacity={0.3} />
                        <stop offset="100%" stopColor="#6366f1" stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="epoch" stroke="var(--text-soft)" fontSize={10} axisLine={false} tickLine={false} label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: 'var(--text-soft)', fontSize: 9 }} />
                    <YAxis stroke="var(--text-soft)" fontSize={10} axisLine={false} tickLine={false} domain={[0, 0.45]} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                    <Area type="monotone" dataKey="loss" name="Train Loss" stroke="#6366f1" strokeWidth={2.5} fill="url(#gradLoss)" />
                    <Line type="monotone" dataKey="valLoss" name="Val Loss" stroke="#f59e0b" strokeWidth={2.5} dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="h-44 mt-3">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={visibleCurve} margin={{ top: 5, right: 10, left: -12, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="epoch" stroke="var(--text-soft)" fontSize={10} axisLine={false} tickLine={false} />
                    <YAxis stroke="var(--text-soft)" fontSize={10} axisLine={false} tickLine={false} domain={[50, 100]} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                    <Line type="monotone" dataKey="acc" name="Accuracy %" stroke="#10b981" strokeWidth={2.5} dot={false} />
                    <Line type="monotone" dataKey="auroc" name="AUROC" stroke="#06b6d4" strokeWidth={2.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <div className="w-16 h-16 rounded-3xl bg-slate-100 dark:bg-slate-800 flex items-center justify-center text-slate-300 mb-4">
                <GraduationCap className="w-7 h-7" />
              </div>
              <p className="text-sm font-bold text-slate-700 dark:text-slate-200">No active training run</p>
              <p className="text-xs text-slate-500 mt-1 max-w-sm">Configure hyperparameters and press <strong>Start Training</strong> to stream live loss, accuracy, and AUROC curves.</p>
            </div>
          )}
        </GlassCard>
      </div>

      {/* Runs table */}
      <GlassCard className="!p-0 overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-100 dark:border-slate-800 flex items-center justify-between">
          <SectionHeader icon={<GraduationCap className="w-4 h-4 text-violet-500" />} title="Experiment Runs" subtitle="MLflow-tracked training jobs" />
          <button className="text-[11px] font-semibold text-slate-400 hover:text-primary-500 flex items-center gap-1 cursor-pointer">
            <Download className="w-3.5 h-3.5" /> Export
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-700 dark:text-slate-300">
            <thead className="bg-slate-50 dark:bg-slate-800/60 text-slate-500 dark:text-slate-400 font-bold text-[10px] uppercase tracking-widest border-b border-slate-100 dark:border-slate-800">
              <tr>
                <th className="p-3.5">Run</th>
                <th className="p-3.5">Model / Dataset</th>
                <th className="p-3.5">Status</th>
                <th className="p-3.5">Epoch</th>
                <th className="p-3.5">Loss</th>
                <th className="p-3.5">Val AUROC</th>
                <th className="p-3.5">GPU</th>
                <th className="p-3.5">Duration</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60">
              {runs.map((r) => (
                <tr key={r.id} className="hover:bg-primary-50/30 dark:hover:bg-primary-900/10 transition-colors">
                  <td className="p-3.5">
                    <p className="font-bold text-slate-900 dark:text-white">{r.name}</p>
                    <p className="text-[10px] font-mono text-slate-400">{r.id}</p>
                  </td>
                  <td className="p-3.5">
                    <p className="font-semibold">{r.model}</p>
                    <p className="text-[10px] text-slate-400">{r.dataset}</p>
                  </td>
                  <td className="p-3.5">
                    <span className={cn(
                      'inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border text-[10px] font-bold font-mono',
                      r.status === 'RUNNING' && 'text-emerald-500 border-emerald-500/30 bg-emerald-500/10',
                      r.status === 'FINISHED' && 'text-sky-500 border-sky-500/30 bg-sky-500/10',
                      r.status === 'FAILED' && 'text-rose-500 border-rose-500/30 bg-rose-500/10',
                      r.status === 'QUEUED' && 'text-amber-500 border-amber-500/30 bg-amber-500/10',
                      r.status === 'PAUSED' && 'text-slate-400 border-slate-400/30 bg-slate-400/10',
                    )}>
                      {r.status === 'RUNNING' && <span className="live-dot w-1.5 h-1.5 rounded-full bg-current" />}
                      {r.status}
                    </span>
                  </td>
                  <td className="p-3.5 font-mono">{r.status === 'RUNNING' ? `${Math.min(r.epoch + Math.round(simEpoch * 0.3), r.totalEpochs)}/${r.totalEpochs}` : `${r.epoch}/${r.totalEpochs}`}</td>
                  <td className="p-3.5 font-mono text-indigo-600 dark:text-indigo-400">{r.loss || '—'}</td>
                  <td className="p-3.5 font-mono text-emerald-600 dark:text-emerald-400 font-bold">{r.valAuroc || '—'}</td>
                  <td className="p-3.5">
                    <div className="flex items-center gap-2">
                      <div className="w-12 h-1.5 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                        <div className={cn('h-full rounded-full', r.status === 'RUNNING' ? 'bg-emerald-500' : r.status === 'FAILED' ? 'bg-rose-500' : 'bg-indigo-500')} style={{ width: `${r.gpuUtil}%` }} />
                      </div>
                      <span className="text-[10px] font-mono text-slate-400">{r.gpuUtil}%</span>
                    </div>
                  </td>
                  <td className="p-3.5 font-mono text-slate-500">{r.duration}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </GlassCard>
    </div>
  );
};
