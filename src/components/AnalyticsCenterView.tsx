import React, { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  AreaChart, Area, PieChart, Pie, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, LineChart, Line
} from 'recharts';
import {
  Activity, Cpu, HardDrive, Zap, TrendingUp, ShieldAlert,
  Download, Calendar, RefreshCw, BarChart3, Database, Filter, Server, Target
} from 'lucide-react';

export const AnalyticsCenterView: React.FC = () => {
  const [timeRange, setTimeRange] = useState<'24h' | '7d' | '30d' | '1y'>('7d');

  // Daily Trends Data
  const trendData = [
    { date: 'Mon', predictions: 340, avgConfidence: 94.2, latencyMs: 24, gpuLoad: 68 },
    { date: 'Tue', predictions: 420, avgConfidence: 95.1, latencyMs: 22, gpuLoad: 74 },
    { date: 'Wed', predictions: 510, avgConfidence: 93.8, latencyMs: 26, gpuLoad: 82 },
    { date: 'Thu', predictions: 480, avgConfidence: 94.7, latencyMs: 21, gpuLoad: 71 },
    { date: 'Fri', predictions: 610, avgConfidence: 96.0, latencyMs: 19, gpuLoad: 89 },
    { date: 'Sat', predictions: 290, avgConfidence: 94.5, latencyMs: 23, gpuLoad: 55 },
    { date: 'Sun', predictions: 310, avgConfidence: 95.4, latencyMs: 20, gpuLoad: 58 },
  ];

  // Disease Prevalence Frequency
  const diseaseFrequency = [
    { disease: 'Pneumonia', count: 1240, color: '#6366f1' },
    { disease: 'Lung Opacity', count: 980, color: '#38bdf8' },
    { disease: 'Atelectasis', count: 640, color: '#10b981' },
    { disease: 'Pleural Effusion', count: 520, color: '#f59e0b' },
    { disease: 'Cardiomegaly', count: 310, color: '#8b5cf6' },
    { disease: 'Pneumothorax', count: 210, color: '#ec4899' },
    { disease: 'Normal / Clear', count: 1850, color: '#64748b' },
  ];

  // GPU & System Telemetry Metrics
  const gpuMetrics = [
    { metric: 'Tensor Core Usage', value: 84.5 },
    { metric: 'VRAM Allocation', value: 72.0 },
    { metric: 'FP16 Throughput', value: 92.3 },
    { metric: 'Batch Parallelism', value: 88.0 },
    { metric: 'Thermal Efficiency', value: 95.0 },
  ];

  // Model evaluation curves (CheXNet validation split)
  const rocData = [
    { fpr: 0, tpr: 0 },
    { fpr: 0.02, tpr: 0.45 },
    { fpr: 0.05, tpr: 0.72 },
    { fpr: 0.1, tpr: 0.88 },
    { fpr: 0.18, tpr: 0.94 },
    { fpr: 0.3, tpr: 0.97 },
    { fpr: 0.5, tpr: 0.99 },
    { fpr: 1.0, tpr: 1.0 },
  ];
  const prData = [
    { recall: 0, precision: 1.0 },
    { recall: 0.2, precision: 0.98 },
    { recall: 0.4, precision: 0.95 },
    { recall: 0.6, precision: 0.91 },
    { recall: 0.8, precision: 0.84 },
    { recall: 0.95, precision: 0.72 },
    { recall: 1.0, precision: 0.6 },
  ];
  const calibrationData = [
    { bin: '0.0-0.2', perfect: 10, actual: 12 },
    { bin: '0.2-0.4', perfect: 30, actual: 34 },
    { bin: '0.4-0.6', perfect: 50, actual: 52 },
    { bin: '0.6-0.8', perfect: 70, actual: 71 },
    { bin: '0.8-1.0', perfect: 90, actual: 89 },
  ];

  // Confusion matrix (binary, normalized %)
  const confusion = [
    { actual: 'Disease', predicted: 'Disease', value: 94.2 },
    { actual: 'Disease', predicted: 'Normal', value: 5.8 },
    { actual: 'Normal', predicted: 'Disease', value: 4.1 },
    { actual: 'Normal', predicted: 'Normal', value: 95.9 },
  ];
  const confusionColor = (v: number) => (v >= 90 ? 'rgba(16,185,129,0.85)' : v >= 10 ? 'rgba(245,158,11,0.7)' : 'rgba(244,63,94,0.75)');

  // Cohort analytics
  const ageDistribution = [
    { range: '0-17', count: 320 },
    { range: '18-30', count: 890 },
    { range: '31-45', count: 1240 },
    { range: '46-60', count: 1680 },
    { range: '61-75', count: 1420 },
    { range: '76+', count: 710 },
  ];
  const featureImportance = [
    { feature: 'Opacities', value: 0.92 },
    { feature: 'Cardiac Size', value: 0.84 },
    { feature: 'Effusion', value: 0.76 },
    { feature: 'Texture', value: 0.61 },
    { feature: 'Position', value: 0.42 },
  ];
  const metricsSummary = [
    { label: 'Precision', value: '92.1%' },
    { label: 'Recall', value: '90.4%' },
    { label: 'F1 Score', value: '0.912' },
    { label: 'Specificity', value: '95.9%' },
    { label: 'Sensitivity', value: '94.2%' },
    { label: 'AUROC', value: '0.948' },
  ];

  // Download export helper
  const handleExportCSV = () => {
    const csvContent = "data:text/csv;charset=utf-8," 
      + "Date,Predictions,AvgConfidence,LatencyMs,GPULoad\n"
      + trendData.map(e => `${e.date},${e.predictions},${e.avgConfidence},${e.latencyMs},${e.gpuLoad}`).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `MedVision_Analytics_${timeRange}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6 text-slate-100">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-slate-900/90 p-4 rounded-xl border border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-indigo-400" />
            <h1 className="text-lg font-bold text-white tracking-tight">Executive Analytics &amp; GPU Center</h1>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-time telemetry, model latency distribution, GPU throughput &amp; disease frequency
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Time Range Selector */}
          <div className="bg-slate-950 p-1 rounded-lg border border-slate-800 flex text-xs">
            {(['24h', '7d', '30d', '1y'] as const).map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-2.5 py-1 rounded font-medium transition-colors cursor-pointer ${
                  timeRange === range ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                {range}
              </button>
            ))}
          </div>

          <button
            onClick={handleExportCSV}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 rounded-lg border border-slate-700 flex items-center gap-1.5 cursor-pointer transition-colors"
          >
            <Download className="w-3.5 h-3.5 text-indigo-400" />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800 shadow-sm space-y-1">
          <div className="flex justify-between items-center text-slate-400 text-xs">
            <span>Total Inferences</span>
            <Activity className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-2xl font-black text-white font-mono">2,960</div>
          <div className="text-[11px] text-emerald-400 font-medium flex items-center gap-1">
            <TrendingUp className="w-3 h-3" /> +18.4% vs last period
          </div>
        </div>

        <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800 shadow-sm space-y-1">
          <div className="flex justify-between items-center text-slate-400 text-xs">
            <span>Avg Model Certainty</span>
            <Zap className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-black text-amber-400 font-mono">94.8%</div>
          <div className="text-[11px] text-slate-400 font-mono">Calibrated Softmax</div>
        </div>

        <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800 shadow-sm space-y-1">
          <div className="flex justify-between items-center text-slate-400 text-xs">
            <span>Mean Latency</span>
            <Cpu className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-black text-cyan-400 font-mono">21.4 ms</div>
          <div className="text-[11px] text-emerald-400 font-medium flex items-center gap-1">
            <span>TensorRT FP16</span>
          </div>
        </div>

        <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800 shadow-sm space-y-1">
          <div className="flex justify-between items-center text-slate-400 text-xs">
            <span>NVIDIA H100 Load</span>
            <Server className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-black text-emerald-400 font-mono">72.4%</div>
          <div className="text-[11px] text-slate-400 font-mono">78 GB VRAM Allocated</div>
        </div>
      </div>

      {/* Chart Row 1: Prediction Volume & Latency Trend */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-slate-900/90 p-4 rounded-xl border border-slate-800 space-y-3">
          <div className="flex justify-between items-center">
            <h3 className="text-sm font-bold text-white">Inference Throughput &amp; Confidence Trend</h3>
            <span className="text-[10px] font-mono text-slate-400">7-Day Rolling Horizon</span>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trendData}>
                <XAxis dataKey="date" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                <Area type="monotone" dataKey="predictions" name="Inferences" stroke="#6366f1" fill="#6366f1" fillOpacity={0.25} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* GPU Performance Radar */}
        <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800 space-y-3">
          <h3 className="text-sm font-bold text-white">GPU Infrastructure Radar</h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={gpuMetrics}>
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis dataKey="metric" stroke="#94a3b8" fontSize={10} />
                <PolarRadiusAxis domain={[0, 100]} stroke="#64748b" fontSize={9} />
                <Radar name="Hardware Efficiency" dataKey="value" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', fontSize: '11px' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Chart Row 2: Disease Frequency Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800 space-y-3">
          <h3 className="text-sm font-bold text-white">Most Common Diagnostic Detections</h3>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={diseaseFrequency} layout="vertical">
                <XAxis type="number" stroke="#64748b" fontSize={10} />
                <YAxis dataKey="disease" type="category" stroke="#cbd5e1" fontSize={10} width={100} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', fontSize: '11px' }} />
                <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                  {diseaseFrequency.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Latency vs GPU Load Line Chart */}
        <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800 space-y-3">
          <h3 className="text-sm font-bold text-white">Latency vs GPU Load Correlation</h3>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trendData}>
                <XAxis dataKey="date" stroke="#64748b" fontSize={10} />
                <YAxis yAxisId="left" stroke="#38bdf8" fontSize={10} unit="ms" />
                <YAxis yAxisId="right" orientation="right" stroke="#f59e0b" fontSize={10} unit="%" />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', fontSize: '11px' }} />
                <Line yAxisId="left" type="monotone" dataKey="latencyMs" name="Latency (ms)" stroke="#38bdf8" strokeWidth={2} />
                <Line yAxisId="right" type="monotone" dataKey="gpuLoad" name="GPU Load (%)" stroke="#f59e0b" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Model Evaluation Suite */}
      <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Target className="w-4.5 h-4.5 text-indigo-400" />
            <h3 className="text-sm font-bold text-white">Model Evaluation Suite</h3>
          </div>
          <span className="text-[10px] font-mono text-slate-400">CheXNet · Validation Split</span>
        </div>

        {/* Metric summary strip */}
        <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
          {metricsSummary.map((m) => (
            <div key={m.label} className="bg-slate-950/80 border border-slate-800 rounded-lg py-2 px-1 text-center">
              <p className="text-[9px] font-mono uppercase tracking-widest text-slate-400">{m.label}</p>
              <p className="text-sm font-black text-emerald-400 font-mono mt-0.5">{m.value}</p>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* ROC */}
          <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
            <div className="flex justify-between items-center mb-2">
              <h4 className="text-xs font-bold text-slate-200">ROC Curve</h4>
              <span className="text-[10px] font-mono text-emerald-400 font-bold">AUROC = 0.948</span>
            </div>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rocData} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                  <XAxis dataKey="fpr" stroke="#64748b" fontSize={9} label={{ value: 'FPR', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 9 }} />
                  <YAxis dataKey="tpr" stroke="#64748b" fontSize={9} label={{ value: 'TPR', angle: -90, position: 'insideLeft', offset: 12, fill: '#64748b', fontSize: 9 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', fontSize: '10px' }} />
                  <Line type="monotone" dataKey="tpr" stroke="#10b981" strokeWidth={2.5} dot={false} />
                  <Line type="monotone" dataKey="fpr" stroke="#475569" strokeWidth={1} strokeDasharray="4 4" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* PR */}
          <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
            <div className="flex justify-between items-center mb-2">
              <h4 className="text-xs font-bold text-slate-200">Precision-Recall Curve</h4>
              <span className="text-[10px] font-mono text-indigo-400 font-bold">AP = 0.921</span>
            </div>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={prData} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                  <XAxis dataKey="recall" stroke="#64748b" fontSize={9} label={{ value: 'Recall', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 9 }} />
                  <YAxis dataKey="precision" stroke="#64748b" fontSize={9} label={{ value: 'Precision', angle: -90, position: 'insideLeft', offset: 12, fill: '#64748b', fontSize: 9 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', fontSize: '10px' }} />
                  <Line type="monotone" dataKey="precision" stroke="#818cf8" strokeWidth={2.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Calibration */}
          <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
            <div className="flex justify-between items-center mb-2">
              <h4 className="text-xs font-bold text-slate-200">Calibration Curve</h4>
              <span className="text-[10px] font-mono text-amber-400 font-bold">ECE = 1.2%</span>
            </div>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={calibrationData} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                  <XAxis dataKey="bin" stroke="#64748b" fontSize={8} />
                  <YAxis stroke="#64748b" fontSize={9} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', fontSize: '10px' }} />
                  <Line type="monotone" dataKey="perfect" stroke="#475569" strokeDasharray="4 4" name="Perfect" dot={false} />
                  <Line type="monotone" dataKey="actual" stroke="#f59e0b" strokeWidth={2.5} name="Model" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Confusion matrix */}
          <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
            <h4 className="text-xs font-bold text-slate-200 mb-2">Confusion Matrix (Normalized)</h4>
            <div className="grid grid-cols-[auto_1fr_1fr] gap-1.5 text-center">
              <div />
              <div className="text-[9px] font-mono text-slate-400 pb-1">Pred: Disease</div>
              <div className="text-[9px] font-mono text-slate-400 pb-1">Pred: Normal</div>
              <div className="text-[9px] font-mono text-slate-400 pr-1 flex items-center">Actual: Disease</div>
              {confusion.slice(0, 2).map((c, i) => (
                <div key={i} className="h-16 rounded-lg flex flex-col items-center justify-center text-white" style={{ background: confusionColor(c.value) }}>
                  <span className="text-lg font-black font-mono">{c.value}%</span>
                  <span className="text-[8px] font-mono opacity-80">{i === 0 ? 'TP' : 'FN'}</span>
                </div>
              ))}
              <div className="text-[9px] font-mono text-slate-400 pr-1 flex items-center">Actual: Normal</div>
              {confusion.slice(2).map((c, i) => (
                <div key={i} className="h-16 rounded-lg flex flex-col items-center justify-center text-white" style={{ background: confusionColor(c.value) }}>
                  <span className="text-lg font-black font-mono">{c.value}%</span>
                  <span className="text-[8px] font-mono opacity-80">{i === 0 ? 'FP' : 'TN'}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Cohort Analytics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Age distribution */}
        <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800 space-y-3">
          <h3 className="text-sm font-bold text-white">Patient Age Distribution</h3>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ageDistribution} margin={{ top: 5, right: 10, left: -15, bottom: 0 }}>
                <XAxis dataKey="range" stroke="#64748b" fontSize={10} />
                <YAxis stroke="#64748b" fontSize={10} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', fontSize: '10px' }} cursor={{ fill: 'rgba(99,102,241,0.06)' }} />
                <Bar dataKey="count" name="Patients" radius={[6, 6, 0, 0]}>
                  {ageDistribution.map((_, i) => (
                    <Cell key={i} fill={['#6366f1', '#818cf8', '#38bdf8', '#22d3ee', '#2dd4bf', '#10b981'][i % 6]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Feature importance */}
        <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800 space-y-3">
          <h3 className="text-sm font-bold text-white">Model Feature Importance</h3>
          <div className="space-y-3">
            {featureImportance.map((f) => (
              <div key={f.feature} className="space-y-1">
                <div className="flex justify-between text-[11px]">
                  <span className="text-slate-300 font-medium">{f.feature}</span>
                  <span className="font-mono text-indigo-300">{f.value.toFixed(2)}</span>
                </div>
                <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-cyan-400 transition-all duration-1000"
                    style={{ width: `${f.value * 100}%` }}
                  />
                </div>
              </div>
            ))}
            <p className="text-[10px] text-slate-500 leading-relaxed pt-1">
              SHAP-based attribution across 1,200 hold-out studies. Opacity and cardiac size dominate the decision boundary — consistent with clinical expectation for chest X-ray triage.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
