import React, { useState } from 'react';
import { Cpu, Zap, Activity, ShieldCheck, BarChart3, Layers, Sparkles } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from 'recharts';
import { MODEL_BENCHMARKS } from '../data/sampleXrays';

export const ModelComparisonView: React.FC = () => {
  const [activeBenchmarkMetric, setActiveBenchmarkMetric] = useState<'latency' | 'accuracy' | 'params'>('latency');

  const chartData = MODEL_BENCHMARKS.map((m) => ({
    name: m.architecture,
    FP32: m.latencyFp32Ms,
    FP16: m.latencyFp16Ms,
    ONNX: m.latencyOnnxMs,
    AUROC: Number((m.auroc * 100).toFixed(1)),
    Accuracy: m.accuracy,
    Params: parseFloat(m.parameters.replace('M', '')),
  }));

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6 text-slate-800">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
        <div>
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-indigo-600" />
            <span>Model Architecture Comparison &amp; Benchmark Suite</span>
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Performance comparison across Convolutional, Efficient, and Vision Transformer backbones on NIH ChestX-ray14.
          </p>
        </div>

        <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-md border border-slate-200">
          <button
            onClick={() => setActiveBenchmarkMetric('latency')}
            className={`px-3 py-1 rounded text-xs font-semibold transition-all ${
              activeBenchmarkMetric === 'latency' ? 'bg-indigo-600 text-white shadow-2xs' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Inference Speed (ms)
          </button>
          <button
            onClick={() => setActiveBenchmarkMetric('accuracy')}
            className={`px-3 py-1 rounded text-xs font-semibold transition-all ${
              activeBenchmarkMetric === 'accuracy' ? 'bg-indigo-600 text-white shadow-2xs' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            AUROC &amp; Accuracy
          </button>
          <button
            onClick={() => setActiveBenchmarkMetric('params')}
            className={`px-3 py-1 rounded text-xs font-semibold transition-all ${
              activeBenchmarkMetric === 'params' ? 'bg-indigo-600 text-white shadow-2xs' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Parameter Size (M)
          </button>
        </div>
      </div>

      {/* Recharts Graphical Benchmark */}
      <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-2">
        <div className="flex justify-between items-center text-xs text-slate-700 font-bold mb-2">
          <span>
            {activeBenchmarkMetric === 'latency' && 'Inference Latency Across Hardware Backends (Lower is Better)'}
            {activeBenchmarkMetric === 'accuracy' && 'Mean AUROC (%) & Overall Accuracy (%) (Higher is Better)'}
            {activeBenchmarkMetric === 'params' && 'Model Parameters in Millions (Lower Memory Footprint)'}
          </span>
          <span className="font-mono text-[10px] text-indigo-600 font-semibold">NVIDIA TensorRT / CUDA 12.2</span>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
              <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
              <YAxis stroke="#64748b" fontSize={11} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', borderRadius: '0.375rem', color: '#f8fafc' }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />

              {activeBenchmarkMetric === 'latency' && (
                <>
                  <Bar dataKey="FP32" name="PyTorch FP32 (ms)" fill="#ef4444" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="FP16" name="PyTorch AMP FP16 (ms)" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="ONNX" name="ONNX Runtime (ms)" fill="#10b981" radius={[4, 4, 0, 0]} />
                </>
              )}

              {activeBenchmarkMetric === 'accuracy' && (
                <>
                  <Bar dataKey="AUROC" name="Mean AUROC (%)" fill="#6366f1" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Accuracy" name="Top-1 Accuracy (%)" fill="#818cf8" radius={[4, 4, 0, 0]} />
                </>
              )}

              {activeBenchmarkMetric === 'params' && (
                <Bar dataKey="Params" name="Parameters (Millions)" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              )}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model Specifications Grid Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {MODEL_BENCHMARKS.map((m) => (
          <div key={m.id} className="bg-slate-50 border border-slate-200 rounded-lg p-4 space-y-3 flex flex-col justify-between">
            <div className="space-y-2">
              <div className="flex items-center justify-between border-b border-slate-200 pb-2">
                <h3 className="font-bold text-slate-900 text-xs">{m.name}</h3>
                <span className="bg-indigo-50 text-indigo-700 border border-indigo-200 px-2 py-0.5 rounded text-[10px] font-mono font-semibold">
                  {m.parameters}
                </span>
              </div>

              <p className="text-[11px] text-slate-600 leading-relaxed">{m.description}</p>
            </div>

            <div className="space-y-2 pt-2 border-t border-slate-200">
              <div className="grid grid-cols-3 gap-1 text-[10px] font-mono bg-white p-2 rounded text-center border border-slate-200 shadow-2xs">
                <div>
                  <span className="text-slate-500 block">AUROC</span>
                  <span className="text-emerald-700 font-bold">{m.auroc.toFixed(3)}</span>
                </div>
                <div>
                  <span className="text-slate-500 block">F1 SCORE</span>
                  <span className="text-indigo-700 font-bold">{m.f1Score.toFixed(3)}</span>
                </div>
                <div>
                  <span className="text-slate-500 block">ONNX LAT.</span>
                  <span className="text-amber-700 font-bold">{m.latencyOnnxMs}ms</span>
                </div>
              </div>

              <div className="text-[10px] text-slate-600">
                <strong className="text-indigo-700">Best Use Case:</strong> {m.recommendedFor}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
