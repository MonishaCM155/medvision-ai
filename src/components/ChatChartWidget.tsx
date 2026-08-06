import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  LineChart, Line, AreaChart, Area
} from 'recharts';
import { DiseaseProbability, SeverityLevel } from '../types';

export type ChartWidgetType =
  | 'probability_bars'
  | 'confidence_gauge'
  | 'disease_distribution'
  | 'top5_predictions'
  | 'severity_meter'
  | 'roc_curve'
  | 'model_benchmark'
  | 'confusion_matrix';

interface ChatChartWidgetProps {
  type: ChartWidgetType;
  diseases?: DiseaseProbability[];
  topDiagnosis?: string;
  topConfidence?: number;
  severity?: SeverityLevel;
  severityScore?: number;
}

export const ChatChartWidget: React.FC<ChatChartWidgetProps> = ({
  type,
  diseases = [],
  topDiagnosis = 'Pneumonia',
  topConfidence = 0.94,
  severity = 'High',
  severityScore = 82,
}) => {
  // Top 5 diseases data
  const top5Data = (diseases.length > 0 ? diseases : [
    { disease: 'Pneumonia', probability: 0.94, category: 'infection' },
    { disease: 'Lung Opacity', probability: 0.82, category: 'lung_opacity' },
    { disease: 'Atelectasis', probability: 0.35, category: 'structural' },
    { disease: 'Pleural Effusion', probability: 0.25, category: 'pleural' },
    { disease: 'Cardiomegaly', probability: 0.08, category: 'structural' },
  ]).slice(0, 5).map(d => ({
    name: d.disease,
    prob: Number((d.probability * 100).toFixed(1)),
    formatted: `${(d.probability * 100).toFixed(1)}%`,
  }));

  // Category distribution data
  const categoryCount: Record<string, number> = {};
  diseases.forEach(d => {
    categoryCount[d.category] = (categoryCount[d.category] || 0) + 1;
  });
  const pieData = Object.keys(categoryCount).map(cat => ({
    name: cat.replace('_', ' ').toUpperCase(),
    value: categoryCount[cat],
  }));

  const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

  // ROC Curve Data
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

  // Model comparison benchmark data
  const modelData = [
    { model: 'DenseNet121', auroc: 88.4, latencyMs: 18, f1: 85.2 },
    { model: 'EfficientNet', auroc: 91.2, latencyMs: 24, f1: 88.1 },
    { model: 'ConvNeXt', auroc: 93.8, latencyMs: 38, f1: 90.6 },
    { model: 'Swin-B', auroc: 94.5, latencyMs: 45, f1: 92.0 },
  ];

  return (
    <div className="my-2 p-3 bg-slate-900/90 rounded-lg border border-slate-700/80 text-slate-100 shadow-inner overflow-hidden text-xs">
      {type === 'top5_predictions' && (
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="font-semibold text-indigo-300">Top 5 Predicted Conditions</span>
            <span className="text-[10px] text-slate-400 font-mono">Softmax Probabilities</span>
          </div>
          <div className="h-40 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={top5Data} layout="vertical" margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                <XAxis type="number" domain={[0, 100]} stroke="#94a3b8" fontSize={10} unit="%" />
                <YAxis dataKey="name" type="category" stroke="#cbd5e1" fontSize={10} width={90} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', borderRadius: '6px', color: '#fff', fontSize: '11px' }}
                  formatter={(val: any) => [`${val}%`, 'Probability']}
                />
                <Bar dataKey="prob" radius={[0, 4, 4, 0]}>
                  {top5Data.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={index === 0 ? '#6366f1' : '#38bdf8'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {type === 'probability_bars' && (
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="font-semibold text-indigo-300">Class Confidence Spectrum</span>
            <span className="text-[10px] text-slate-400">DenseNet-121 Logits</span>
          </div>
          <div className="space-y-1.5">
            {top5Data.map((item, idx) => (
              <div key={idx} className="space-y-0.5">
                <div className="flex justify-between text-[11px]">
                  <span className="font-medium text-slate-200">{item.name}</span>
                  <span className="font-mono text-indigo-300">{item.formatted}</span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      idx === 0 ? 'bg-indigo-500' : idx === 1 ? 'bg-blue-400' : 'bg-slate-500'
                    }`}
                    style={{ width: `${item.prob}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {type === 'confidence_gauge' && (
        <div className="text-center py-1">
          <div className="text-[11px] font-semibold text-slate-300 mb-1">Model Confidence Meter</div>
          <div className="inline-flex items-center justify-center p-3 rounded-full bg-slate-800/80 border border-indigo-500/40 my-1">
            <div className="text-center">
              <span className="text-2xl font-black text-indigo-400 tracking-tight font-mono">
                {(topConfidence * 100).toFixed(1)}%
              </span>
              <p className="text-[10px] text-slate-400 mt-0.5">Certainty Index</p>
            </div>
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2 text-left bg-slate-950/60 p-2 rounded border border-slate-800 text-[10px]">
            <div>
              <span className="text-slate-400">Diagnosis:</span>
              <p className="font-semibold text-white truncate">{topDiagnosis}</p>
            </div>
            <div>
              <span className="text-slate-400">Severity Grade:</span>
              <p className="font-semibold text-amber-400">{severity} ({severityScore}/100)</p>
            </div>
          </div>
        </div>
      )}

      {type === 'severity_meter' && (
        <div>
          <div className="flex justify-between items-center mb-1">
            <span className="font-semibold text-amber-300">Clinical Severity Score</span>
            <span className="font-mono font-bold text-amber-400">{severityScore} / 100</span>
          </div>
          <div className="w-full bg-slate-800 rounded-full h-3 overflow-hidden p-0.5 border border-slate-700">
            <div
              className={`h-full rounded-full transition-all duration-700 ${
                severityScore > 80
                  ? 'bg-gradient-to-r from-amber-500 to-red-600'
                  : severityScore > 50
                  ? 'bg-gradient-to-r from-yellow-400 to-amber-500'
                  : 'bg-emerald-500'
              }`}
              style={{ width: `${severityScore}%` }}
            />
          </div>
          <div className="flex justify-between text-[9px] text-slate-400 mt-1 font-mono">
            <span>Low (0-35)</span>
            <span>Moderate (36-65)</span>
            <span>High (66-85)</span>
            <span>Critical (86+)</span>
          </div>
        </div>
      )}

      {type === 'roc_curve' && (
        <div>
          <div className="flex justify-between items-center mb-1">
            <span className="font-semibold text-emerald-400">ROC Curve (AUROC = 0.948)</span>
            <span className="text-[10px] text-slate-400">CheXNet Validation</span>
          </div>
          <div className="h-36 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={rocData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <XAxis dataKey="fpr" label={{ value: 'FPR', position: 'insideBottom', offset: -2, fontSize: 9 }} stroke="#64748b" fontSize={9} />
                <YAxis dataKey="tpr" label={{ value: 'TPR', angle: -90, position: 'insideLeft', offset: 12, fontSize: 9 }} stroke="#64748b" fontSize={9} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', fontSize: '10px' }} />
                <Area type="monotone" dataKey="tpr" stroke="#10b981" fill="#10b981" fillOpacity={0.2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {type === 'model_benchmark' && (
        <div>
          <div className="font-semibold text-indigo-300 mb-2">Model Performance Benchmarks</div>
          <div className="h-36 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelData} margin={{ top: 5, right: 10, left: -15, bottom: 0 }}>
                <XAxis dataKey="model" stroke="#94a3b8" fontSize={9} />
                <YAxis domain={[80, 100]} stroke="#94a3b8" fontSize={9} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', fontSize: '10px' }} />
                <Bar dataKey="auroc" name="AUROC (%)" fill="#6366f1" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {type === 'disease_distribution' && (
        <div>
          <div className="font-semibold text-indigo-300 mb-2">Disease Category Spectrum</div>
          <div className="h-36 w-full flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={pieData.length ? pieData : [{ name: 'Infection', value: 3 }, { name: 'Structural', value: 2 }]} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={45} label={({ name }) => name}>
                  {pieData.map((_, idx) => (
                    <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', fontSize: '10px' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {type === 'confusion_matrix' && (
        <div>
          <div className="font-semibold text-indigo-300 mb-2">Normalized Confusion Matrix</div>
          <div className="grid grid-cols-2 gap-1 text-center font-mono text-[10px]">
            <div className="p-2 bg-indigo-950/90 rounded border border-indigo-700/50">
              <div className="text-emerald-400 font-bold text-sm">94.2%</div>
              <div className="text-slate-400">True Positive</div>
            </div>
            <div className="p-2 bg-slate-800 rounded border border-slate-700">
              <div className="text-amber-400 font-bold text-sm">5.8%</div>
              <div className="text-slate-400">False Negative</div>
            </div>
            <div className="p-2 bg-slate-800 rounded border border-slate-700">
              <div className="text-amber-400 font-bold text-sm">4.1%</div>
              <div className="text-slate-400">False Positive</div>
            </div>
            <div className="p-2 bg-indigo-950/90 rounded border border-indigo-700/50">
              <div className="text-emerald-400 font-bold text-sm">95.9%</div>
              <div className="text-slate-400">True Negative</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
