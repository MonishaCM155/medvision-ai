import React, { useState } from 'react';
import { BarChart2, PieChart, ShieldAlert, CheckCircle2, AlertTriangle } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ReferenceLine } from 'recharts';
import { DiseaseProbability } from '../types';

interface ConfidenceChartProps {
  diseases: DiseaseProbability[];
}

export const ConfidenceChart: React.FC<ConfidenceChartProps> = ({ diseases }) => {
  const [threshold] = useState<number>(0.5);

  const chartData = diseases.map((d) => ({
    name: d.disease,
    confidence: Number((d.probability * 100).toFixed(1)),
    category: d.category,
    rawProb: d.probability,
    description: d.description,
  }));

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6 text-slate-800">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
        <div>
          <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <BarChart2 className="w-5 h-5 text-indigo-600" />
            <span>Multi-Disease Probability Distribution</span>
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Multi-label sigmoid probabilities for 10 chest radiograph pathologies.
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono bg-slate-50 px-3 py-1.5 rounded-md border border-slate-200 text-slate-700 shadow-2xs">
          <span className="w-2 h-2 rounded-full bg-indigo-600"></span>
          <span>Decision Threshold: 50.0%</span>
        </div>
      </div>

      {/* Main Bar Chart */}
      <div className="h-72 w-full bg-slate-50 p-2 rounded-lg border border-slate-200">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ top: 10, right: 30, left: 90, bottom: 10 }}>
            <XAxis type="number" domain={[0, 100]} stroke="#64748b" fontSize={11} tickFormatter={(v) => `${v}%`} />
            <YAxis type="category" dataKey="name" stroke="#475569" fontSize={11} width={85} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', borderRadius: '0.375rem', color: '#f8fafc' }}
              formatter={(value: any) => [`${value}% Probability`, 'Confidence']}
            />
            <ReferenceLine x={50} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'Cutoff 50%', fill: '#ef4444', fontSize: 10 }} />
            <Bar dataKey="confidence" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, index) => {
                let color = '#6366f1'; // indigo default
                if (entry.name === 'No Finding') {
                  color = '#10b981'; // emerald
                } else if (entry.confidence >= 75) {
                  color = '#ef4444'; // red high
                } else if (entry.confidence >= 50) {
                  color = '#f59e0b'; // amber moderate
                }
                return <Cell key={`cell-${index}`} fill={color} />;
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Pathological Findings Table Breakdown */}
      <div className="space-y-2">
        <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Pathology Risk Breakdown</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
          {diseases.map((item) => {
            const isPositive = item.probability >= 0.5;
            const isNormal = item.disease === 'No Finding';

            return (
              <div
                key={item.disease}
                className={`p-3 rounded-lg border flex items-start justify-between gap-3 text-xs ${
                  isNormal && item.probability > 0.5
                    ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
                    : isPositive
                    ? 'bg-rose-50 border-rose-200 text-rose-900'
                    : 'bg-slate-50 border-slate-200 text-slate-800'
                }`}
              >
                <div className="space-y-0.5">
                  <div className="font-bold flex items-center gap-1.5">
                    {isPositive ? (
                      isNormal ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                      ) : (
                        <AlertTriangle className="w-3.5 h-3.5 text-rose-600 shrink-0" />
                      )
                    ) : (
                      <span className="w-2 h-2 rounded-full bg-slate-400 shrink-0"></span>
                    )}
                    <span>{item.disease}</span>
                  </div>
                  <p className="text-[10px] text-slate-500 leading-tight">{item.description}</p>
                </div>

                <div className="text-right font-mono font-bold whitespace-nowrap">
                  <span className={isPositive ? (isNormal ? 'text-emerald-700' : 'text-rose-700') : 'text-slate-500'}>
                    {(item.probability * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
