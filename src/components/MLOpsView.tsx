import React, { useState } from 'react';
import { BarChart3, Database, ShieldCheck, Activity, Layers, GitBranch, RefreshCw, Sliders } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, Legend } from 'recharts';

export const MLOpsView: React.FC = () => {
  const [calibrationMethod, setCalibrationMethod] = useState<'Platt' | 'Isotonic' | 'Uncalibrated'>('Platt');

  // Calibration reliability plot points
  const reliabilityData = [
    { binConfidence: 10, trueAccuracy: 9.8, uncalibrated: 14.2 },
    { binConfidence: 30, trueAccuracy: 29.5, uncalibrated: 38.0 },
    { binConfidence: 50, trueAccuracy: 49.8, uncalibrated: 61.5 },
    { binConfidence: 70, trueAccuracy: 70.2, uncalibrated: 81.0 },
    { binConfidence: 90, trueAccuracy: 89.6, uncalibrated: 96.8 },
  ];

  const mlflowRuns = [
    { runId: 'run-8821a', name: 'densenet121-focal-loss-v2', status: 'FINISHED', epoch: 50, valAuroc: 0.841, loss: 0.142, date: '2026-07-30' },
    { runId: 'run-9943b', name: 'efficientnet-b3-weighted-bce', status: 'FINISHED', epoch: 40, valAuroc: 0.865, loss: 0.128, date: '2026-07-28' },
    { runId: 'run-1029c', name: 'convnext-base-cosine-lr', status: 'FINISHED', epoch: 60, valAuroc: 0.882, loss: 0.115, date: '2026-07-25' },
    { runId: 'run-1104d', name: 'swin-b-mixed-precision-amp', status: 'FINISHED', epoch: 45, valAuroc: 0.889, loss: 0.108, date: '2026-07-22' },
  ];

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6 text-slate-800">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
        <div>
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-indigo-600" />
            <span>MLOps, Model Calibration &amp; MLflow Registry</span>
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Model versioning, Platt scaling probability calibration, and experiment tracking logs.
          </p>
        </div>

        <div className="flex items-center gap-2 bg-emerald-50 px-3 py-1.5 rounded-md border border-emerald-200 text-xs font-mono text-emerald-800 font-medium">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping"></span>
          <span>MLflow Server Online (Port 5000)</span>
        </div>
      </div>

      {/* Grid: Calibration Plot + Registry Details */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Reliability Diagram */}
        <div className="lg:col-span-7 bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-xs font-bold text-slate-900">Probability Calibration Reliability Diagram</h3>
              <p className="text-[10px] text-slate-500">Assesses expected calibration error (ECE) vs perfectly calibrated probabilities.</p>
            </div>

            <select
              value={calibrationMethod}
              onChange={(e: any) => setCalibrationMethod(e.target.value)}
              className="bg-white border border-slate-300 text-slate-800 text-xs rounded-md px-2.5 py-1 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-mono shadow-2xs"
            >
              <option value="Platt">Platt Scaling (Logistic)</option>
              <option value="Isotonic">Isotonic Regression</option>
              <option value="Uncalibrated">Raw Model Outputs</option>
            </select>
          </div>

          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={reliabilityData} margin={{ top: 10, right: 30, left: 0, bottom: 10 }}>
                <XAxis dataKey="binConfidence" stroke="#64748b" fontSize={11} label={{ value: 'Predicted Confidence (%)', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }} />
                <YAxis stroke="#64748b" fontSize={11} label={{ value: 'Empirical Accuracy (%)', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }} />
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f8fafc', borderRadius: '0.375rem' }} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Line type="monotone" dataKey="binConfidence" name="Perfect Calibration (Ideal)" stroke="#10b981" strokeDasharray="5 5" />
                <Line type="monotone" dataKey="trueAccuracy" name="Platt Scaled (ECE = 1.2%)" stroke="#6366f1" strokeWidth={2} />
                <Line type="monotone" dataKey="uncalibrated" name="Uncalibrated (ECE = 8.4%)" stroke="#ef4444" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Explainability Toggles & Stats */}
        <div className="lg:col-span-5 bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-4 flex flex-col justify-between">
          <div className="space-y-3">
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest border-b border-slate-200 pb-2">
              Explainability Framework Configuration
            </h3>

            <div className="space-y-2 text-xs">
              <div className="p-2.5 bg-white rounded-md border border-slate-200 flex items-center justify-between shadow-2xs">
                <div>
                  <div className="font-semibold text-slate-900">Integrated Gradients</div>
                  <div className="text-[10px] text-slate-500">Path integral of gradients relative to baseline</div>
                </div>
                <span className="text-[10px] font-mono text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200 font-bold">ACTIVE</span>
              </div>

              <div className="p-2.5 bg-white rounded-md border border-slate-200 flex items-center justify-between shadow-2xs">
                <div>
                  <div className="font-semibold text-slate-900">SHAP Kernel Explainer</div>
                  <div className="text-[10px] text-slate-500">Shapley additive feature importance estimation</div>
                </div>
                <span className="text-[10px] font-mono text-indigo-800 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-200 font-bold">READY</span>
              </div>

              <div className="p-2.5 bg-white rounded-md border border-slate-200 flex items-center justify-between shadow-2xs">
                <div>
                  <div className="font-semibold text-slate-900">Captum PyTorch Integration</div>
                  <div className="text-[10px] text-slate-500">Model interpretability hooks for PyTorch 2.x</div>
                </div>
                <span className="text-[10px] font-mono text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200 font-bold">HOOKED</span>
              </div>
            </div>
          </div>

          <div className="bg-indigo-50 border border-indigo-200 p-3 rounded-md text-[11px] text-indigo-900">
            <strong>Expected Calibration Error (ECE):</strong> Platt Scaling reduces overconfidence bias from 8.4% ECE down to 1.2% ECE on the CheXNet validation split.
          </div>
        </div>
      </div>

      {/* MLflow Runs Table */}
      <div className="space-y-3">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest">MLflow Experiment Registry Runs</h3>
        <div className="overflow-x-auto border border-slate-200 rounded-lg shadow-2xs">
          <table className="w-full text-left text-xs text-slate-700">
            <thead className="bg-slate-50 text-slate-500 font-bold text-[11px] uppercase tracking-widest border-b border-slate-200">
              <tr>
                <th className="p-3">Run ID</th>
                <th className="p-3">Experiment Name</th>
                <th className="p-3">Status</th>
                <th className="p-3">Epochs</th>
                <th className="p-3">Val AUROC</th>
                <th className="p-3">Validation Loss</th>
                <th className="p-3">Timestamp</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white font-mono text-xs">
              {mlflowRuns.map((r) => (
                <tr key={r.runId} className="hover:bg-slate-50">
                  <td className="p-3 text-indigo-700 font-bold">{r.runId}</td>
                  <td className="p-3 text-slate-900 font-semibold">{r.name}</td>
                  <td className="p-3 text-emerald-700 font-bold">{r.status}</td>
                  <td className="p-3">{r.epoch}</td>
                  <td className="p-3 text-indigo-800 font-bold">{r.valAuroc}</td>
                  <td className="p-3 text-slate-500">{r.loss}</td>
                  <td className="p-3 text-slate-400">{r.date}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
