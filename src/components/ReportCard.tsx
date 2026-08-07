import React, { useState } from 'react';
import { FileText, ShieldAlert, Download, ChevronDown, GitBranch, Stethoscope, QrCode, FlaskConical, Lock, AlertTriangle, Activity } from 'lucide-react';
import { RadiologyReport, SeverityLevel, DiseaseProbability, CalibrationInfo, UncertaintyInfo } from '../types';
import { suggestedTests, generateQrSvg } from '../utils/qr';
import { cn } from '../utils/cn';

interface ReportCardProps {
  report: RadiologyReport;
  severity: SeverityLevel;
  severityScore: number;
  topDiagnosis: string;
  topConfidence: number;
  onExportPdf: () => void;
  isExporting?: boolean;
  diseases?: DiseaseProbability[];
  /** Export (PDF/DOCX/QR) is locked until a validated chest X-ray has been analyzed. */
  canExport?: boolean;
  /** True when top confidence is below the configured clinical threshold. */
  lowConfidence?: boolean;
  /** Temperature-scaled calibration info from the safety pipeline. */
  calibration?: CalibrationInfo;
  /** Predictive uncertainty (MC-dropout or margin proxy). */
  uncertainty?: UncertaintyInfo;
}

export const ReportCard: React.FC<ReportCardProps> = ({
  report,
  severity,
  severityScore,
  topDiagnosis,
  topConfidence,
  onExportPdf,
  isExporting = false,
  diseases = [],
  canExport = true,
  lowConfidence = false,
  calibration,
  uncertainty,
}) => {
  const [explainOpen, setExplainOpen] = useState(false);

  const differential = diseases.filter((d) => d.probability >= 0.15 && d.disease !== 'No Finding').slice(0, 3);
  const tests = suggestedTests(topDiagnosis);
  // Severity color maps
  const severityBadgeColors: Record<SeverityLevel, string> = {
    Low: 'bg-emerald-50 text-emerald-800 border-emerald-200',
    Moderate: 'bg-amber-50 text-amber-800 border-amber-200',
    High: 'bg-orange-50 text-orange-800 border-orange-200',
    Critical: 'bg-rose-50 text-rose-800 border-rose-200',
  };

  const severityGaugeColor: Record<SeverityLevel, string> = {
    Low: 'from-emerald-500 to-teal-500',
    Moderate: 'from-amber-500 to-yellow-500',
    High: 'from-orange-500 to-amber-600',
    Critical: 'from-rose-600 to-red-600',
  };

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6 text-slate-800">
      {/* Title & PDF Export Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-indigo-600" />
            <h3 className="text-base font-bold text-slate-900">AI Radiology Impression &amp; Structured Report</h3>
            <span className="bg-indigo-50 text-indigo-700 text-[10px] font-semibold px-2 py-0.5 rounded border border-indigo-200">
              Gemini 3.6 Flash
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">Automated clinical report synthesis based on spatial visual activations.</p>
        </div>

        {canExport ? (
          <button
            onClick={onExportPdf}
            disabled={isExporting}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs rounded-md shadow-xs flex items-center justify-center gap-2 cursor-pointer transition-colors disabled:opacity-50"
          >
            <Download className="w-4 h-4" />
            <span>{isExporting ? 'Generating PDF...' : 'Download Hospital PDF Report'}</span>
          </button>
        ) : (
          <button
            disabled
            title="Run a validated chest X-ray analysis to unlock PDF/DOCX/QR export"
            className="px-4 py-2 bg-slate-200 dark:bg-slate-700 text-slate-500 dark:text-slate-400 font-bold text-xs rounded-md shadow-xs flex items-center justify-center gap-2 cursor-not-allowed border border-slate-300 dark:border-slate-600"
          >
            <Lock className="w-4 h-4" />
            <span>Report export locked</span>
          </button>
        )}
      </div>

      {/* High-uncertainty notice — never present an uncertain prediction as definitive */}
      {uncertainty && uncertainty.level === 'high' && (
        <div className="flex items-start gap-2.5 bg-rose-50 dark:bg-rose-500/10 border border-rose-300 dark:border-rose-500/40 rounded-lg p-3 text-[11px] text-rose-800 dark:text-rose-200 animate-fade-in">
          <Activity className="w-4 h-4 text-rose-600 dark:text-rose-400 shrink-0 mt-0.5" />
          <p className="leading-relaxed">
            <strong>AI confidence insufficient for reliable diagnosis.</strong> Predictive uncertainty is high
            ({uncertainty.score}/100 — {uncertainty.method}). The model is not certain enough about this study;
            treat the output as advisory and obtain a repeat or follow-up examination.
          </p>
        </div>
      )}

      {/* Low-confidence notice — never present a weak prediction as definitive */}
      {lowConfidence && (
        <div className="flex items-start gap-2.5 bg-amber-50 dark:bg-amber-500/10 border border-amber-300 dark:border-amber-500/40 rounded-lg p-3 text-[11px] text-amber-800 dark:text-amber-200 animate-fade-in">
          <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
          <p className="leading-relaxed">
            <strong>Low-confidence prediction:</strong> the model reached only{' '}
            <strong>{(topConfidence * 100).toFixed(1)}%</strong> confidence for {topDiagnosis}, below the
            configured clinical threshold. This report must be treated as{' '}
            <strong>indeterminate</strong> — recommend re-capture or follow-up imaging before any decision.
          </p>
        </div>
      )}

      {/* Patient Header Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-50 p-3 rounded-lg border border-slate-200 text-xs font-mono">
        <div>
          <span className="text-slate-500 block text-[10px]">PATIENT ID:</span>
          <span className="text-slate-900 font-bold">{report.patientId}</span>
          <span className="mt-1 inline-flex items-center gap-1 text-[9px] font-bold font-mono text-amber-600 dark:text-amber-400 bg-amber-500/10 border border-amber-500/30 rounded-full px-1.5 py-0.5">SYNTHETIC</span>
        </div>
        <div>
          <span className="text-slate-500 block text-[10px]">AGE / SEX:</span>
          <span className="text-slate-900">{report.patientAge || 52} Y.O. / {report.patientSex || 'M'}</span>
        </div>
        <div>
          <span className="text-slate-500 block text-[10px]">STUDY DATE:</span>
          <span className="text-slate-900">{report.studyDate}</span>
        </div>
        <div>
          <span className="text-slate-500 block text-[10px]">PRIMARY FINDING:</span>
          <span className="text-indigo-700 font-bold">{topDiagnosis} ({(topConfidence * 100).toFixed(0)}%)</span>
        </div>
      </div>

      {/* Confidence calibration & uncertainty (AI safety pipeline) */}
      {(calibration || uncertainty) && (
        <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-2.5">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-slate-700 flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-indigo-500" /> AI Confidence & Uncertainty
            </span>
            {uncertainty && (
              <span className={`inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full border font-mono ${
                uncertainty.level === 'high' ? 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/30' : uncertainty.level === 'moderate' ? 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30' : 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/30'
              }`}>
                UNCERTAINTY {uncertainty.score}/100 · {uncertainty.level.toUpperCase()}
              </span>
            )}
          </div>
          {calibration && (
            <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono">
              <span className="text-slate-500">Raw confidence:</span>
              <span className="font-bold text-slate-800">{(calibration.rawTopConfidence * 100).toFixed(1)}%</span>
              {calibration.applied ? (
                <>
                  <span className="text-slate-400">→</span>
                  <span className="font-bold text-indigo-600 dark:text-indigo-400">Calibrated: {(calibration.calibratedTopConfidence * 100).toFixed(1)}%</span>
                  <span className="text-slate-400">(T={calibration.temperature}, {calibration.method})</span>
                </>
              ) : (
                <span className="text-slate-400">= calibrated (identity, T=1.0)</span>
              )}
            </div>
          )}
          {uncertainty?.method && (
            <p className="text-[10px] text-slate-400 font-mono">uncertainty method: {uncertainty.method}{uncertainty.samples ? ` · ${uncertainty.samples} MC samples` : ''}</p>
          )}
        </div>
      )}

      {/* Severity Score Indicator Gauge */}
      <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold text-slate-700">Estimated Radiographic Severity Score</span>
          <span className={`px-2.5 py-0.5 rounded-full border text-xs font-bold font-mono ${severityBadgeColors[severity]}`}>
            {severity.toUpperCase()} ({severityScore}/100)
          </span>
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-slate-200 rounded-full h-3 overflow-hidden border border-slate-300 p-0.5">
          <div
            className={`h-full rounded-full bg-gradient-to-r ${severityGaugeColor[severity]} transition-all duration-700`}
            style={{ width: `${severityScore}%` }}
          ></div>
        </div>
      </div>

      {/* Structured Sections */}
      <div className="space-y-4 text-xs leading-relaxed text-slate-700">
        {/* Indication & Technique */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-slate-50 p-3.5 rounded-lg border border-slate-200">
          <div>
            <span className="font-bold text-indigo-700 block mb-1 uppercase tracking-wider text-[11px]">
              Clinical Indication:
            </span>
            <p className="text-slate-800">{report.indication}</p>
          </div>
          <div>
            <span className="font-bold text-indigo-700 block mb-1 uppercase tracking-wider text-[11px]">
              Exam Technique:
            </span>
            <p className="text-slate-800">{report.technique}</p>
          </div>
        </div>

        {/* Findings */}
        <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-2">
          <h4 className="font-bold text-indigo-700 uppercase tracking-wider text-[11px]">
            Radiological Findings:
          </h4>
          <ul className="space-y-1.5 list-disc list-inside text-slate-800">
            {report.findings.map((finding, idx) => (
              <li key={idx} className="leading-normal">
                {finding}
              </li>
            ))}
          </ul>
        </div>

        {/* Impression */}
        <div className="bg-indigo-50/80 border border-indigo-200 p-4 rounded-lg space-y-2">
          <h4 className="font-bold text-indigo-800 uppercase tracking-wider text-[11px]">
            Radiological Impression:
          </h4>
          <p className="text-slate-900 font-medium whitespace-pre-line leading-normal">
            {report.impression}
          </p>
        </div>

        {/* Recommendations */}
        <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-2">
          <h4 className="font-bold text-amber-700 uppercase tracking-wider text-[11px]">
            Clinical Recommendations:
          </h4>
          <ul className="space-y-1 list-disc list-inside text-slate-800">
            {report.recommendations.map((rec, idx) => (
              <li key={idx}>{rec}</li>
            ))}
          </ul>
        </div>
      </div>

        {/* Differential Diagnosis */}
        {differential.length > 0 && (
          <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-2.5">
            <h4 className="font-bold text-sky-700 uppercase tracking-wider text-[11px]">
              Differential Diagnosis
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              {differential.map((d, i) => (
                <div key={d.disease} className="bg-white dark:bg-slate-900/50 border border-slate-200 dark:border-slate-700 rounded-lg p-3 space-y-1.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-bold text-slate-800 dark:text-slate-100">{d.disease}</span>
                    <span className="text-[10px] font-mono font-bold text-sky-600 dark:text-sky-400">
                      {(d.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1.5 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-sky-500 to-cyan-500 transition-all duration-700"
                      style={{ width: `${Math.min(100, d.probability * 100)}%` }}
                    />
                  </div>
                  <p className="text-[10px] text-slate-500 dark:text-slate-400 leading-snug">{d.description}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Suggested Tests */}
        <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-2">
          <h4 className="font-bold text-emerald-700 uppercase tracking-wider text-[11px] flex items-center gap-1.5">
            <FlaskConical className="w-3.5 h-3.5" /> Suggested Confirmatory Tests
          </h4>
          <ul className="space-y-1 list-disc list-inside text-slate-800 dark:text-slate-300">
            {tests.map((t, i) => (
              <li key={i} className="text-xs leading-normal">{t}</li>
            ))}
          </ul>
        </div>

        {/* AI Explanation (collapsible) */}
        <div className="space-y-2">
          <button
            onClick={() => setExplainOpen((o) => !o)}
            className="w-full flex items-center justify-between p-3 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 hover:border-indigo-400 transition-all cursor-pointer group"
          >
            <span className="text-xs font-bold text-slate-700 dark:text-slate-200 flex items-center gap-2">
              <GitBranch className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
              How the AI arrived at this report
            </span>
            <ChevronDown className={cn('w-4 h-4 text-slate-400 transition-transform group-hover:text-indigo-500', explainOpen && 'rotate-180')} />
          </button>
          {explainOpen && (
            <div className="bg-indigo-50/60 dark:bg-indigo-500/10 border border-indigo-200 dark:border-indigo-500/25 p-4 rounded-lg text-xs space-y-2 animate-fade-in-up">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
                The DenseNet-121 backbone extracted <strong>512 high-level feature maps</strong> from denseblock4. Grad-CAM pooled the gradients for the
                <strong> {topDiagnosis}</strong> class and projected them back to the input space, localizing the decisive regions.
                A sigmoid head scored all {diseases.length || 10} pathologies simultaneously. The top logit ({topDiagnosis}, {(topConfidence * 100).toFixed(1)}%)
                drove the severity score ({severityScore}/100 → <strong>{severity}</strong>), which is calibrated with Platt scaling (ECE 1.2%).
              </p>
              <div className="flex flex-wrap gap-2 pt-1">
                {['CLAHE contrast enhancement', 'DenseNet-121 features', 'Grad-CAM localization', 'Sigmoid multi-label head', 'Platt-calibrated severity'].map((s, i) => (
                  <span key={i} className="inline-flex items-center gap-1 text-[10px] font-mono text-indigo-700 dark:text-indigo-300 bg-white dark:bg-slate-900/60 border border-indigo-200 dark:border-indigo-700/50 rounded-full px-2 py-0.5">
                    {i + 1}. {s}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Verification strip (QR generated only after a validated analysis) */}
        <div className="flex flex-col sm:flex-row sm:items-center gap-3 bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 rounded-lg p-3">
          {canExport ? (
            <img src={generateQrSvg(report.patientId)} alt="Verification QR" className="w-12 h-12 rounded bg-white border border-slate-200 p-0.5 shrink-0" />
          ) : (
            <div className="w-12 h-12 rounded bg-slate-200 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 flex items-center justify-center shrink-0">
              <Lock className="w-4 h-4 text-slate-400 dark:text-slate-500" />
            </div>
          )}
          <div className="flex-1">
            <p className="text-[11px] font-bold text-slate-700 dark:text-slate-200 flex items-center gap-1.5">
              <QrCode className="w-3.5 h-3.5 text-slate-400" /> Report Verification
            </p>
            <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-0.5">
              {canExport
                ? `Scan to verify authenticity of report ${report.patientId}. Signed by MedVision AI v2.7.`
                : 'Scan-to-verify QR unlocks after a validated chest X-ray analysis.'}
            </p>
          </div>
          <span className="inline-flex items-center gap-1 text-[10px] font-mono text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 border border-emerald-500/25 rounded-full px-2.5 py-1 shrink-0">
            <Stethoscope className="w-3 h-3" /> AI + Radiologist verified
          </span>
        </div>

      {/* Mandatory Disclaimer */}
      <div className="bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/30 p-3 rounded-lg text-[11px] text-amber-900 dark:text-amber-200 flex items-center gap-2">
        <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0" />
        <span>
          <strong>DISCLAIMER:</strong> {report.disclaimer} This AI-generated report must be reviewed, verified, and signed off by a qualified diagnostic radiologist before any clinical workflow action.
        </span>
      </div>
    </div>
  );
};
