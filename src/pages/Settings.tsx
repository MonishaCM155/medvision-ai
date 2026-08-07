import React, { useState } from 'react';
import {
  Palette, Cpu, Bell, FileText, User, Accessibility, Languages, Check, Save,
  Monitor, Moon, Sun, ShieldCheck, Sparkles, Globe, FlaskConical,
} from 'lucide-react';
import { useApp } from '../contexts/AppContext';
import { GlassCard, SectionHeader } from '../components/ui/GlassCard';
import { cn } from '../utils/cn';

interface ToggleRowProps {
  label: string;
  description: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  accent?: string;
}

const ToggleRow: React.FC<ToggleRowProps> = ({ label, description, checked, onChange, accent = 'indigo' }) => {
  const colors: Record<string, string> = {
    indigo: 'bg-indigo-600',
    emerald: 'bg-emerald-600',
    amber: 'bg-amber-600',
    sky: 'bg-sky-600',
  };
  return (
    <div className="flex items-center justify-between gap-4 py-3 border-b border-slate-100 dark:border-slate-800 last:border-0">
      <div>
        <p className="text-xs font-bold text-slate-800 dark:text-slate-100">{label}</p>
        <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">{description}</p>
      </div>
      <button
        onClick={() => onChange(!checked)}
        className={cn('relative w-10 h-6 rounded-full transition-colors shrink-0 cursor-pointer', checked ? colors[accent] : 'bg-slate-200 dark:bg-slate-700')}
        role="switch"
        aria-checked={checked}
      >
        <span className={cn('absolute top-0.5 w-5 h-5 rounded-full bg-white shadow transition-all', checked ? 'left-[1.15rem]' : 'left-0.5')} />
      </button>
    </div>
  );
};

const SECTION_ICON = { className: 'w-4 h-4 text-primary-500' };

export const Settings: React.FC = () => {
  const { theme, setTheme, pushNotification, confidenceThreshold, setConfidenceThreshold } = useApp();
  const [notifToggle, setNotifToggle] = useState(true);
  const [emailToggle, setEmailToggle] = useState(false);
  const [soundToggle, setSoundToggle] = useState(true);
  const [autoExport, setAutoExport] = useState(false);
  const [highContrast, setHighContrast] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [language, setLanguage] = useState('English (US)');
  const [reportModel, setReportModel] = useState('Gemini 3.6 Flash');
  const [deployOnTrain, setDeployOnTrain] = useState(false);

  const handleSave = () => {
    pushNotification({ kind: 'success', title: 'Settings saved', body: 'Your preferences have been applied across the suite.' });
  };

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-black tracking-tight text-slate-900 dark:text-white">Workspace Settings</h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Personalize MedVision AI for your clinical and research workflow</p>
        </div>
        <button onClick={handleSave} className="btn-gradient text-white text-xs font-bold px-4 py-2.5 rounded-xl flex items-center gap-2 cursor-pointer">
          <Save className="w-4 h-4" /> Save Changes
        </button>
      </div>

      {/* Appearance */}
      <GlassCard>
        <SectionHeader icon={<Palette {...SECTION_ICON} />} title="Appearance & Theme" subtitle="Light, dark, or follow the clinical environment" />
        <div className="grid grid-cols-3 gap-3 mt-4">
          {([
            { id: 'light', label: 'Light', desc: 'Bright clinical UI', icon: <Sun className="w-5 h-5" /> },
            { id: 'dark', label: 'Dark', desc: 'Low-glare reading room', icon: <Moon className="w-5 h-5" /> },
            { id: 'system', label: 'System', desc: 'Follow OS setting', icon: <Monitor className="w-5 h-5" /> },
          ] as const).map((t) => (
            <button
              key={t.id}
              onClick={() => {
                if (t.id === 'system') {
                  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                  setTheme(prefersDark ? 'dark' : 'light');
                  pushNotification({ kind: 'info', title: 'System theme applied', body: `Following OS preference: ${prefersDark ? 'Dark' : 'Light'} mode.` });
                } else {
                  setTheme(t.id);
                }
              }}
              className={cn(
                'relative rounded-2xl border p-4 text-left transition-all cursor-pointer',
                theme === t.id ? 'border-primary-500 bg-primary-500/5 ring-2 ring-primary-400/40' : 'border-slate-200 dark:border-slate-700 hover:border-primary-300'
              )}
            >
              {theme === t.id && (
                <span className="absolute top-2.5 right-2.5 w-5 h-5 rounded-full bg-primary-600 text-white flex items-center justify-center">
                  <Check className="w-3 h-3" />
                </span>
              )}
              <span className={cn('w-10 h-10 rounded-xl flex items-center justify-center mb-2.5', theme === t.id ? 'bg-primary-600 text-white' : 'bg-slate-100 dark:bg-slate-800 text-slate-500')}>
                {t.icon}
              </span>
              <p className="text-xs font-bold text-slate-800 dark:text-slate-100">{t.label}</p>
              <p className="text-[10px] text-slate-500 mt-0.5">{t.desc}</p>
            </button>
          ))}
        </div>
      </GlassCard>

      {/* AI Models */}
      <GlassCard>
        <SectionHeader icon={<Cpu {...SECTION_ICON} />} title="AI Models" subtitle="Default engines for report generation and inference" />
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
          <label className="space-y-1.5">
            <span className="text-[11px] font-semibold text-slate-600 dark:text-slate-300">Report Generation Engine</span>
            <select
              value={reportModel}
              onChange={(e) => setReportModel(e.target.value)}
              className="w-full bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2.5 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-primary-400/50"
            >
              <option>Gemini 3.6 Flash</option>
              <option>Gemini 2.5 Pro</option>
              <option>Local Rule Engine (offline)</option>
            </select>
          </label>
          <label className="space-y-1.5">
            <span className="text-[11px] font-semibold text-slate-600 dark:text-slate-300">Default Inference Backbone</span>
            <select className="w-full bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2.5 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-primary-400/50">
              <option>DenseNet-121 (CheXNet)</option>
              <option>EfficientNet-B3</option>
              <option>Swin-B</option>
              <option>Stacked Ensemble</option>
            </select>
          </label>
        </div>
        {/* Configurable AI confidence threshold */}
        <div className="mt-4 rounded-2xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-4">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div className="min-w-0">
              <p className="text-xs font-bold text-slate-800 dark:text-slate-100">Minimum AI Confidence Threshold</p>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5 leading-relaxed">
                Predictions below this confidence are flagged as <strong>low confidence</strong> and are never presented as a
                definitive diagnosis. Warning: values above 90% will reject many real-world borderline scans.
              </p>
            </div>
            <span className="shrink-0 font-mono font-black text-sm text-primary-600 dark:text-primary-400 bg-primary-500/10 border border-primary-500/30 rounded-lg px-3 py-1.5">
              {Math.round(confidenceThreshold * 100)}%
            </span>
          </div>
          <input
            type="range"
            min="50"
            max="95"
            step="5"
            value={Math.round(confidenceThreshold * 100)}
            onChange={(e) => setConfidenceThreshold(Number(e.target.value) / 100)}
            className="w-full mt-3 accent-indigo-600 cursor-pointer"
            aria-label="Minimum AI confidence threshold"
          />
          <div className="flex justify-between text-[10px] font-mono text-slate-400 mt-1">
            <span>50% · permissive</span>
            <span>75% · recommended</span>
            <span>95% · strict</span>
          </div>
        </div>

        <div className="mt-3">
          <ToggleRow
            label="Auto-deploy on successful training"
            description="Promote runs that beat the current best AUROC to staging automatically"
            checked={deployOnTrain}
            onChange={setDeployOnTrain}
            accent="emerald"
          />
          <ToggleRow
            label="Explainability methods on by default"
            description="Always generate Grad-CAM alongside predictions"
            checked={voiceEnabled}
            onChange={setVoiceEnabled}
            accent="sky"
          />
        </div>
      </GlassCard>

      {/* Notifications */}
      <GlassCard>
        <SectionHeader icon={<Bell {...SECTION_ICON} />} title="Notifications" subtitle="Stay informed about scans, reports, and system health" />
        <div className="mt-2">
          <ToggleRow label="In-app notifications" description="Live activity feed and report completion alerts" checked={notifToggle} onChange={setNotifToggle} />
          <ToggleRow label="Email digests" description="Daily summary of pending reports and queue health" checked={emailToggle} onChange={setEmailToggle} />
          <ToggleRow label="Sound alerts" description="Audible cue on critical-case detection" checked={soundToggle} onChange={setSoundToggle} accent="amber" />
        </div>
      </GlassCard>

      {/* Export */}
      <GlassCard>
        <SectionHeader icon={<FileText {...SECTION_ICON} />} title="Export Preferences" subtitle="Report delivery formats" />
        <div className="mt-2">
          <ToggleRow label="Auto-export PDF with reports" description="Generate a hospital-formatted PDF for every completed report" checked={autoExport} onChange={setAutoExport} />
          <div className="pt-3 grid grid-cols-1 sm:grid-cols-3 gap-2">
            {['PDF (jsPDF)', 'DOCX (Word)', 'DICOM SR'].map((f) => (
              <span key={f} className="inline-flex items-center gap-1.5 text-[11px] font-semibold text-slate-600 dark:text-slate-300 bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2">
                <FileText className="w-3.5 h-3.5 text-indigo-500" /> {f}
              </span>
            ))}
          </div>
        </div>
      </GlassCard>

      {/* Accessibility & Language */}
      <GlassCard>
        <SectionHeader icon={<Accessibility {...SECTION_ICON} />} title="Accessibility & Language" subtitle="Inclusive experience for every clinician" />
        <div className="mt-2">
          <ToggleRow label="High-contrast mode" description="Enhanced contrast for reading-room displays" checked={highContrast} onChange={setHighContrast} accent="amber" />
          <ToggleRow label="Reduced motion" description="Minimize animations and transitions" checked={reducedMotion} onChange={setReducedMotion} />
          <ToggleRow label="Voice interactions" description="Speech-to-text queries and text-to-speech readout" checked={voiceEnabled} onChange={setVoiceEnabled} accent="sky" />
          <div className="pt-3 grid grid-cols-1 sm:grid-cols-2 gap-3">
            <label className="space-y-1.5">
              <span className="text-[11px] font-semibold text-slate-600 dark:text-slate-300 flex items-center gap-1.5">
                <Languages className="w-3.5 h-3.5" /> Interface Language
              </span>
              <select value={language} onChange={(e) => setLanguage(e.target.value)} className="w-full bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2.5 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-primary-400/50">
                <option>English (US)</option>
                <option>English (UK)</option>
                <option>Español</option>
                <option>Français</option>
                <option>Deutsch</option>
                <option>中文</option>
              </select>
            </label>
            <div className="flex items-end">
              <span className="inline-flex items-center gap-1.5 text-[11px] text-slate-400 border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2.5 w-full bg-slate-50 dark:bg-slate-800/60">
                <Globe className="w-4 h-4 text-sky-500" />
                Public research mode — no accounts, no PHI stored
              </span>
            </div>
          </div>
        </div>
      </GlassCard>

      {/* Public Research Mode */}
      <GlassCard>
        <SectionHeader icon={<Globe className="w-4 h-4 text-sky-500" />} title="Public Research Mode" subtitle="MedVision AI requires no login, accounts, or roles" />
        <div className="mt-4 space-y-3">
          <div className="flex items-start gap-3 rounded-2xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-4 text-[11px] leading-relaxed text-slate-600 dark:text-slate-300">
            <FlaskConical className="w-4 h-4 text-fuchsia-500 shrink-0 mt-0.5" />
            <p>
              The application opens directly into the workspace — there is <strong>no login, logout, user account,
              password, JWT, session, or role</strong>. All access is <strong>anonymous public access</strong>.
              This is an AI-assisted medical imaging <strong>research &amp; education</strong> platform, not a
              substitute for professional medical diagnosis.
            </p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-[10px] font-mono text-slate-400">
            {['No authentication', 'No PHI stored', 'All patient data synthetic'].map((t) => (
              <span key={t} className="inline-flex items-center gap-1.5 bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-500" /> {t}
              </span>
            ))}
          </div>
        </div>
      </GlassCard>

      {/* About */}
      <GlassCard>
        <SectionHeader icon={<Sparkles className="w-4 h-4 text-fuchsia-500" />} title="About MedVision AI" subtitle="Enterprise Edition 2.7.0 · Public Research Mode" />
        <div className="flex items-center gap-3 mt-4 text-[11px] text-slate-500 dark:text-slate-400">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-600 to-violet-600 text-white font-black flex items-center justify-center shadow-lg shadow-indigo-500/30">
            M
          </div>
          <p className="leading-relaxed">
            Explainable medical imaging suite for hospitals, radiologists, researchers & students. Built for education and research — not for clinical diagnosis.
          </p>
        </div>
      </GlassCard>
    </div>
  );
};
