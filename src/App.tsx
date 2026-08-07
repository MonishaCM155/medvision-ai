import React, { useState, Suspense, lazy, useEffect } from 'react';
import { AlertTriangle, ShieldCheck, ScanLine, Activity, Gauge, FlaskConical } from 'lucide-react';
import { api } from './services/api';
import { Header, NavTab } from './components/Header';
import { Footer } from './components/Footer';
import { Sidebar } from './components/Sidebar';
import { XrayUploader } from './components/XrayUploader';
import { GradCamViewer } from './components/GradCamViewer';
import { ConfidenceChart } from './components/ConfidenceChart';
import { ReportCard } from './components/ReportCard';
import { PdfExportModal } from './components/PdfExportModal';
import { HistoryView } from './components/HistoryView';
import { BatchPredictionView } from './components/BatchPredictionView';
import { ModelComparisonView } from './components/ModelComparisonView';
import { MLOpsView } from './components/MLOpsView';
import { DocsView } from './components/DocsView';
import { AnalyticsCenterView } from './components/AnalyticsCenterView';
import { KnowledgeCenterView } from './components/KnowledgeCenterView';
import { DoctorWorkspaceView } from './components/DoctorWorkspaceView';
import { MedicalCopilot } from './components/MedicalCopilot';
import { SAMPLE_XRAYS } from './data/sampleXrays';
import { PredictionResult } from './types';
import { AppProvider, useApp } from './contexts/AppContext';
import { ErrorBoundary } from './components/ErrorBoundary';
import { ToastHost } from './components/ToastHost';
import { CommandPalette } from './components/CommandPalette';
import { ShortcutsHelp } from './components/ShortcutsHelp';
import { PipelineProgress, PipelineStatus } from './components/PipelineProgress';
import { ValidationReport } from './utils/imageValidation';
import { SafetyReport } from './types';
import { cn } from './utils/cn';

const IMAGE_TYPE_LABELS: Record<string, string> = {
  chest_xray: 'Chest X-ray',
  other_xray: 'Other X-ray',
  ct: 'CT',
  mri: 'MRI',
  ultrasound: 'Ultrasound',
  pet: 'PET',
  mammography: 'Mammography',
  photograph: 'Photograph',
  animal: 'Animal',
  document: 'Document',
  unknown: 'Unknown',
};

const OOD_META: Record<string, { label: string; cls: string }> = {
  in: { label: 'In distribution', cls: 'text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 border-emerald-500/30' },
  borderline: { label: 'Borderline OOD', cls: 'text-amber-600 dark:text-amber-400 bg-amber-500/10 border-amber-500/30' },
  out: { label: 'Out of distribution', cls: 'text-rose-600 dark:text-rose-400 bg-rose-500/10 border-rose-500/30' },
};

// Lazy-loaded enterprise pages (code-splitting for faster initial load)
const Dashboard = lazy(() => import('./pages/Dashboard').then((m) => ({ default: m.Dashboard })));
const Patients = lazy(() => import('./pages/Patients').then((m) => ({ default: m.Patients })));
const ModelHub = lazy(() => import('./pages/ModelHub').then((m) => ({ default: m.ModelHub })));
const Training = lazy(() => import('./pages/Training').then((m) => ({ default: m.Training })));
const Datasets = lazy(() => import('./pages/Datasets').then((m) => ({ default: m.Datasets })));
const Settings = lazy(() => import('./pages/Settings').then((m) => ({ default: m.Settings })));

const PageSkeleton: React.FC = () => (
  <div className="space-y-6 animate-fade-in">
    <div className="skeleton h-10 w-64" />
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="skeleton h-28" />
      ))}
    </div>
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2 skeleton h-80" />
      <div className="skeleton h-80" />
    </div>
  </div>
);

function AppContent() {
  const { sidebarCollapsed, toggleTheme, pushNotification, confidenceThreshold } = useApp();
  const [activeTab, setActiveTab] = useState<NavTab>('dashboard');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [currentResult, setCurrentResult] = useState<PredictionResult | null>(null);
  const [history, setHistory] = useState<PredictionResult[]>([]);
  const [showPdfModal, setShowPdfModal] = useState<boolean>(false);
  // Copilot starts minimized (FAB) — opened only by explicit user action
  const [isCopilotOpen, setIsCopilotOpen] = useState<boolean>(false);
  const [mobileNavOpen, setMobileNavOpen] = useState<boolean>(false);
  const [shortcutsOpen, setShortcutsOpen] = useState<boolean>(false);
  const [inferenceError, setInferenceError] = useState<string | null>(null);
  const [bookmarkedIds, setBookmarkedIds] = useState<Set<string>>(new Set());
  // Analysis pipeline: Upload → Validation → Quality → AI Analysis → Explainability → Report
  const [pipelineStage, setPipelineStage] = useState<number>(0);
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus>('idle');
  const [pipelineMessage, setPipelineMessage] = useState<string | undefined>(undefined);
  // Report export (PDF/DOCX/QR) unlocks only after a validated chest X-ray analysis
  const [hasValidAnalysis, setHasValidAnalysis] = useState<boolean>(false);
  // Per-result export gating: a specific completed analysis id must match, and
  // confidence/uncertainty gates must pass (Phase 14 + stale-state protection).
  const [validatedResultId, setValidatedResultId] = useState<string | null>(null);
  // Server-side AI safety gate result (type classifier + OOD + quality)
  const [safetyReport, setSafetyReport] = useState<SafetyReport | null>(null);

  // Demo/sample results are explicitly labelled synthetic — the uncertainty
  // gate (Phase 14) applies to authoritative real inference; demo exports stay
  // available so the sample workflow remains usable. Confidence gating applies
  // to both.
  const isDemoResult =
    currentResult?.workflow === 'sample' || currentResult?.engine?.engineMode === 'demo-engine';
  const canExportAnalysis =
    hasValidAnalysis &&
    !!currentResult &&
    currentResult.id === validatedResultId &&
    currentResult.topConfidence >= confidenceThreshold &&
    (isDemoResult || currentResult.uncertainty?.level !== 'high');

  // Global keyboard shortcuts
  useEffect(() => {
    const isTyping = (t: EventTarget | null) => {
      const el = t as HTMLElement | null;
      return !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable);
    };

    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') return; // handled by palette
      // Never hijack browser/OS chords (Ctrl+T new tab, Ctrl+C copy, etc.) or auto-repeat
      if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
      if (isTyping(e.target)) return;
      switch (e.key.toLowerCase()) {
        case 'c':
          setIsCopilotOpen((o) => !o);
          break;
        case 't':
          window.dispatchEvent(new CustomEvent('medvision:toggle-theme'));
          break;
        case 'n':
          setActiveTab('inference');
          window.scrollTo({ top: 0, behavior: 'smooth' });
          break;
        case '/': {
          e.preventDefault();
          const searchInput = document.getElementById('global-search-input');
          // Visible (≥md screens)? Focus it. Otherwise open the command palette.
          if (searchInput && searchInput.offsetParent !== null) {
            searchInput.focus();
          } else {
            window.dispatchEvent(new CustomEvent('medvision:open-palette'));
          }
          break;
        }
        case '?':
          setShortcutsOpen((o) => !o);
          break;
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  // Command palette copilot toggle bridge
  useEffect(() => {
    const toggle = () => setIsCopilotOpen((o) => !o);
    const theme = () => toggleTheme();
    window.addEventListener('medvision:toggle-copilot', toggle);
    window.addEventListener('medvision:toggle-theme', theme);
    return () => {
      window.removeEventListener('medvision:toggle-copilot', toggle);
      window.removeEventListener('medvision:toggle-theme', theme);
    };
  }, [toggleTheme]);

  // Initialize with initial sample prediction result on mount
  useEffect(() => {
    const initialSample = SAMPLE_XRAYS[0].sampleResult;
    const initialResult: PredictionResult = {
      id: 'sample-init-001',
      timestamp: new Date().toISOString(),
      ...initialSample,
      // Explicit demo provenance so the DEMO banner renders for the on-load sample.
      workflow: 'sample',
      validationSource: 'sample-demo',
      engine: { ...(initialSample.engine || {}), engineMode: 'demo-engine', source: 'demo', predictionSource: 'demo-profile' },
    };
    setCurrentResult(initialResult);
    setHistory([initialResult]);
  }, []);

  const handleRunInference = async (data: {
    imageName: string;
    imageData: string;
    model: string;
    clahe: boolean;
    noiseRemoval: boolean;
    validation: ValidationReport;
  }) => {
    setIsLoading(true);
    setInferenceError(null);
    setSafetyReport(null);
    // Stale-state protection: any new image invalidates the previous
    // prediction/report/exports until the new study completes the pipeline.
    setCurrentResult(null);
    setHasValidAnalysis(false);
    setValidatedResultId(null);

    // Stage 1 — Chest X-ray validation (client-side gate)
    setPipelineStatus('running');
    setPipelineStage(1);
    setPipelineMessage('Running AI chest X-ray validation & quality assessment…');
    await new Promise((r) => setTimeout(r, 450));

    if (!data.validation.passed) {
      setPipelineStatus('failed');
      setPipelineMessage(data.validation.message);
      pushNotification({ kind: 'warning', title: 'Image rejected', body: 'Only valid frontal chest X-rays can be analyzed. See the validation panel for details.' });
      setIsLoading(false);
      return; // No disease prediction on invalid images
    }

    // Stage 2 — Quality + AI safety gate (server-side classifier/OOD when the engine is online)
    setPipelineStage(2);
    setPipelineMessage('Querying the AI safety gate — image type, out-of-distribution & quality scoring…');
    api
      .validateImage({
        imageName: data.imageName,
        imageData: data.imageData,
        clientValidation: { passed: true, score: data.validation.score },
      })
      .then((report) => setSafetyReport(report))
      .catch(() => setSafetyReport(null));
    await new Promise((r) => setTimeout(r, 350));

    // Stage 3 — AI analysis
    setPipelineStage(3);
    setPipelineMessage(`Running ${data.model} inference with Grad-CAM explainability…`);

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // NOTE: client validation is advisory only — the server performs its
        // own authoritative validation and refuses unsafe requests (422/503).
        body: JSON.stringify({
          imageName: data.imageName,
          imageData: data.imageData,
          model: data.model,
          clahe: data.clahe,
          noiseRemoval: data.noiseRemoval,
        }),
      });

      // Server-side rejection (422) — no fallback: the image itself was refused
      if (res.status === 422) {
        let reason = 'Image rejected by server-side validation.';
        try {
          const body = await res.json();
          if (body?.error) reason = body.error;
        } catch {
          /* ignore */
        }
        setPipelineStatus('failed');
        setPipelineMessage(reason);
        pushNotification({ kind: 'warning', title: 'Image rejected', body: reason });
        setIsLoading(false);
        return;
      }

      // Engine unavailable (503/502) — NO prediction, NO demo fallback, NO report.
      if (res.status === 503 || res.status === 502) {
        let reason = 'AI inference is temporarily unavailable. No prediction was generated.';
        try {
          const body = await res.json();
          if (body?.message) reason = body.message;
        } catch {
          /* ignore */
        }
        setPipelineStatus('failed');
        setPipelineMessage(reason);
        setInferenceError(reason);
        pushNotification({ kind: 'warning', title: 'Engine unavailable', body: 'No prediction was generated — retry when the inference engine is back online.' });
        setIsLoading(false);
        return;
      }

      if (!res.ok) {
        throw new Error(`Server error: ${res.statusText}`);
      }

      const result: PredictionResult = await res.json();
      setCurrentResult(result);
      setHistory((prev) => [result, ...prev]);
      setHasValidAnalysis(true);
      setValidatedResultId(result.id);
      if (result.workflow === 'sample' || result.engine?.engineMode === 'demo-engine') {
        pushNotification({ kind: 'info', title: 'Demo study analyzed', body: 'Sample studies produce a clearly-labelled demo profile — not real model inference.' });
      }

      // Stages 4-5 — Explainability + Report complete
      setPipelineStage(5);
      setPipelineStatus('passed');
      setPipelineMessage(`Analysis complete — ${result.topDiagnosis} at ${(result.topConfidence * 100).toFixed(1)}% confidence. Grad-CAM map and report ready.`);
      // Copilot intentionally NOT auto-opened — it stays minimized until the user clicks it
    } catch (err) {
      console.error('Inference error:', err);
      // The frontend NEVER fabricates a diagnosis. A backend failure surfaces as
      // engine-unavailable — no prediction, no report, no export unlocks.
      const reason = 'AI inference is temporarily unavailable. No prediction was generated.';
      setPipelineStatus('failed');
      setPipelineMessage(reason);
      setInferenceError(reason);
      pushNotification({ kind: 'warning', title: 'Engine unavailable', body: 'The medical inference engine could not be reached. No prediction was generated — retry when it is back online.' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteHistoryItem = (id: string) => {
    setHistory((prev) => prev.filter((item) => item.id !== id));
  };

  const handleSelectHistoryItem = (item: PredictionResult) => {
    setCurrentResult(item);
    // History items are previous server responses — exports follow the same
    // per-result gates (confidence/uncertainty re-checked via canExportAnalysis).
    setValidatedResultId(item.id);
    setHasValidAnalysis(true);
    setActiveTab('inference');
  };

  const handleToggleBookmark = (id: string) => {
    setBookmarkedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleNavigate = (tab: NavTab) => {
    setActiveTab(tab);
    setMobileNavOpen(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen text-slate-800 dark:text-slate-100 flex flex-col font-sans">
      {/* Ambient animated backdrop */}
      <div className="app-backdrop" />

      {/* Top Navbar */}
      <Header
        activeTab={activeTab}
        setActiveTab={handleNavigate}
        onToggleSidebar={() => setMobileNavOpen((o) => !o)}
        onToggleCopilot={() => setIsCopilotOpen((o) => !o)}
        isCopilotOpen={isCopilotOpen}
      />

      <div className="flex flex-1">
        {/* Desktop sidebar */}
        <Sidebar activeTab={activeTab} onNavigate={handleNavigate} onToggleCopilot={() => setIsCopilotOpen((o) => !o)} isCopilotOpen={isCopilotOpen} />

        {/* Mobile nav drawer */}
        {mobileNavOpen && (
          <div className="lg:hidden fixed inset-0 z-40 bg-slate-950/60 backdrop-blur-sm animate-fade-in" onClick={() => setMobileNavOpen(false)}>
            <Sidebar
              variant="mobile"
              activeTab={activeTab}
              onNavigate={handleNavigate}
              onCloseMobile={() => setMobileNavOpen(false)}
              onToggleCopilot={() => {
                setIsCopilotOpen((o) => !o);
                setMobileNavOpen(false);
              }}
              isCopilotOpen={isCopilotOpen}
            />
          </div>
        )}

        {/* Main Body Stage (copilot now floats as an overlay window) */}
        <div className="flex-1 min-w-0">
          <main
            className={cn(
              'max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 w-full space-y-6',
              sidebarCollapsed ? 'lg:pl-20' : 'lg:pl-60'
            )}
          >
            <Suspense fallback={<PageSkeleton />}>
              <div key={activeTab} className="animate-fade-in">
                {/* NEW: Enterprise Dashboard */}
                {activeTab === 'dashboard' && <Dashboard onNavigate={handleNavigate} />}

                {/* TAB 1: INFERENCE WORKSPACE */}
                {activeTab === 'inference' && (
                  <div className="space-y-6">
                    {/* Pipeline progress: Upload → Validation → Quality → AI Analysis → Explainability → Report */}
                    <PipelineProgress stage={pipelineStage} status={pipelineStatus} message={pipelineMessage} />

                    {/* AI Safety Gate summary (type classifier + OOD + quality) */}
                    {safetyReport && (
                      <div className="rounded-2xl border border-slate-200 dark:border-slate-700/80 bg-white/70 dark:bg-slate-900/50 backdrop-blur-sm p-4 space-y-3 animate-fade-in-up">
                        <div className="flex items-center justify-between gap-3 flex-wrap">
                          <span className="text-[11px] font-black uppercase tracking-widest text-slate-500 dark:text-slate-400 flex items-center gap-2">
                            <ShieldCheck className="w-4 h-4 text-emerald-500" />
                            AI Safety Gate
                          </span>
                          <span className="text-[10px] font-mono text-slate-400">
                            {safetyReport.source === 'pytorch-engine' ? 'PYTORCH ENGINE' : safetyReport.source === 'client-heuristic' ? 'CLIENT HEURISTIC' : '—'}
                          </span>
                        </div>
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-[11px]">
                          {/* Image type classification */}
                          <div className="rounded-xl border border-slate-200 dark:border-slate-700/70 bg-slate-50 dark:bg-slate-800/40 p-3">
                            <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5 mb-2">
                              <ScanLine className="w-3.5 h-3.5 text-indigo-500" /> Image type
                            </p>
                            {safetyReport.type ? (
                              <>
                                <p className="font-black text-slate-800 dark:text-slate-100">
                                  {IMAGE_TYPE_LABELS[safetyReport.type.predicted] ?? safetyReport.type.predicted}
                                </p>
                                <p className="text-[10px] text-slate-400 mt-1 font-mono">
                                  classifier: {safetyReport.type.method} · top: {Math.round((safetyReport.type.confidences?.[safetyReport.type.predicted] ?? 0) * 100)}%
                                </p>
                              </>
                            ) : (
                              <p className="text-slate-400">Not available (engine offline)</p>
                            )}
                          </div>
                          {/* OOD */}
                          <div className="rounded-xl border border-slate-200 dark:border-slate-700/70 bg-slate-50 dark:bg-slate-800/40 p-3">
                            <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5 mb-2">
                              <Activity className="w-3.5 h-3.5 text-amber-500" /> Out-of-distribution
                            </p>
                            {safetyReport.ood ? (
                              <span className={`inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full border ${OOD_META[safetyReport.ood.verdict]?.cls ?? 'text-slate-400 border-slate-400/30'}`}>
                                {OOD_META[safetyReport.ood.verdict]?.label ?? safetyReport.ood.verdict} · {safetyReport.ood.score}/100
                              </span>
                            ) : (
                              <p className="text-slate-400">Passed client-side check</p>
                            )}
                          </div>
                          {/* Quality score */}
                          <div className="rounded-xl border border-slate-200 dark:border-slate-700/70 bg-slate-50 dark:bg-slate-800/40 p-3">
                            <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5 mb-2">
                              <Gauge className="w-3.5 h-3.5 text-emerald-500" /> Quality score
                            </p>
                            {safetyReport.quality ? (
                              <p className="font-black text-slate-800 dark:text-slate-100">
                                {safetyReport.quality.score}/100
                                <span className="text-[10px] font-mono text-slate-400 ml-1.5">threshold {safetyReport.quality.threshold ?? 55}</span>
                              </p>
                            ) : (
                              <p className="text-slate-400">—</p>
                            )}
                          </div>
                        </div>
                        {safetyReport.note && <p className="text-[10px] text-slate-400 leading-relaxed">{safetyReport.note}</p>}
                      </div>
                    )}

                    {inferenceError && (
                      <div className="flex items-start gap-2.5 bg-rose-500/10 border border-rose-500/30 rounded-xl p-3.5 text-xs text-rose-700 dark:text-rose-300 animate-fade-in">
                        <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                        <div>
                          <p className="font-bold">Inference degraded</p>
                          <p className="mt-0.5 opacity-90">{inferenceError}</p>
                        </div>
                      </div>
                    )}
                    <XrayUploader onRunInference={handleRunInference} isLoading={isLoading} />

                    {/* Low-confidence warning: never present a weak prediction as definitive */}
                    {currentResult && currentResult.topConfidence < confidenceThreshold && (
                      <div className="flex items-start gap-2.5 bg-amber-500/10 border border-amber-500/40 rounded-xl p-3.5 text-xs text-amber-800 dark:text-amber-200 animate-fade-in">
                        <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-amber-600 dark:text-amber-400" />
                        <div className="space-y-1">
                          <p className="font-bold">AI confidence too low — result is indeterminate</p>
                          <p className="opacity-90 leading-relaxed">
                            Top prediction <strong>{currentResult.topDiagnosis}</strong> reached only{' '}
                            <strong>{(currentResult.topConfidence * 100).toFixed(1)}%</strong> confidence — below the{' '}
                            <strong>{Math.round(confidenceThreshold * 100)}%</strong> clinical threshold. This is{' '}
                            <strong>not a definitive diagnosis</strong>. Consider re-capturing with better technique,
                            higher resolution, or requesting a follow-up study.
                          </p>
                        </div>
                      </div>
                    )}

                    {currentResult && (
                      <div className="space-y-6 animate-fade-in">
                        {/* Explicit demo/sample banner — never implied as real inference */}
                        {(currentResult.workflow === 'sample' || currentResult.engine?.engineMode === 'demo-engine') && (
                          <div className="flex items-start gap-2.5 bg-fuchsia-500/10 border border-fuchsia-500/30 rounded-xl p-3.5 text-xs text-fuchsia-700 dark:text-fuchsia-300 animate-fade-in">
                            <FlaskConical className="w-4 h-4 shrink-0 mt-0.5" />
                            <p>
                              <strong>DEMO SAMPLE ANALYSIS</strong> — this result is a synthetic demo profile
                              (engine: demo-engine · validationSource: sample-demo), not real model inference.
                              For demonstration only — not a clinical diagnosis.
                            </p>
                          </div>
                        )}
                        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                          <GradCamViewer result={currentResult} />
                          <ConfidenceChart diseases={currentResult.diseases} />
                        </div>

                        <ReportCard
                          report={currentResult.report}
                          severity={currentResult.severity}
                          severityScore={currentResult.severityScore}
                          topDiagnosis={currentResult.topDiagnosis}
                          topConfidence={currentResult.topConfidence}
                          diseases={currentResult.diseases}
                          onExportPdf={() => setShowPdfModal(true)}
                          canExport={canExportAnalysis}
                          lowConfidence={currentResult.topConfidence < confidenceThreshold}
                        />
                      </div>
                    )}
                  </div>
                )}

                {/* TAB 2: BATCH MODE */}
                {activeTab === 'batch' && <BatchPredictionView />}

                {/* TAB 3: ANALYTICS & GPU CENTER */}
                {activeTab === 'analytics' && <AnalyticsCenterView />}

                {/* TAB 4: DICOM STUDIO & DOCTOR WORKSPACE */}
                {activeTab === 'doctor' && (
                  <DoctorWorkspaceView currentResult={currentResult} onSelectResult={handleSelectHistoryItem} />
                )}

                {/* TAB 5: KNOWLEDGE HUB */}
                {activeTab === 'knowledge' && <KnowledgeCenterView />}

                {/* TAB 6: HISTORY */}
                {activeTab === 'history' && (
                  <HistoryView
                    history={history}
                    bookmarkedIds={bookmarkedIds}
                    onToggleBookmark={handleToggleBookmark}
                    onSelectResult={handleSelectHistoryItem}
                    onDeleteResult={handleDeleteHistoryItem}
                  />
                )}

                {/* TAB 7: MODELS & BENCHMARKS */}
                {activeTab === 'models' && <ModelComparisonView />}

                {/* TAB 8: MLOPS & CALIBRATION */}
                {activeTab === 'mlops' && <MLOpsView />}

                {/* TAB 9: DOCS & ARCHITECTURE */}
                {activeTab === 'docs' && <DocsView />}

                {/* NEW: PATIENTS */}
                {activeTab === 'patients' && <Patients />}

                {/* NEW: MODEL HUB */}
                {activeTab === 'modelhub' && <ModelHub />}

                {/* NEW: TRAINING STUDIO */}
                {activeTab === 'training' && <Training />}

                {/* NEW: DATASET REGISTRY */}
                {activeTab === 'datasets' && <Datasets />}

                {/* NEW: SETTINGS */}
                {activeTab === 'settings' && <Settings />}
              </div>
            </Suspense>
          </main>
        </div>
      </div>

      {/* Global command palette (Ctrl/Cmd+K) */}
      <CommandPalette onNavigate={handleNavigate} onNewScan={() => window.scrollTo({ top: 0, behavior: 'smooth' })} />

      {/* Keyboard shortcuts help (?) */}
      <ShortcutsHelp open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />

      {/* Visible toast notifications */}
      <ToastHost />

      {/* Floating AI Medical Copilot — starts minimized, opens on demand */}
      <MedicalCopilot
        predictionResult={currentResult}
        onExportPdf={() => setShowPdfModal(true)}
        isOpen={isCopilotOpen}
        onToggleOpen={() => setIsCopilotOpen(!isCopilotOpen)}
        canExport={canExportAnalysis}
      />

      {/* Printable Hospital PDF Modal (locked until a validated analysis exists) */}
      {showPdfModal && currentResult && canExportAnalysis && (
        <PdfExportModal result={currentResult} onClose={() => setShowPdfModal(false)} />
      )}

      {/* Footer */}
      <div>
        <Footer />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <AppProvider>
        <AppContent />
      </AppProvider>
    </ErrorBoundary>
  );
}
