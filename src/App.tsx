import React, { useState, Suspense, lazy, useEffect } from 'react';
import { AlertTriangle } from 'lucide-react';
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
import { cn } from './utils/cn';

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
  const { sidebarCollapsed, toggleTheme } = useApp();
  const [activeTab, setActiveTab] = useState<NavTab>('dashboard');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [currentResult, setCurrentResult] = useState<PredictionResult | null>(null);
  const [history, setHistory] = useState<PredictionResult[]>([]);
  const [showPdfModal, setShowPdfModal] = useState<boolean>(false);
  const [isCopilotOpen, setIsCopilotOpen] = useState<boolean>(true);
  const [mobileNavOpen, setMobileNavOpen] = useState<boolean>(false);
  const [shortcutsOpen, setShortcutsOpen] = useState<boolean>(false);
  const [inferenceError, setInferenceError] = useState<string | null>(null);
  const [bookmarkedIds, setBookmarkedIds] = useState<Set<string>>(new Set());

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
  }) => {
    setIsLoading(true);
    setInferenceError(null);

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          imageName: data.imageName,
          imageData: data.imageData,
          model: data.model,
          clahe: data.clahe,
          noiseRemoval: data.noiseRemoval,
        }),
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.statusText}`);
      }

      const result: PredictionResult = await res.json();
      setCurrentResult(result);
      setHistory((prev) => [result, ...prev]);
      setIsCopilotOpen(true); // Auto open copilot on new prediction
    } catch (err) {
      console.error('Inference error:', err);
      setInferenceError('Backend unreachable — showing local fallback analysis. Predictions are simulated for this demo.');
      // Fallback local prediction calculation if backend unavailable
      const fallbackResult: PredictionResult = {
        id: `pred_${Date.now()}`,
        timestamp: new Date().toISOString(),
        imageName: data.imageName,
        originalImageUrl: data.imageData,
        heatmapOverlayUrl: data.imageData,
        claheApplied: data.clahe,
        noiseRemovalApplied: data.noiseRemoval,
        modelUsed: `${data.model} (PyTorch + Grad-CAM)`,
        inferenceTimeMs: 148,
        diseases: SAMPLE_XRAYS[0].sampleResult.diseases,
        topDiagnosis: 'Pneumonia',
        topConfidence: 0.94,
        severity: 'High',
        severityScore: 82,
        gradCamRegions: SAMPLE_XRAYS[0].sampleResult.gradCamRegions,
        report: SAMPLE_XRAYS[0].sampleResult.report,
        keyMetrics: { snr: 28.5, resolution: '1024x1024', meanIntensity: 114.2, contrastRatio: 4.8 },
      };
      setCurrentResult(fallbackResult);
      setHistory((prev) => [fallbackResult, ...prev]);
      setIsCopilotOpen(true);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteHistoryItem = (id: string) => {
    setHistory((prev) => prev.filter((item) => item.id !== id));
  };

  const handleSelectHistoryItem = (item: PredictionResult) => {
    setCurrentResult(item);
    setActiveTab('inference');
    setIsCopilotOpen(true);
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

        {/* Main Body Stage with padding offset when Copilot is open */}
        <div className={cn('flex-1 transition-all duration-300 min-w-0', isCopilotOpen ? 'lg:mr-[380px] xl:mr-[420px]' : 'mr-0')}>
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

                    {currentResult && (
                      <div className="space-y-6 animate-fade-in">
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

      {/* Docked AI Medical Copilot Chatbot */}
      <MedicalCopilot
        predictionResult={currentResult}
        onExportPdf={() => setShowPdfModal(true)}
        isOpen={isCopilotOpen}
        onToggleOpen={() => setIsCopilotOpen(!isCopilotOpen)}
      />

      {/* Printable Hospital PDF Modal */}
      {showPdfModal && currentResult && <PdfExportModal result={currentResult} onClose={() => setShowPdfModal(false)} />}

      {/* Footer (offset so it clears the docked Copilot panel) */}
      <div className={cn('transition-all duration-300', isCopilotOpen ? 'lg:mr-[380px] xl:mr-[420px]' : 'mr-0')}>
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
