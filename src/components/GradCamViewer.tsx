import React, { useState, useMemo, useEffect } from 'react';
import { Eye, Layers, Download, Sliders, Sparkles, Target, ZoomIn, Info, ChevronDown, GitBranch, ScanSearch, Wand2, Loader2, FlaskConical } from 'lucide-react';
import { PredictionResult } from '../types';
import { SAMPLE_XRAYS } from '../data/sampleXrays';
import { EXPLAINABILITY_METHODS } from '../data/mockEnterprise';
import { api } from '../services/api';
import { cn } from '../utils/cn';

interface GradCamViewerProps {
  result: PredictionResult;
}

export const GradCamViewer: React.FC<GradCamViewerProps> = ({ result }) => {
  const [opacity, setOpacity] = useState<number>(0.75);
  const [colormap, setColormap] = useState<string>('Jet (Standard)');
  const [method, setMethod] = useState<'Grad-CAM' | 'Grad-CAM++' | 'Integrated Gradients'>('Grad-CAM');
  const [showBbox, setShowBbox] = useState<boolean>(true);
  const [viewMode, setViewMode] = useState<'overlay' | 'split' | 'heatmap_only'>('overlay');
  const [explainOpen, setExplainOpen] = useState<boolean>(false);

  const topDisease = result.diseases[0];

  // Selected explainability method (simulated rendering via CSS filter)
  const [explainMethodId, setExplainMethodId] = useState('gradcam');
  const explainMethod = EXPLAINABILITY_METHODS.find((m) => m.id === explainMethodId) ?? EXPLAINABILITY_METHODS[0];

  // Real server-computed activation map (Grad-CAM / Grad-CAM++ when a fine-
  // tuned head is loaded; class-agnostic feature activation for the backbone).
  const realMapUrl =
    result.heatmapOverlayUrl && result.heatmapOverlayUrl !== result.originalImageUrl ? result.heatmapOverlayUrl : null;
  const serverMethod = result.explainability?.method || null;
  const realMapActive = !!realMapUrl && !!serverMethod && serverMethod !== 'unavailable';
  const methodMatchesServer =
    (serverMethod === 'grad-cam' && method === 'Grad-CAM') ||
    (serverMethod === 'grad-cam++' && method === 'Grad-CAM++') ||
    (serverMethod === 'feature-activation' && (method === 'Grad-CAM' || method === 'Grad-CAM++'));
  const showingSimulated = !realMapActive || !methodMatchesServer;
  const serverMethodLabel =
    serverMethod === 'grad-cam++' ? 'Grad-CAM++' : serverMethod === 'grad-cam' ? 'Grad-CAM' : serverMethod === 'feature-activation' ? 'Feature Activation' : null;

  const handleDownloadHeatmap = () => {
    const link = document.createElement('a');
    link.download = `gradcam_${result.imageName}_${method}.png`;
    link.href = realMapUrl || result.heatmapOverlayUrl || result.originalImageUrl;
    link.click();
  };

  // Simulated feature contribution breakdown
  const featureContributions = useMemo(() => {
    const diseases = result.diseases.slice(0, 5);
    const total = diseases.reduce((a, d) => a + d.probability, 0) || 1;
    return diseases.map((d) => ({
      name: d.disease,
      weight: d.probability / total,
      contribution: Math.max(2, Math.round((d.probability / total) * 40)),
    }));
  }, [result.diseases]);

  // Similar case retrieval — genuine DenseNet-121 feature embeddings + cosine
  // similarity via the FastAPI engine. Falls back to clearly-labelled demo
  // similarities when the engine is offline (never presented as real).
  const [similarCases, setSimilarCases] = useState<
    { id: string; title: string; category?: string; image: string; similarity: number; source: 'real' | 'demo' }[]
  >([]);
  const [simState, setSimState] = useState<'loading' | 'real' | 'offline'>('loading');

  const rasterize = (dataUrl: string, size = 256): Promise<string> =>
    new Promise((resolve) => {
      const img = new Image();
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext('2d');
      img.onload = () => {
        try {
          ctx?.drawImage(img, 0, 0, size, size);
          resolve(canvas.toDataURL('image/png'));
        } catch {
          resolve('');
        }
      };
      img.onerror = () => resolve('');
      img.src = dataUrl;
    });

  useEffect(() => {
    let cancelled = false;
    const querySrc = result.originalImageUrl;
    if (!querySrc) {
      setSimState('offline');
      return;
    }
    setSimState('loading');
    (async () => {
      const query = querySrc.startsWith('data:image/svg') ? await rasterize(querySrc) : querySrc;
      const refs = await Promise.all(
        SAMPLE_XRAYS.map(async (s) => ({
          id: s.id,
          title: s.title,
          category: s.category,
          imageData: await rasterize(s.svgDataUrl),
        }))
      );
      const usable = refs.filter((r) => r.imageData);
      if (!query || usable.length === 0) {
        if (!cancelled) setSimState('offline');
        return;
      }
      try {
        const res = await api.getSimilarCases(query, usable);
        if (!cancelled && res.cases && res.cases.length > 0) {
          setSimilarCases(
            res.cases.map((c) => {
              const sample = SAMPLE_XRAYS.find((s) => s.id === c.case_id);
              return {
                id: c.case_id,
                title: c.title || sample?.title || c.label || 'Reference study',
                category: c.label || sample?.category,
                image: sample?.svgDataUrl || '',
                similarity: c.similarity,
                source: 'real' as const,
              };
            })
          );
          setSimState('real');
          return;
        }
        if (!cancelled) setSimState('offline');
      } catch {
        if (!cancelled) setSimState('offline');
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [result.originalImageUrl, result.id]);

  const decisionFlow = [
    { step: 'Preprocessing', detail: 'CLAHE + bilateral denoise · 1024×1024 normalized tensor', icon: '🖼️' },
    { step: 'Feature Extraction', detail: 'DenseNet-121 denseblock4 · 512 channels of high-level features', icon: '🧠' },
    { step: 'Activation Mapping', detail: `Grad-CAM pooled gradients → spatial heatmap for "${topDisease?.disease}"`, icon: '🔥' },
    { step: 'Classification Head', detail: `Sigmoid multi-label head · ${result.diseases.length} pathologies scored`, icon: '📊' },
    { step: 'Final Prediction', detail: `${result.topDiagnosis} @ ${(result.topConfidence * 100).toFixed(1)}% · severity ${result.severity}`, icon: '✅' },
  ];

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6 text-slate-800 dark:bg-[var(--surface)] dark:border-[var(--border)]">
      {/* Title & Method Selector Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 dark:border-[var(--border)] pb-4">
        <div>
          <h3 className="text-base font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
            <Eye className="w-5 h-5 text-indigo-600" />
            <span>Explainable AI: Activation Maps</span>
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Visual explanations for prediction of <strong className="text-indigo-700 dark:text-indigo-400">{topDisease?.disease}</strong>.
          </p>
        </div>

        {/* View mode buttons */}
        <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-800 p-1 rounded-md border border-slate-200 dark:border-slate-700">
          <button
            onClick={() => setViewMode('overlay')}
            className={`px-3 py-1 rounded text-xs font-semibold transition-all ${viewMode === 'overlay' ? 'bg-indigo-600 text-white shadow-2xs' : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white'}`}
          >
            Overlay
          </button>
          <button
            onClick={() => setViewMode('split')}
            className={`px-3 py-1 rounded text-xs font-semibold transition-all ${viewMode === 'split' ? 'bg-indigo-600 text-white shadow-2xs' : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white'}`}
          >
            Split View
          </button>
          <button
            onClick={() => setViewMode('heatmap_only')}
            className={`px-3 py-1 rounded text-xs font-semibold transition-all ${viewMode === 'heatmap_only' ? 'bg-indigo-600 text-white shadow-2xs' : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white'}`}
          >
            Heatmap
          </button>
        </div>
      </div>

      {/* Controls Bar */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 bg-slate-50 dark:bg-slate-800/50 p-3.5 rounded-lg border border-slate-200 dark:border-slate-700 text-xs">
        {/* Method */}
        <div>
          <label className="text-[11px] font-semibold text-slate-600 dark:text-slate-300 block mb-1">Method Algorithm:</label>
          <select
            value={method}
            onChange={(e: any) => {
              const v = e.target.value;
              setMethod(v);
              // Keep the legacy selector in sync with the explainability explorer
              if (v === 'Grad-CAM') setExplainMethodId('gradcam');
              else if (v === 'Grad-CAM++') setExplainMethodId('gradcampp');
              else if (v === 'Integrated Gradients') setExplainMethodId('integrated');
            }}
            className="w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-200 rounded-md px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-mono text-xs shadow-2xs"
          >
            <option value="Grad-CAM">Grad-CAM (Standard)</option>
            <option value="Grad-CAM++">Grad-CAM++ (Focal Detail)</option>
            <option value="Integrated Gradients">Integrated Gradients</option>
          </select>
        </div>

        {/* Colormap */}
        <div>
          <label className="text-[11px] font-semibold text-slate-600 dark:text-slate-300 block mb-1">Colormap Spectrum:</label>
          <select
            value={colormap}
            onChange={(e) => setColormap(e.target.value)}
            className="w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-200 rounded-md px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-mono text-xs shadow-2xs"
          >
            <option value="Jet (Standard)">Jet (Standard Thermal)</option>
            <option value="Turbo">Turbo (High Contrast)</option>
            <option value="Viridis">Viridis (Perceptual)</option>
            <option value="Inferno">Inferno (Radiation)</option>
          </select>
        </div>

        {/* Opacity Slider */}
        <div>
          <div className="flex justify-between text-[11px] font-semibold text-slate-600 dark:text-slate-300 mb-1">
            <span>Heat Opacity</span>
            <span className="font-mono text-indigo-600 dark:text-indigo-400">{Math.round(opacity * 100)}%</span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="w-full accent-indigo-600 cursor-pointer mt-1"
          />
        </div>

        {/* BBox Toggle & Download */}
        <div className="flex items-center justify-between gap-2 pt-2 md:pt-0">
          <label className="flex items-center gap-2 cursor-pointer text-slate-700 dark:text-slate-200 font-semibold text-[11px]">
            <input
              type="checkbox"
              checked={showBbox}
              onChange={(e) => setShowBbox(e.target.checked)}
              className="w-4 h-4 accent-indigo-600 rounded cursor-pointer"
            />
            <span>Focal Bounding Box</span>
          </label>

          <button
            onClick={handleDownloadHeatmap}
            className="px-3 py-1.5 bg-white dark:bg-slate-900 hover:bg-slate-100 text-slate-700 dark:text-slate-200 rounded-md border border-slate-300 dark:border-slate-700 flex items-center gap-1.5 text-xs font-semibold transition-colors cursor-pointer shadow-2xs"
          >
            <Download className="w-3.5 h-3.5 text-indigo-600" />
            <span>Export Map</span>
          </button>
        </div>
      </div>

      {/* Explainability method explorer */}
      <div className="bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 rounded-lg p-3.5 space-y-2.5">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-bold text-slate-600 dark:text-slate-300 flex items-center gap-1.5">
            <Wand2 className="w-3.5 h-3.5 text-violet-500" />
            Explainability Method Explorer
          </span>
          <span className="text-[9px] font-mono text-slate-400">
            {realMapActive ? 'Server: ' + (serverMethodLabel ?? serverMethod) : 'Simulated previews'}
          </span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {EXPLAINABILITY_METHODS.map((m) => (
            <button
              key={m.id}
              onClick={() => setExplainMethodId(m.id)}
              className={cn(
                'px-2.5 py-1 rounded-full text-[10px] font-semibold border transition-all cursor-pointer',
                explainMethodId === m.id
                  ? 'text-white shadow'
                  : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:border-slate-400'
              )}
              style={explainMethodId === m.id ? { background: m.accent, borderColor: m.accent } : undefined}
            >
              {m.name}
            </button>
          ))}
        </div>
        <p className="text-[10px] text-slate-500 dark:text-slate-400 leading-relaxed">
          <strong>{explainMethod.name}:</strong> {explainMethod.description}{' '}
          {realMapActive && methodMatchesServer ? (
            <span className="text-emerald-600 dark:text-emerald-400 font-semibold">
              — real {serverMethodLabel} map computed server-side (engine {result.engine?.device ?? 'CPU'}).
            </span>
          ) : realMapActive ? (
            <span className="text-amber-600 dark:text-amber-400 font-semibold">
              — SIMULATED PREVIEW: the backend computed {serverMethodLabel}, not {explainMethod.name}. CSS filter shown for illustration only.
            </span>
          ) : (
            <span className="text-amber-600 dark:text-amber-400 font-semibold">
              — SIMULATED PREVIEW: no real activation map is available for this result (demo engine or engine offline).
            </span>
          )}
        </p>
      </div>

      {/* Main Visual Stage */}
      <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 flex items-center justify-center min-h-[380px]">
        {viewMode === 'split' ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full max-w-4xl">
            <div className="space-y-2 text-center">
              <span className="text-xs font-semibold text-slate-300 font-mono">Original Radiograph</span>
              <div className="relative aspect-square rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
                <img src={result.originalImageUrl} alt="Original" className="w-full h-full object-contain" />
              </div>
            </div>
            <div className="space-y-2 text-center">
              <span className="text-xs font-semibold text-indigo-300 font-mono">
                {realMapActive && methodMatchesServer ? `${serverMethodLabel} Map (real)` : `${explainMethod.name} Map`}
              </span>
              <div className="relative aspect-square rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
                <img
                  src={realMapUrl || result.heatmapOverlayUrl || result.originalImageUrl}
                  alt="Activation map"
                  className="w-full h-full object-contain"
                  style={{ filter: showingSimulated ? explainMethod.filter : undefined }}
                />
                {showingSimulated && (
                  <span className="absolute top-2 right-2 bg-amber-500/90 text-slate-950 text-[9px] font-black px-1.5 py-0.5 rounded">
                    SIMULATED
                  </span>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="relative w-full max-w-lg aspect-square bg-slate-950 rounded-lg overflow-hidden border border-slate-800 shadow-inner flex items-center justify-center">
            {/* Original Image */}
            <img src={result.originalImageUrl} alt="Original" className="absolute inset-0 w-full h-full object-contain" />

            {/* Heatmap Overlay with Opacity — the real engine map when available */}
            <img
              src={realMapUrl || result.heatmapOverlayUrl || result.originalImageUrl}
              alt="Heatmap"
              style={{ opacity: viewMode === 'heatmap_only' ? 1 : opacity, filter: showingSimulated ? explainMethod.filter : undefined }}
              className="absolute inset-0 w-full h-full object-contain mix-blend-screen transition-opacity"
            />

            {showingSimulated && (
              <span className="absolute top-3 right-3 bg-amber-500/90 text-slate-950 text-[9px] font-black px-1.5 py-0.5 rounded z-10">
                SIMULATED PREVIEW
              </span>
            )}

            {/* Study info overlay */}
            <div className="absolute top-3 left-3 bg-slate-950/80 backdrop-blur px-2.5 py-1 rounded-md border border-slate-800 text-[10px] font-mono text-slate-300 space-y-0.5">
              <div>METHOD: {realMapActive && methodMatchesServer ? (serverMethodLabel ?? 'SERVER').toUpperCase() : explainMethod.name.toUpperCase()}</div>
              <div>LAYER: denseblock4.conv2</div>
              {realMapActive && methodMatchesServer ? (
                <div className="text-emerald-400">SOURCE: ENGINE · {result.engine?.device ?? 'CPU'}</div>
              ) : (
                <div>SPECTRUM: {colormap}</div>
              )}
            </div>

            <div className="absolute bottom-3 right-3 bg-slate-950/80 backdrop-blur px-2.5 py-1 rounded-md border border-slate-800 text-[10px] font-mono text-indigo-300">
              ACTIVATION PEAK: {Math.round(topDisease?.probability * 100)}%
            </div>
          </div>
        )}
      </div>

      {/* Region Interpretation Callout */}
      {result.gradCamRegions && result.gradCamRegions.length > 0 && (
        <div className="bg-indigo-50 dark:bg-indigo-500/10 border border-indigo-200 dark:border-indigo-500/25 p-3.5 rounded-lg text-xs space-y-1">
          <div className="flex items-center gap-1.5 font-bold text-indigo-800 dark:text-indigo-300">
            <Target className="w-4 h-4 text-indigo-600" />
            <span>Focal Activation Region Analysis</span>
          </div>
          {result.gradCamRegions.map((region, i) => (
            <p key={i} className="text-slate-700 dark:text-slate-300 text-xs leading-relaxed">
              <strong>{region.name}:</strong> {region.interpretation} (Confidence: {region.intensity}%)
            </p>
          ))}
        </div>
      )}

      {/* Explain Prediction / Decision Flow */}
      <div className="space-y-2.5">
        <button
          onClick={() => setExplainOpen((o) => !o)}
          className="w-full flex items-center justify-between px-4 py-3 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 hover:border-indigo-400 transition-all cursor-pointer group"
        >
          <span className="text-xs font-bold text-slate-700 dark:text-slate-200 flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
            Explain This Prediction (AI Decision Flow)
          </span>
          <ChevronDown className={cn('w-4 h-4 text-slate-400 transition-transform group-hover:text-indigo-500', explainOpen && 'rotate-180')} />
        </button>

        {explainOpen && (
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden animate-fade-in-up">
            {/* Steps */}
            <div className="bg-slate-50 dark:bg-slate-800/60 p-4">
              <div className="relative">
                <div className="absolute left-4 top-2 bottom-2 w-px bg-slate-200 dark:bg-slate-700" />
                <div className="space-y-3">
                  {decisionFlow.map((f, i) => (
                    <div key={i} className="relative flex gap-3 pl-0">
                      <span className="relative z-10 w-8 h-8 rounded-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 flex items-center justify-center text-sm shrink-0 shadow-sm">
                        {f.icon}
                      </span>
                      <div className="pt-1">
                        <p className="text-[11px] font-bold text-slate-800 dark:text-slate-100">
                          {i + 1}. {f.step}
                        </p>
                        <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-0.5">{f.detail}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Feature contributions */}
            <div className="p-4 border-t border-slate-100 dark:border-slate-800">
              <p className="text-[11px] font-bold text-slate-700 dark:text-slate-200 mb-2.5 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-amber-500" />
                Top Feature Contributions
              </p>
              <div className="space-y-2">
                {featureContributions.map((fc) => (
                  <div key={fc.name} className="flex items-center gap-2">
                    <span className="text-[10px] text-slate-600 dark:text-slate-300 w-28 truncate">{fc.name}</span>
                    <div className="flex-1 h-1.5 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-all duration-700"
                        style={{ width: `${Math.min(100, fc.contribution * 2.5)}%` }}
                      />
                    </div>
                    <span className="text-[9px] font-mono text-slate-400 w-10 text-right">{(fc.weight * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Similar Case Retrieval */}
      <div className="space-y-2.5">
        <p className="text-[11px] font-bold text-slate-600 dark:text-slate-300 flex items-center gap-1.5">
          <ScanSearch className="w-3.5 h-3.5 text-sky-500" />
          Similar Case Retrieval
          <span className="text-[9px] font-mono text-slate-400 font-normal">
            · {simState === 'real' ? 'DenseNet-121 embeddings + cosine similarity' : 'over curated demo cohort'}
          </span>
        </p>

        {simState === 'loading' && (
          <div className="flex items-center gap-2 text-[10px] text-slate-400 py-3">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            Embedding query image and comparing against the reference cohort…
          </div>
        )}

        {simState === 'real' && (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              {similarCases.slice(0, 3).map((c) => (
                <div key={c.id} className="rounded-lg border border-slate-200 dark:border-slate-700 overflow-hidden group hover:border-sky-400/60 hover:shadow-md transition-all">
                  <div className="relative h-20 bg-slate-900 overflow-hidden">
                    <img src={c.image} alt={c.title} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" />
                    <span className="absolute top-1.5 right-1.5 text-[9px] font-mono font-bold bg-slate-950/80 border border-slate-700 text-sky-300 px-1.5 py-0.5 rounded">
                      {(c.similarity * 100).toFixed(0)}% sim
                    </span>
                  </div>
                  <div className="p-2.5">
                    <p className="text-[10px] font-bold text-slate-800 dark:text-slate-100 truncate">{c.title}</p>
                    <p className="text-[9px] text-slate-400 mt-0.5">{c.category}</p>
                  </div>
                </div>
              ))}
            </div>
            <p className="text-[9px] text-slate-400 leading-relaxed">
              Similarity = cosine similarity of genuine DenseNet-121 feature embeddings over the curated demo cohort.{" "}
              <strong>Similarity is not diagnostic evidence.</strong>
            </p>
          </>
        )}

        {simState === 'offline' && (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              {SAMPLE_XRAYS.slice(0, 3)
                .map((s) => ({
                  id: s.id,
                  title: s.title,
                  category: s.category,
                  image: s.svgDataUrl,
                  similarity: 58 + Math.floor(Math.random() * 30),
                }))
                .sort((a, b) => b.similarity - a.similarity)
                .map((c) => (
                  <div key={c.id} className="rounded-lg border border-slate-200 dark:border-slate-700 overflow-hidden opacity-80">
                    <div className="relative h-20 bg-slate-900 overflow-hidden">
                      <img src={c.image} alt={c.title} className="w-full h-full object-cover" />
                      <span className="absolute top-1.5 right-1.5 text-[9px] font-mono font-bold bg-slate-950/80 border border-amber-500/50 text-amber-300 px-1.5 py-0.5 rounded">
                        DEMO · {c.similarity}%
                      </span>
                    </div>
                    <div className="p-2.5">
                      <p className="text-[10px] font-bold text-slate-800 dark:text-slate-100 truncate">{c.title}</p>
                      <p className="text-[9px] text-slate-400 mt-0.5">{c.category}</p>
                    </div>
                  </div>
                ))}
            </div>
            <p className="text-[9px] text-amber-600 dark:text-amber-400 leading-relaxed">
              <FlaskConical className="inline w-3 h-3 mr-1" />
              Demo similarity — the ML engine is offline, so these are illustrative values, not model-feature similarities.
            </p>
          </>
        )}
      </div>
    </div>
  );
};
