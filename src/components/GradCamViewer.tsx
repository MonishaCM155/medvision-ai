import React, { useState, useMemo } from 'react';
import { Eye, Layers, Download, Sliders, Sparkles, Target, ZoomIn, Info, ChevronDown, GitBranch, ScanSearch, Wand2 } from 'lucide-react';
import { PredictionResult } from '../types';
import { SAMPLE_XRAYS } from '../data/sampleXrays';
import { EXPLAINABILITY_METHODS } from '../data/mockEnterprise';
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

  const handleDownloadHeatmap = () => {
    const link = document.createElement('a');
    link.download = `gradcam_${result.imageName}_${method}.png`;
    link.href = result.heatmapOverlayUrl || result.originalImageUrl;
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

  // Similar case retrieval (simulated embeddings search)
  const similarCases = useMemo(() => {
    return SAMPLE_XRAYS.map((s) => ({
      id: s.id,
      title: s.title,
      category: s.category,
      image: s.svgDataUrl,
      similarity:
        s.category === (result.topDiagnosis.includes('No Finding') ? 'No Finding' : result.topDiagnosis.split(' ')[0])
          ? 94 + Math.floor(Math.random() * 5)
          : 58 + Math.floor(Math.random() * 30),
    }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 3);
  }, [result.topDiagnosis]);

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
          <span className="text-[9px] font-mono text-slate-400">Captum · 6 methods</span>
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
          <strong>{explainMethod.name}:</strong> {explainMethod.description} Visualization rendered for the {result.diseases[0]?.disease ?? 'primary'} class.
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
              <span className="text-xs font-semibold text-indigo-300 font-mono">{explainMethod.name} Map</span>
              <div className="relative aspect-square rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
                <img
                  src={result.heatmapOverlayUrl}
                  alt="Grad-CAM"
                  className="w-full h-full object-contain"
                  style={{ filter: explainMethod.filter }}
                />
              </div>
            </div>
          </div>
        ) : (
          <div className="relative w-full max-w-lg aspect-square bg-slate-950 rounded-lg overflow-hidden border border-slate-800 shadow-inner flex items-center justify-center">
            {/* Original Image */}
            <img src={result.originalImageUrl} alt="Original" className="absolute inset-0 w-full h-full object-contain" />

            {/* Heatmap Overlay with Opacity */}
            <img
              src={result.heatmapOverlayUrl}
              alt="Heatmap"
              style={{ opacity: viewMode === 'heatmap_only' ? 1 : opacity, filter: explainMethod.filter }}
              className="absolute inset-0 w-full h-full object-contain mix-blend-screen transition-opacity"
            />

            {/* DICOM Info overlay */}
            <div className="absolute top-3 left-3 bg-slate-950/80 backdrop-blur px-2.5 py-1 rounded-md border border-slate-800 text-[10px] font-mono text-slate-300 space-y-0.5">
              <div>METHOD: {explainMethod.name.toUpperCase()}</div>
              <div>LAYER: denseblock4.conv2</div>
              <div>SPECTRUM: {colormap}</div>
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
          <span className="text-[9px] font-mono text-slate-400 font-normal">· Embedding search over curated cohort</span>
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
          {similarCases.map((c) => (
            <div key={c.id} className="rounded-lg border border-slate-200 dark:border-slate-700 overflow-hidden group hover:border-sky-400/60 hover:shadow-md transition-all">
              <div className="relative h-20 bg-slate-900 overflow-hidden">
                <img src={c.image} alt={c.title} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" />
                <span className="absolute top-1.5 right-1.5 text-[9px] font-mono font-bold bg-slate-950/80 border border-slate-700 text-sky-300 px-1.5 py-0.5 rounded">
                  {c.similarity}% sim
                </span>
              </div>
              <div className="p-2.5">
                <p className="text-[10px] font-bold text-slate-800 dark:text-slate-100 truncate">{c.title}</p>
                <p className="text-[9px] text-slate-400 mt-0.5">{c.category}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
