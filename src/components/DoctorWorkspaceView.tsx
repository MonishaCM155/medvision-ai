import React, { useState, useRef } from 'react';
import {
  ZoomIn, ZoomOut, RotateCcw, Eye, Sun, Contrast, Sliders,
  SlidersHorizontal, FlipHorizontal, Scissors, Download, FileText,
  Bookmark, MessageSquare, Check, Layers, ChevronRight, Activity,
  Maximize2, User, Clock, AlertTriangle, ShieldCheck, Move, Ruler, PenLine, RotateCw, Trash2,
} from 'lucide-react';
import { PredictionResult } from '../types';
import { SAMPLE_XRAYS } from '../data/sampleXrays';
import { cn } from '../utils/cn';

interface DoctorWorkspaceProps {
  currentResult: PredictionResult | null;
  onSelectResult?: (res: PredictionResult) => void;
}

interface DrawLine {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  kind: 'measure' | 'annotate';
}

// Simulated U-Net lung segmentation mask (SVG overlay)
const LungSegmentationMask: React.FC<{ opacity: number }> = ({ opacity }) => (
  <svg
    viewBox="0 0 100 100"
    preserveAspectRatio="none"
    className="absolute inset-0 w-full h-full pointer-events-none"
    style={{ opacity, mixBlendMode: 'screen' }}
    aria-hidden
  >
    <path d="M49 22 C40 15 27 21 25 35 C23 51 29 67 39 75 C47 80 52 72 50 62 C48 50 48 30 49 22 Z" fill="#34d399" fillOpacity="0.55" />
    <path d="M51 22 C60 15 73 21 75 35 C77 51 71 67 61 75 C53 80 48 72 50 62 C52 50 52 30 51 22 Z" fill="#34d399" fillOpacity="0.55" />
    <path d="M47.5 10 L52.5 10 M48 12 L52 12" stroke="#34d399" strokeWidth="1.4" fill="none" />
    <circle cx="40" cy="47" r="6" fill="#f472b6" fillOpacity="0.35" />
    <circle cx="62" cy="52" r="8" fill="#f472b6" fillOpacity="0.35" />
  </svg>
);

export const DoctorWorkspaceView: React.FC<DoctorWorkspaceProps> = ({ currentResult, onSelectResult }) => {
  // DICOM Viewer Manipulation State
  const [zoom, setZoom] = useState(1.0);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [isInverted, setIsInverted] = useState(false);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.65);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showSegmentation, setShowSegmentation] = useState(false);
  const [isSplitView, setIsSplitView] = useState(false);
  const [rotation, setRotation] = useState(0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [activeTool, setActiveTool] = useState<'pan' | 'measure' | 'annotate'>('pan');
  const [lines, setLines] = useState<DrawLine[]>([]);
  const [draft, setDraft] = useState<DrawLine | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  // Clinical Notes & Bookmarks
  const [doctorNotes, setDoctorNotes] = useState(
    'Patient presents with 4-day history of productive cough, fever (38.8°C), and right lower chest pleuritic pain. X-ray shows right lower lobe alveolar consolidation consistent with acute lobar pneumonia.'
  );
  const [isSaved, setIsSaved] = useState(false);

  // Default image if currentResult is null
  const activeImage = currentResult?.originalImageUrl || SAMPLE_XRAYS[0].svgDataUrl;
  const heatmapImage = currentResult?.heatmapOverlayUrl || SAMPLE_XRAYS[0].heatmapSvgUrl;

  const handleReset = () => {
    setZoom(1.0);
    setBrightness(100);
    setContrast(100);
    setIsInverted(false);
    setHeatmapOpacity(0.65);
    setRotation(0);
    setPan({ x: 0, y: 0 });
    setLines([]);
  };

  const handleSaveCase = () => {
    setIsSaved(true);
    setTimeout(() => setIsSaved(false), 2500);
  };

  // Pointer helpers: convert event to SVG coordinates
  const getSvgPoint = (e: React.PointerEvent) => {
    const svg = svgRef.current;
    if (!svg) return { x: 0, y: 0 };
    const rect = svg.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const onPointerDown = (e: React.PointerEvent) => {
    (e.target as Element).setPointerCapture?.(e.pointerId);
    if (activeTool === 'pan') {
      setIsPanning(true);
      setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    } else {
      const pt = getSvgPoint(e);
      setDraft({ id: `line_${Date.now()}`, x1: pt.x, y1: pt.y, x2: pt.x, y2: pt.y, kind: activeTool });
    }
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (activeTool === 'pan' && isPanning) {
      setPan({ x: e.clientX - panStart.x, y: e.clientY - panStart.y });
    } else if (draft && activeTool !== 'pan') {
      const pt = getSvgPoint(e);
      setDraft((d) => (d ? { ...d, x2: pt.x, y2: pt.y } : d));
    }
  };

  const onPointerUp = () => {
    if (activeTool === 'pan') {
      setIsPanning(false);
      return;
    }
    if (draft) {
      const len = Math.hypot(draft.x2 - draft.x1, draft.y2 - draft.y1);
      if (len > 6) setLines((prev) => [...prev, draft]);
      setDraft(null);
    }
  };

  const measureLength = (l: { x1: number; y1: number; x2: number; y2: number }) =>
    Math.round(Math.hypot(l.x2 - l.x1, l.y2 - l.y1) / zoom);

  return (
    <div className="space-y-6 text-slate-100">
      {/* Top Title Bar */}
      <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4 backdrop-blur">
        <div>
          <div className="flex items-center gap-2">
            <User className="w-5 h-5 text-indigo-400" />
            <h1 className="text-lg font-bold text-white tracking-tight">Radiologist Clinical Workspace &amp; DICOM Studio</h1>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Advanced workstation with Window/Level controls, pan, rotate, measurements, annotations &amp; Grad-CAM fusion
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsSplitView(!isSplitView)}
            className={`px-3 py-1.5 rounded-lg border text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer ${
              isSplitView ? 'bg-indigo-600 text-white border-indigo-500' : 'bg-slate-800 text-slate-200 border-slate-700 hover:bg-slate-700'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>{isSplitView ? 'Single View' : 'Split Comparison'}</span>
          </button>

          <button
            onClick={handleSaveCase}
            className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded-lg flex items-center gap-1.5 transition-colors cursor-pointer shadow-xs"
          >
            {isSaved ? <Check className="w-3.5 h-3.5" /> : <Bookmark className="w-3.5 h-3.5" />}
            <span>{isSaved ? 'Case Saved' : 'Save to EHR'}</span>
          </button>
        </div>
      </div>

      {/* Main Studio Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left Interactive DICOM Viewer Canvas (3 Cols) */}
        <div className="lg:col-span-3 bg-slate-900/90 rounded-xl border border-slate-800 p-4 flex flex-col space-y-4 backdrop-blur">
          {/* DICOM Control Toolbar */}
          <div className="bg-slate-950 p-2 rounded-lg border border-slate-800 flex flex-wrap items-center justify-between gap-2 text-xs">
            {/* Tool buttons */}
            <div className="flex items-center gap-1">
              <button
                onClick={() => setZoom((z) => Math.min(z + 0.25, 3.0))}
                className="p-1.5 bg-slate-900 hover:bg-slate-800 text-slate-300 rounded border border-slate-800 cursor-pointer"
                title="Zoom In"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                onClick={() => setZoom((z) => Math.max(z - 0.25, 0.5))}
                className="p-1.5 bg-slate-900 hover:bg-slate-800 text-slate-300 rounded border border-slate-800 cursor-pointer"
                title="Zoom Out"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <button
                onClick={() => setRotation((r) => (r + 90) % 360)}
                className="p-1.5 bg-slate-900 hover:bg-slate-800 text-slate-300 rounded border border-slate-800 cursor-pointer"
                title="Rotate 90°"
              >
                <RotateCw className="w-4 h-4" />
              </button>
              <button
                onClick={handleReset}
                className="p-1.5 bg-slate-900 hover:bg-slate-800 text-slate-300 rounded border border-slate-800 cursor-pointer"
                title="Reset View"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
              <span className="w-px h-5 bg-slate-800 mx-1" />

              {/* Interaction tools */}
              {([
                { id: 'pan', icon: <Move className="w-3.5 h-3.5" />, label: 'Pan' },
                { id: 'measure', icon: <Ruler className="w-3.5 h-3.5" />, label: 'Measure' },
                { id: 'annotate', icon: <PenLine className="w-3.5 h-3.5" />, label: 'Annotate' },
              ] as const).map((t) => (
                <button
                  key={t.id}
                  onClick={() => setActiveTool(t.id)}
                  className={cn(
                    'px-2 py-1.5 rounded border text-[10px] font-semibold flex items-center gap-1 transition-colors cursor-pointer',
                    activeTool === t.id
                      ? 'bg-indigo-950 text-indigo-300 border-indigo-700'
                      : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
                  )}
                  title={t.label}
                >
                  {t.icon}
                  <span className="hidden md:inline">{t.label}</span>
                </button>
              ))}

              {lines.length > 0 && (
                <button
                  onClick={() => setLines([])}
                  className="p-1.5 bg-rose-950/60 text-rose-300 rounded border border-rose-900 cursor-pointer hover:bg-rose-900/60"
                  title="Clear annotations"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}

              {/* Window/Level presets */}
              <span className="w-px h-5 bg-slate-800 mx-1" />
              {[
                { id: 'lung', label: 'Lung', b: 100, c: 135 },
                { id: 'mediastinum', label: 'Mediastinum', b: 95, c: 170 },
                { id: 'bone', label: 'Bone', b: 108, c: 88 },
              ].map((p) => (
                <button
                  key={p.id}
                  onClick={() => {
                    setBrightness(p.b);
                    setContrast(p.c);
                  }}
                  className={cn(
                    'px-2 py-1 rounded border text-[10px] font-semibold transition-colors cursor-pointer',
                    brightness === p.b && contrast === p.c
                      ? 'bg-cyan-950 text-cyan-300 border-cyan-800'
                      : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
                  )}
                  title={`Window/Level: ${p.label} preset`}
                >
                  {p.label}
                </button>
              ))}

              <span className="w-px h-5 bg-slate-800 mx-1" />

              {/* Invert Colors */}
              <button
                onClick={() => setIsInverted(!isInverted)}
                className={`px-2.5 py-1 rounded border text-[11px] font-medium transition-colors cursor-pointer ${
                  isInverted ? 'bg-amber-950 text-amber-300 border-amber-800' : 'bg-slate-900 text-slate-300 border-slate-800'
                }`}
              >
                Invert
              </button>

              {/* Heatmap Toggle */}
              <button
                onClick={() => setShowHeatmap(!showHeatmap)}
                className={`px-2.5 py-1 rounded border text-[11px] font-medium transition-colors cursor-pointer ${
                  showHeatmap ? 'bg-indigo-950 text-indigo-300 border-indigo-800' : 'bg-slate-900 text-slate-300 border-slate-800'
                }`}
              >
                Grad-CAM
              </button>

              {/* Segmentation Toggle */}
              <button
                onClick={() => setShowSegmentation(!showSegmentation)}
                className={cn(
                  'px-2.5 py-1 rounded border text-[11px] font-medium transition-colors cursor-pointer',
                  showSegmentation ? 'bg-emerald-950 text-emerald-300 border-emerald-800' : 'bg-slate-900 text-slate-300 border-slate-800'
                )}
                title="U-Net lung segmentation overlay"
              >
                <Scissors className="w-3.5 h-3.5 inline -mt-0.5 mr-1" />
                Segment
              </button>
            </div>

            {/* Adjustment Sliders */}
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
                <Sun className="w-3.5 h-3.5" />
                <input type="range" min="50" max="150" value={brightness} onChange={(e) => setBrightness(Number(e.target.value))} className="w-16 accent-indigo-500 cursor-pointer" />
              </div>

              <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
                <Contrast className="w-3.5 h-3.5" />
                <input type="range" min="50" max="150" value={contrast} onChange={(e) => setContrast(Number(e.target.value))} className="w-16 accent-indigo-500 cursor-pointer" />
              </div>

              <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
                <Sliders className="w-3.5 h-3.5 text-indigo-400" />
                <span className="hidden sm:inline">Opacity</span>
                <input type="range" min="0" max="1" step="0.05" value={heatmapOpacity} onChange={(e) => setHeatmapOpacity(Number(e.target.value))} className="w-16 accent-indigo-500 cursor-pointer" />
              </div>

              {/* Rotation indicator */}
              <span className="text-[10px] font-mono text-slate-500 border border-slate-800 rounded px-1.5 py-0.5">
                {rotation}°
              </span>
            </div>
          </div>

          {/* Canvas Container */}
          <div
            className={cn('relative bg-black rounded-xl overflow-hidden min-h-[420px] max-h-[520px] flex items-center justify-center border border-slate-800', activeTool === 'pan' ? 'cursor-grab active:cursor-grabbing' : 'cursor-crosshair')}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerLeave={() => {
              if (activeTool !== 'pan') onPointerUp();
            }}
          >
            {/* SVG drawing + interaction overlay */}
            <svg
              ref={svgRef}
              className="absolute inset-0 w-full h-full z-20 touch-none"
              style={{ cursor: activeTool === 'pan' ? 'grab' : 'crosshair' }}
            >
              {[...lines, ...(draft ? [draft] : [])].map((l) => {
                const isMeasure = l.kind === 'measure';
                return (
                  <g key={l.id}>
                    <line
                      x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2}
                      stroke={isMeasure ? '#f59e0b' : '#38bdf8'}
                      strokeWidth={isMeasure ? 2 : 2.5}
                      strokeDasharray={isMeasure ? '6 3' : undefined}
                      strokeLinecap="round"
                    />
                    {isMeasure && (
                      <g transform={`translate(${(l.x1 + l.x2) / 2 + 6}, ${(l.y1 + l.y2) / 2 - 8})`}>
                        <rect width={52} height={16} rx={4} fill="#0f172a" stroke="#f59e0b" strokeWidth={0.75} />
                        <text x={26} y={11} textAnchor="middle" fill="#fbbf24" fontSize={9} fontFamily="monospace" fontWeight="bold">
                          {measureLength(l)}px
                        </text>
                      </g>
                    )}
                  </g>
                );
              })}
            </svg>

            {isSplitView ? (
              <div className="grid grid-cols-2 w-full h-full divide-x divide-slate-800">
                {/* Left Raw DICOM */}
                <div className="relative overflow-hidden flex items-center justify-center p-2">
                  <span className="absolute top-2 left-2 bg-slate-950/80 px-2 py-0.5 rounded text-[10px] text-slate-300 font-mono z-10 border border-slate-800">
                    Raw X-Ray
                  </span>
                  <img
                    src={activeImage}
                    alt="Raw DICOM"
                    className="max-h-full max-w-full object-contain transition-transform duration-200"
                    style={{
                      transform: `scale(${zoom}) rotate(${rotation}deg) translate(${pan.x}px, ${pan.y}px)`,
                      filter: `brightness(${brightness}%) contrast(${contrast}%) ${isInverted ? 'invert(100%)' : ''}`,
                    }}
                  />
                </div>

                {/* Right Grad-CAM Overlay */}
                <div className="relative overflow-hidden flex items-center justify-center p-2">
                  <span className="absolute top-2 left-2 bg-indigo-950/80 px-2 py-0.5 rounded text-[10px] text-indigo-300 font-mono z-10 border border-indigo-800">
                    Grad-CAM Activation
                  </span>
                  <div
                    className={cn('relative max-h-full max-w-full flex items-center justify-center', !isPanning && 'transition-transform duration-200')}
                    style={{ transform: `scale(${zoom}) rotate(${rotation}deg) translate(${pan.x}px, ${pan.y}px)` }}
                  >
                    <img
                      src={activeImage}
                      alt="Base X-Ray"
                      className="max-h-full max-w-full object-contain"
                      style={{
                        filter: `brightness(${brightness}%) contrast(${contrast}%) ${isInverted ? 'invert(100%)' : ''}`,
                      }}
                    />
                    {showHeatmap && (
                      <img
                        src={heatmapImage}
                        alt="Heatmap"
                        className="absolute inset-0 w-full h-full object-contain mix-blend-screen pointer-events-none"
                        style={{ opacity: heatmapOpacity }}
                      />
                    )}
                    {showSegmentation && <LungSegmentationMask opacity={0.65} />}
                  </div>
                </div>
              </div>
            ) : (
              /* Single View Stage */
              <div className="relative w-full h-full flex items-center justify-center p-4">
                <div
                  className={cn('relative max-h-full max-w-full flex items-center justify-center', !isPanning && 'transition-transform duration-200')}
                  style={{ transform: `scale(${zoom}) rotate(${rotation}deg) translate(${pan.x}px, ${pan.y}px)` }}
                >
                  <img
                    src={activeImage}
                    alt="Radiogram"
                    className="max-h-[460px] object-contain rounded select-none"
                    draggable={false}
                    style={{
                      filter: `brightness(${brightness}%) contrast(${contrast}%) ${isInverted ? 'invert(100%)' : ''}`,
                    }}
                  />
                  {showHeatmap && (
                    <img
                      src={heatmapImage}
                      alt="Grad-CAM"
                      className="absolute inset-0 w-full h-full object-contain mix-blend-screen pointer-events-none"
                      style={{ opacity: heatmapOpacity }}
                    />
                  )}
                  {showSegmentation && <LungSegmentationMask opacity={0.65} />}
                </div>

                {/* Tool hint */}
                <div className="absolute bottom-3 left-3 bg-slate-950/80 backdrop-blur px-2.5 py-1 rounded-md border border-slate-800 text-[10px] font-mono text-slate-400 z-10">
                  {activeTool === 'pan' && 'DRAG to pan · wheel-free zoom buttons'}
                  {activeTool === 'measure' && 'DRAG to draw a measurement'}
                  {activeTool === 'annotate' && 'DRAG to annotate'}
                </div>
              </div>
            )}
          </div>

          {/* Measurements readout */}
          {lines.filter((l) => l.kind === 'measure').length > 0 && (
            <div className="flex flex-wrap gap-2">
              {lines.filter((l) => l.kind === 'measure').map((l, i) => (
                <span key={l.id} className="inline-flex items-center gap-1.5 text-[10px] font-mono text-amber-300 bg-amber-950/60 border border-amber-800/70 rounded-full px-2.5 py-1">
                  <Ruler className="w-3 h-3" />
                  Measurement {i + 1}: {measureLength(l)} px
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Right Sidebar: Patient Case EHR & Notes (1 Col) */}
        <div className="space-y-4">
          {/* Patient Card */}
          <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800 space-y-3 backdrop-blur">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <span className="text-xs font-bold text-white">Patient Record</span>
              <span className="text-[10px] font-mono text-indigo-400 bg-indigo-950 px-1.5 py-0.2 rounded border border-indigo-900">
                EHR #89204
              </span>
            </div>

            <div className="space-y-1.5 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-400">Name:</span>
                <span className="font-semibold text-white">John Doe (58M)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Scan Type:</span>
                <span className="font-mono text-slate-200">PA Chest X-Ray</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Top Prediction:</span>
                <span className="font-bold text-amber-400">{currentResult?.topDiagnosis || 'Pneumonia'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Confidence:</span>
                <span className="font-mono text-indigo-300">{((currentResult?.topConfidence || 0.943) * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>

          {/* Radiologist Clinical Notes Box */}
          <div className="bg-slate-900/90 p-4 rounded-xl border border-slate-800 space-y-2 backdrop-blur">
            <h3 className="text-xs font-bold text-slate-300 flex items-center gap-1.5">
              <MessageSquare className="w-3.5 h-3.5 text-indigo-400" />
              Physician Diagnostic Impression
            </h3>
            <textarea
              value={doctorNotes}
              onChange={(e) => setDoctorNotes(e.target.value)}
              rows={6}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 focus:outline-hidden focus:border-indigo-500 leading-relaxed font-sans"
              placeholder="Enter attending radiologist comments..."
            />
          </div>
        </div>
      </div>
    </div>
  );
};
