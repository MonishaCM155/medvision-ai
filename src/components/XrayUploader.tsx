import React, { useState, useRef, useEffect } from 'react';
import {
  UploadCloud, Sliders, Play, RefreshCw, CheckCircle2, AlertCircle,
  ShieldCheck, ShieldX, Loader2, ChevronDown, Lock, ScanLine, BadgeCheck,
} from 'lucide-react';
import { SAMPLE_XRAYS, SampleXray } from '../data/sampleXrays';
import { analyzeImage, ValidationReport } from '../utils/imageValidation';
import { cn } from '../utils/cn';

interface XrayUploaderProps {
  onRunInference: (data: {
    imageName: string;
    imageData: string;
    model: string;
    clahe: boolean;
    noiseRemoval: boolean;
    validation: ValidationReport;
  }) => void;
  isLoading: boolean;
}

export const XrayUploader: React.FC<XrayUploaderProps> = ({ onRunInference, isLoading }) => {
  const [selectedSample, setSelectedSample] = useState<SampleXray | null>(SAMPLE_XRAYS[0]);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [imageName, setImageName] = useState<string>('chest_xray_pneumonia_rll.dcm');
  const [selectedModel, setSelectedModel] = useState<string>('DenseNet-121 (CheXNet)');

  // Preprocessing states
  const [applyClahe, setApplyClahe] = useState<boolean>(true);
  const [applyNoiseRemoval, setApplyNoiseRemoval] = useState<boolean>(true);
  const [claheClipLimit, setClaheClipLimit] = useState<number>(2.5);

  const [dragActive, setDragActive] = useState<boolean>(false);
  const [fileError, setFileError] = useState<string | null>(null);

  // Validation state
  const [validationReport, setValidationReport] = useState<ValidationReport | null>(null);
  const [validationBusy, setValidationBusy] = useState<boolean>(false);
  const [showDetails, setShowDetails] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const analysisToken = useRef(0);

  // Active image source
  const currentImageSrc = uploadedImage || selectedSample?.svgDataUrl || SAMPLE_XRAYS[0].svgDataUrl;

  // Run chest X-ray validation + quality assessment whenever the image changes
  useEffect(() => {
    const token = ++analysisToken.current;
    setValidationBusy(true);
    setValidationReport(null);
    setFileError(null);
    analyzeImage(currentImageSrc)
      .then((report) => {
        if (token !== analysisToken.current) return;
        setValidationReport(report);
        setValidationBusy(false);
      })
      .catch(() => {
        if (token !== analysisToken.current) return;
        setValidationReport({
          passed: false,
          score: 0,
          message: 'Image validation failed to run in this browser.',
          checks: [{ key: 'format', label: 'Validation engine', status: 'fail', detail: 'The validation engine could not analyze this image in the current browser.' }],
          metrics: { width: 0, height: 0, aspectRatio: 0, orientation: 'square', meanIntensity: 0, stdIntensity: 0, contrastRatio: 0, colorDeviation: 255, sharpness: 0, structureScore: 0 },
        });
        setValidationBusy(false);
      });
  }, [currentImageSrc]);

  // Process canvas preview with CLAHE / Contrast adjustment
  useEffect(() => {
    if (!currentImageSrc) return;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      canvas.width = 320;
      canvas.height = 320;

      ctx.drawImage(img, 0, 0, 320, 320);

      const imageData = ctx.getImageData(0, 0, 320, 320);
      const data = imageData.data;

      if (applyClahe) {
        // Contrast enhancement simulation
        const factor = 1 + (claheClipLimit - 1) * 0.35;
        for (let i = 0; i < data.length; i += 4) {
          // Grayscale check & CLAHE boost
          const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
          let newVal = (avg - 128) * factor + 128;
          newVal = Math.min(255, Math.max(0, newVal));
          data[i] = newVal;
          data[i + 1] = newVal;
          data[i + 2] = newVal;
        }
      }

      ctx.putImageData(imageData, 0, 0);
    };
    img.src = currentImageSrc;
  }, [currentImageSrc, applyClahe, applyNoiseRemoval, claheClipLimit]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFileError(null);
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setFileError('Please select a valid image file (PNG, JPEG, or WebP). DICOM files are not supported in this research build — convert to PNG/JPEG first.');
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setFileError('File size exceeds 20MB threshold.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      if (event.target?.result) {
        setUploadedImage(event.target.result as string);
        setSelectedSample(null);
        setImageName(file.name);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    setFileError(null);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleSampleClick = (sample: SampleXray) => {
    if (isLoading) return; // don't swap the source mid-inference
    setSelectedSample(sample);
    setUploadedImage(null);
    setImageName(sample.sampleResult.imageName);
    setFileError(null);
  };

  const handleSubmit = () => {
    if (!validationReport?.passed) return;
    onRunInference({
      imageName,
      imageData: currentImageSrc,
      model: selectedModel,
      clahe: applyClahe,
      noiseRemoval: applyNoiseRemoval,
      validation: validationReport,
    });
  };

  const validationPassed = validationReport?.passed ?? false;
  const validationFailed = validationReport !== null && !validationPassed;

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6 text-slate-800">
      {/* Header section */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
        <div>
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <UploadCloud className="w-5 h-5 text-indigo-600" />
            <span>Chest Radiograph Preprocessing &amp; Upload</span>
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Select a curated clinical sample or upload a PNG/JPEG/WebP Chest X-ray. Every image is AI-validated before analysis.
          </p>
        </div>

        {/* Model Selector */}
        <div className="flex items-center gap-2">
          <label className="text-xs font-semibold text-slate-600 whitespace-nowrap">Target Model:</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="bg-white border border-slate-300 text-slate-800 text-xs rounded-md px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-mono shadow-xs"
          >
            <option value="DenseNet-121 (CheXNet)">DenseNet-121 (CheXNet)</option>
            <option value="EfficientNet-B3">EfficientNet-B3</option>
            <option value="ConvNeXt-Base">ConvNeXt-Base</option>
            <option value="Swin Transformer (Swin-B)">Swin Transformer (Swin-B)</option>
            <option value="Vision Transformer (ViT-B/16)">Vision Transformer (ViT-B/16)</option>
          </select>
        </div>
      </div>

      {/* Main Grid: Upload & Preview */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Drag & Drop Zone + Sample Picker (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-all flex flex-col items-center justify-center min-h-[220px] ${
              dragActive
                ? 'border-indigo-500 bg-indigo-50/60 shadow-xs'
                : 'border-slate-300 bg-slate-50 hover:border-slate-400 hover:bg-slate-100/80'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp"
              onChange={handleFileChange}
              className="hidden"
            />
            <div className="w-12 h-12 rounded-full bg-white border border-slate-200 shadow-xs flex items-center justify-center mb-3 text-indigo-600">
              <UploadCloud className="w-6 h-6" />
            </div>
            <p className="text-xs font-semibold text-slate-700">
              Drag &amp; drop Chest X-ray image here, or <span className="text-indigo-600 underline">browse</span>
            </p>
            <p className="text-[11px] text-slate-500 mt-1">Supports PNG, JPEG, WebP. Max 20MB. DICOM parsing is not supported in this research build — convert DICOM to PNG/JPEG first.</p>

            {uploadedImage && (
              <div className="mt-3 inline-flex items-center gap-1.5 bg-indigo-50 text-indigo-700 border border-indigo-200 px-3 py-1 rounded-full text-xs font-mono font-medium">
                <CheckCircle2 className="w-3.5 h-3.5 text-indigo-600" />
                <span>Uploaded: {imageName}</span>
              </div>
            )}
          </div>

          {/* File-level errors (type / size) */}
          {fileError && (
            <div className="bg-rose-50 border border-rose-200 text-rose-700 p-3 rounded-md text-xs flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
              <span>{fileError}</span>
            </div>
          )}

          {/* AI Validation & Quality Assessment panel */}
          {validationBusy ? (
            <div className="flex items-center gap-2.5 bg-indigo-50/70 dark:bg-indigo-500/10 border border-indigo-200 dark:border-indigo-500/25 text-indigo-700 dark:text-indigo-300 p-3 rounded-lg text-xs animate-fade-in">
              <Loader2 className="w-4 h-4 animate-spin shrink-0" />
              <div>
                <p className="font-bold">Running AI validation &amp; quality assessment…</p>
                <p className="text-[11px] opacity-80 mt-0.5">Checking format, resolution, grayscale, orientation, exposure, focus &amp; thoracic anatomy</p>
              </div>
            </div>
          ) : validationReport && (
            <div
              className={cn(
                'rounded-lg border p-3 text-xs animate-fade-in-up',
                validationPassed
                  ? 'bg-emerald-50/80 dark:bg-emerald-500/10 border-emerald-200 dark:border-emerald-500/30'
                  : 'bg-rose-50/80 dark:bg-rose-500/10 border-rose-200 dark:border-rose-500/30'
              )}
            >
              {/* Result banner */}
              <div className="flex items-start gap-2.5">
                {validationPassed ? (
                  <BadgeCheck className="w-4.5 h-4.5 text-emerald-600 dark:text-emerald-400 shrink-0 mt-0.5" />
                ) : (
                  <ShieldX className="w-4.5 h-4.5 text-rose-600 dark:text-rose-400 shrink-0 mt-0.5" />
                )}
                <div className="flex-1 min-w-0">
                  <p className={cn('font-bold', validationPassed ? 'text-emerald-800 dark:text-emerald-300' : 'text-rose-800 dark:text-rose-300')}>
                    {validationPassed ? `Valid frontal chest X-ray — suitability ${validationReport.score}/100` : 'Image rejected — cannot run disease analysis'}
                  </p>
                  <p className={cn('text-[11px] leading-relaxed mt-0.5', validationPassed ? 'text-emerald-700/90 dark:text-emerald-400/80' : 'text-rose-700/90 dark:text-rose-400/80')}>
                    {validationReport.message}
                  </p>
                </div>
                <span className="shrink-0 font-mono text-[10px] font-bold px-2 py-1 rounded-full border bg-white/70 dark:bg-slate-900/60 text-slate-500 dark:text-slate-400">
                  {validationReport.score}/100
                </span>
              </div>

              {/* Toggle details */}
              <button
                onClick={() => setShowDetails((s) => !s)}
                className="mt-2.5 inline-flex items-center gap-1 text-[10px] font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors cursor-pointer"
              >
                <ShieldCheck className="w-3.5 h-3.5" />
                {showDetails ? 'Hide validation details' : 'View validation details'}
                <ChevronDown className={cn('w-3 h-3 transition-transform', showDetails && 'rotate-180')} />
              </button>

              {showDetails && (
                <ul className="mt-2 space-y-1.5 border-t border-current/10 pt-2.5 animate-fade-in">
                  {validationReport.checks.map((c) => (
                    <li key={c.key} className="flex items-start gap-2 text-[11px] leading-snug">
                      {c.status === 'pass' ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0 mt-0.5" />
                      ) : c.status === 'warn' ? (
                        <AlertCircle className="w-3.5 h-3.5 text-amber-500 shrink-0 mt-0.5" />
                      ) : (
                        <AlertCircle className="w-3.5 h-3.5 text-rose-500 shrink-0 mt-0.5" />
                      )}
                      <span>
                        <strong className="text-slate-700 dark:text-slate-200">{c.label}: </strong>
                        <span className={cn(
                          c.status === 'pass' ? 'text-emerald-700 dark:text-emerald-300' : c.status === 'warn' ? 'text-amber-700 dark:text-amber-300' : 'text-rose-700 dark:text-rose-300'
                        )}>
                          {c.status === 'pass' ? 'PASS' : c.status === 'warn' ? 'WARN' : 'FAIL'}
                        </span>
                        <span className="text-slate-500 dark:text-slate-400"> — {c.detail}</span>
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {/* Curated Sample Gallery */}
          <div className="space-y-2">
            <span className="text-xs font-bold text-slate-500 uppercase tracking-widest block">
              Or Choose Curated Clinical Dataset Case:
            </span>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
              {SAMPLE_XRAYS.map((sample) => {
                const isSelected = selectedSample?.id === sample.id;
                return (
                  <button
                    key={sample.id}
                    onClick={() => handleSampleClick(sample)}
                    disabled={isLoading}
                    className={`p-2.5 rounded-lg border text-left transition-all ${
                      isSelected
                        ? 'bg-indigo-50/80 border-indigo-500 ring-1 ring-indigo-500 shadow-xs'
                        : 'bg-white border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                    }`}
                  >
                    <div className="w-full h-20 bg-slate-900 rounded-md overflow-hidden mb-2 border border-slate-800 flex items-center justify-center">
                      <img src={sample.svgDataUrl} alt={sample.title} className="h-full object-contain" />
                    </div>
                    <div className="text-[11px] font-bold text-slate-800 truncate">{sample.title}</div>
                    <div className="text-[10px] text-indigo-600 font-semibold">{sample.category}</div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right: Preprocessing Controls & Canvas Live Preview (5 Cols) */}
        <div className="lg:col-span-5 bg-slate-50 border border-slate-200 rounded-lg p-4 space-y-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3 border-b border-slate-200 pb-2">
              <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest flex items-center gap-1.5">
                <Sliders className="w-4 h-4 text-indigo-600" />
                <span>Preprocessing Pipeline</span>
              </h3>
              <span className="text-[10px] text-slate-500 font-mono">OpenCV Albumentations</span>
            </div>

            {/* Canvas Preview Box */}
            <div className="relative w-full aspect-square bg-slate-900 rounded-lg border border-slate-800 overflow-hidden flex items-center justify-center mb-4 shadow-inner">
              <canvas ref={canvasRef} className="w-full h-full object-contain" />
              <div className="absolute top-2 left-2 bg-slate-900/90 backdrop-blur px-2 py-0.5 rounded text-[10px] font-mono text-indigo-300 border border-slate-700">
                1024x1024 Tensor
              </div>
            </div>

            {/* Preprocessing Toggles */}
            <div className="space-y-3 bg-white p-3 rounded-md border border-slate-200 text-xs shadow-2xs">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-semibold text-slate-800">CLAHE Contrast Enhancement</div>
                  <div className="text-[10px] text-slate-500">Contrast Limited Adaptive Histogram Equalization</div>
                </div>
                <input
                  type="checkbox"
                  checked={applyClahe}
                  onChange={(e) => setApplyClahe(e.target.checked)}
                  className="w-4 h-4 accent-indigo-600 rounded cursor-pointer"
                />
              </div>

              {applyClahe && (
                <div className="pt-1 space-y-1">
                  <div className="flex justify-between text-[10px] font-mono text-slate-500">
                    <span>Clip Limit</span>
                    <span>{claheClipLimit.toFixed(1)}</span>
                  </div>
                  <input
                    type="range"
                    min="1.0"
                    max="5.0"
                    step="0.1"
                    value={claheClipLimit}
                    onChange={(e) => setClaheClipLimit(parseFloat(e.target.value))}
                    className="w-full accent-indigo-600 cursor-pointer"
                  />
                </div>
              )}

              <div className="flex items-center justify-between border-t border-slate-100 pt-2">
                <div>
                  <div className="font-semibold text-slate-800">Bilateral Noise Filter</div>
                  <div className="text-[10px] text-slate-500">Preserves anatomical edges while smoothing noise</div>
                </div>
                <input
                  type="checkbox"
                  checked={applyNoiseRemoval}
                  onChange={(e) => setApplyNoiseRemoval(e.target.checked)}
                  className="w-4 h-4 accent-indigo-600 rounded cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Validation scope note */}
          <div className="flex items-start gap-2 text-[10px] leading-relaxed text-slate-500 bg-white/60 border border-slate-200 rounded-md p-2.5">
            <ScanLine className="w-3.5 h-3.5 text-indigo-500 shrink-0 mt-0.5" />
            <span>
              <strong className="text-slate-600">AI gating:</strong> photographs, CT/MRI/ultrasound slices, limb X-rays, documents, and blurry/under-exposed images are rejected before inference.
            </span>
          </div>

          {/* Action Trigger Button */}
          <button
            onClick={handleSubmit}
            disabled={isLoading || validationBusy || !validationPassed}
            className={cn(
              'w-full py-3 px-4 rounded-md font-bold text-xs tracking-wider uppercase transition-colors shadow-xs flex items-center justify-center gap-2 cursor-pointer',
              validationFailed
                ? 'bg-slate-300 dark:bg-slate-700 text-slate-600 dark:text-slate-400 cursor-not-allowed'
                : 'bg-indigo-600 hover:bg-indigo-700 text-white disabled:opacity-50'
            )}
          >
            {isLoading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Running PyTorch Grad-CAM Inference...</span>
              </>
            ) : validationBusy ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Validating image…</span>
              </>
            ) : validationFailed ? (
              <>
                <Lock className="w-4 h-4" />
                <span>Image Rejected — Cannot Analyze</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-white" />
                <span>Analyze Radiograph &amp; Generate Report</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
