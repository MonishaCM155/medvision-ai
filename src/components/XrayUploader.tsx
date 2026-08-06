import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, Image as ImageIcon, Sliders, Play, Sparkles, CheckCircle2, AlertCircle, RefreshCw } from 'lucide-react';
import { SAMPLE_XRAYS, SampleXray } from '../data/sampleXrays';

interface XrayUploaderProps {
  onRunInference: (data: {
    imageName: string;
    imageData: string;
    model: string;
    clahe: boolean;
    noiseRemoval: boolean;
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
  const [validationError, setValidationError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Active image source
  const currentImageSrc = uploadedImage || selectedSample?.svgDataUrl || SAMPLE_XRAYS[0].svgDataUrl;

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
    setValidationError(null);
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setValidationError('Please select a valid image file (PNG, JPEG, DICOM).');
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setValidationError('File size exceeds 20MB threshold.');
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
    setValidationError(null);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleSampleClick = (sample: SampleXray) => {
    setSelectedSample(sample);
    setUploadedImage(null);
    setImageName(sample.sampleResult.imageName);
    setValidationError(null);
  };

  const handleSubmit = () => {
    onRunInference({
      imageName,
      imageData: currentImageSrc,
      model: selectedModel,
      clahe: applyClahe,
      noiseRemoval: applyNoiseRemoval,
    });
  };

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
            Select a curated clinical sample or upload a DICOM/PNG Chest X-ray.
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
              accept="image/*,.dcm"
              onChange={handleFileChange}
              className="hidden"
            />
            <div className="w-12 h-12 rounded-full bg-white border border-slate-200 shadow-xs flex items-center justify-center mb-3 text-indigo-600">
              <UploadCloud className="w-6 h-6" />
            </div>
            <p className="text-xs font-semibold text-slate-700">
              Drag &amp; drop Chest X-ray image here, or <span className="text-indigo-600 underline">browse</span>
            </p>
            <p className="text-[11px] text-slate-500 mt-1">Supports PNG, JPEG, DICOM 16-bit. Max 20MB.</p>

            {uploadedImage && (
              <div className="mt-3 inline-flex items-center gap-1.5 bg-indigo-50 text-indigo-700 border border-indigo-200 px-3 py-1 rounded-full text-xs font-mono font-medium">
                <CheckCircle2 className="w-3.5 h-3.5 text-indigo-600" />
                <span>Uploaded: {imageName}</span>
              </div>
            )}
          </div>

          {validationError && (
            <div className="bg-rose-50 border border-rose-200 text-rose-700 p-3 rounded-md text-xs flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
              <span>{validationError}</span>
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

          {/* Action Trigger Button */}
          <button
            onClick={handleSubmit}
            disabled={isLoading}
            className="w-full py-3 px-4 rounded-md font-bold text-xs tracking-wider uppercase text-white bg-indigo-600 hover:bg-indigo-700 transition-colors shadow-xs flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
          >
            {isLoading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Running PyTorch Grad-CAM Inference...</span>
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
