import React, { useState } from 'react';
import { FileText, Cpu, Terminal, Layers, BookOpen, ExternalLink, Code2, ShieldAlert } from 'lucide-react';

export const DocsView: React.FC = () => {
  const [activeSection, setActiveSection] = useState<'architecture' | 'api' | 'training' | 'docker'>('architecture');

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6 text-slate-800">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
        <div>
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-indigo-600" />
            <span>Technical Documentation &amp; Architecture Diagrams</span>
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Comprehensive system specifications, PyTorch training pipelines, REST API specs, and Docker deployment guidelines.
          </p>
        </div>

        {/* Section Navigation */}
        <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-md border border-slate-200">
          <button
            onClick={() => setActiveSection('architecture')}
            className={`px-3 py-1 rounded text-xs font-semibold transition-all ${
              activeSection === 'architecture' ? 'bg-indigo-600 text-white shadow-2xs' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Architecture
          </button>
          <button
            onClick={() => setActiveSection('api')}
            className={`px-3 py-1 rounded text-xs font-semibold transition-all ${
              activeSection === 'api' ? 'bg-indigo-600 text-white shadow-2xs' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            REST APIs
          </button>
          <button
            onClick={() => setActiveSection('training')}
            className={`px-3 py-1 rounded text-xs font-semibold transition-all ${
              activeSection === 'training' ? 'bg-indigo-600 text-white shadow-2xs' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            PyTorch Pipeline
          </button>
          <button
            onClick={() => setActiveSection('docker')}
            className={`px-3 py-1 rounded text-xs font-semibold transition-all ${
              activeSection === 'docker' ? 'bg-indigo-600 text-white shadow-2xs' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Docker &amp; CI
          </button>
        </div>
      </div>

      {/* Content based on active section */}
      {activeSection === 'architecture' && (
        <div className="space-y-6 text-xs text-slate-700 leading-relaxed">
          {/* Overview */}
          <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-2">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Layers className="w-4 h-4 text-indigo-600" />
              <span>System Architecture Overview</span>
            </h3>
            <p>
              MedVision AI decouples real-time computer vision inference from natural language report synthesis. The deep learning backbone (DenseNet-121 / ConvNeXt / Swin Transformer) processes incoming chest radiographs using PyTorch with TorchVision and OpenCV, extracting spatial activation maps via Grad-CAM hooks. The resulting activations feed into Gemini 3.6 Flash for contextual radiology report generation.
            </p>
          </div>

          {/* Architecture Box Diagram */}
          <div className="bg-slate-50 p-5 rounded-lg border border-slate-200 font-mono text-[11px] space-y-3">
            <div className="text-indigo-700 font-bold text-xs uppercase tracking-widest">End-to-End Inference Data Flow Diagram</div>
            <pre className="bg-slate-900 p-4 rounded-md border border-slate-800 overflow-x-auto text-slate-200">
{`+-----------------------+     +--------------------------+     +-------------------------+
| Chest X-Ray Input     | --> | Image Preprocessing      | --> | PyTorch Deep Backbone   |
| (1024x1024 DICOM/PNG) |     | (CLAHE + Noise Removal)  |     | (DenseNet121 / Swin)    |
+-----------------------+     +--------------------------+     +-------------------------+
                                                                            |
                                                                            v
+-----------------------+     +--------------------------+     +-------------------------+
| Gemini 3.6 Flash      | <-- | Grad-CAM Hook Generator  | <-- | Sigmoid Classifier      |
| Structured Radiology  |     | (Target Layer: Conv5)    |     | (10 Pathology Probabilities)|
+-----------------------+     +--------------------------+     +-------------------------+
           |                                                                |
           v                                                                v
+-----------------------+                                      +-------------------------+
| Hospital PDF Export   | <---------------------------------- | Visual Heatmap Overlay  |
| (jsPDF + html2canvas) |                                      | (Jet Colormap & BBoxes) |
+-----------------------+                                      +-------------------------+`}
            </pre>
          </div>
        </div>
      )}

      {activeSection === 'api' && (
        <div className="space-y-4 text-xs font-mono">
          <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-3">
            <div className="flex items-center gap-2 text-emerald-800 font-bold text-sm">
              <span className="bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded text-xs">POST</span>
              <span>/api/predict</span>
            </div>
            <p className="text-slate-600 text-xs font-sans">
              Main inference endpoint for chest radiograph disease classification, Grad-CAM heatmap generation, and Gemini report creation.
            </p>

            <div className="space-y-1">
              <span className="text-slate-500 text-[10px] block font-bold uppercase tracking-wider">REQUEST BODY JSON:</span>
              <pre className="bg-slate-900 p-3 rounded-md border border-slate-800 text-indigo-300 text-[11px]">
{`{
  "imageName": "chest_xray_001.dcm",
  "imageData": "data:image/png;base64,...",
  "model": "DenseNet-121 (CheXNet)",
  "clahe": true,
  "noiseRemoval": true
}`}
              </pre>
            </div>

            <div className="space-y-1">
              <span className="text-slate-500 text-[10px] block font-bold uppercase tracking-wider">SUCCESS RESPONSE (200 OK):</span>
              <pre className="bg-slate-900 p-3 rounded-md border border-slate-800 text-slate-200 text-[11px]">
{`{
  "id": "pred_172248_a8b9",
  "topDiagnosis": "Pneumonia",
  "topConfidence": 0.94,
  "severity": "High",
  "severityScore": 82,
  "diseases": [ ... ],
  "gradCamRegions": [ ... ],
  "report": { ... }
}`}
              </pre>
            </div>
          </div>

          <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-2">
            <div className="flex items-center gap-2 text-indigo-700 font-bold text-sm">
              <span className="bg-indigo-50 border border-indigo-200 px-2 py-0.5 rounded text-xs">GET</span>
              <span>/api/history</span>
            </div>
            <p className="text-slate-600 text-xs font-sans">
              Returns all stored inference logs and history records in chronological order.
            </p>
          </div>
        </div>
      )}

      {activeSection === 'training' && (
        <div className="space-y-4 text-xs leading-relaxed text-slate-700">
          <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-3">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Terminal className="w-4 h-4 text-indigo-600" />
              <span>PyTorch Training Pipeline Command</span>
            </h3>
            <p className="text-slate-600">
              Run multi-label chest X-ray training using weighted Binary Cross Entropy (BCE) with mixed precision AMP and Cosine Annealing learning rate scheduler.
            </p>

            <pre className="bg-slate-900 p-3.5 rounded-md border border-slate-800 text-indigo-300 font-mono text-[11px] overflow-x-auto">
{`# Train DenseNet121 on NIH ChestX-ray14 dataset
python training/train.py \\
    --data_dir ./datasets/nih_chestxray \\
    --architecture densenet121 \\
    --epochs 50 \\
    --batch_size 32 \\
    --lr 1e-4 \\
    --loss_fn weighted_bce \\
    --amp_fp16 \\
    --output_dir ./checkpoints/densenet121_best.pt`}
            </pre>
          </div>
        </div>
      )}

      {activeSection === 'docker' && (
        <div className="space-y-4 text-xs leading-relaxed text-slate-700">
          <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-3">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Code2 className="w-4 h-4 text-indigo-600" />
              <span>Docker &amp; Docker Compose Commands</span>
            </h3>
            <p className="text-slate-600">
              Build and launch the complete multi-container stack including PyTorch CUDA runtime, FastAPI backend, Vite React frontend, and MLflow server.
            </p>

            <pre className="bg-slate-900 p-3.5 rounded-md border border-slate-800 text-emerald-400 font-mono text-[11px] overflow-x-auto">
{`# Launch container stack with NVIDIA GPU runtime
docker-compose up --build -d

# Check service logs
docker-compose logs -f backend

# Run pytest unit tests inside backend container
docker-compose exec backend pytest tests/`}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};
