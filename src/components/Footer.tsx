import React from 'react';
import { Activity, ShieldAlert, FileText, Github, ExternalLink, Cpu } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="bg-slate-900 border-t border-slate-800 text-slate-400 text-xs py-8 mt-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Column 1: Project Identity */}
          <div className="space-y-3 md:col-span-1">
            <div className="flex items-center gap-2 text-slate-100 font-bold text-sm">
              <div className="w-5 h-5 bg-indigo-500 rounded flex items-center justify-center text-white text-xs font-bold">M</div>
              <span>MedVision AI</span>
            </div>
            <p className="text-slate-400 text-xs leading-relaxed">
              An explainable deep learning platform for multi-label chest X-ray disease detection, Grad-CAM heatmap visualization, and automated structured radiology report generation.
            </p>
            <div className="flex items-center gap-2 pt-1">
              <span className="inline-flex items-center gap-1 bg-emerald-950/80 text-emerald-400 border border-emerald-800/60 px-2 py-0.5 rounded text-[10px] font-mono">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping"></span>
                PyTorch v2.2 CUDA
              </span>
              <span className="inline-flex items-center gap-1 bg-slate-800 text-slate-300 px-2 py-0.5 rounded text-[10px] font-mono">
                FP16 ONNX Runtime
              </span>
            </div>
          </div>

          {/* Column 2: Research Papers */}
          <div className="space-y-2">
            <h4 className="text-slate-200 font-bold text-xs uppercase tracking-widest">Key Research Papers</h4>
            <ul className="space-y-1.5 text-slate-400">
              <li className="hover:text-indigo-400 transition-colors cursor-pointer flex items-center gap-1">
                <span>CheXNet: Radiologist-Level Chest X-Ray Diagnosis (Rajpurkar et al.)</span>
                <ExternalLink className="w-3 h-3 text-slate-500" />
              </li>
              <li className="hover:text-indigo-400 transition-colors cursor-pointer flex items-center gap-1">
                <span>Grad-CAM: Visual Explanations from Deep Networks (Selvaraju et al.)</span>
                <ExternalLink className="w-3 h-3 text-slate-500" />
              </li>
              <li className="hover:text-indigo-400 transition-colors cursor-pointer flex items-center gap-1">
                <span>NIH ChestX-ray14 Benchmark Dataset (Wang et al.)</span>
                <ExternalLink className="w-3 h-3 text-slate-500" />
              </li>
            </ul>
          </div>

          {/* Column 3: Tech Stack */}
          <div className="space-y-2">
            <h4 className="text-slate-200 font-bold text-xs uppercase tracking-widest">Tech Stack &amp; Architecture</h4>
            <p className="text-slate-400 leading-relaxed text-[11px]">
              PyTorch, TorchVision, OpenCV, Albumentations, Captum Grad-CAM, FastAPI, React 19, TypeScript, TailwindCSS, Gemini 3.6 Flash, Chart.js / Recharts, jsPDF.
            </p>
            <div className="flex items-center gap-2 pt-1 text-slate-300">
              <Cpu className="w-3.5 h-3.5 text-indigo-400" />
              <span>NVIDIA TensorRT Optimized</span>
            </div>
          </div>

          {/* Column 4: Disclaimer & Licensing */}
          <div className="space-y-2 bg-amber-950/40 border border-amber-800/60 p-3 rounded-md text-amber-200/90">
            <div className="flex items-center gap-1.5 font-bold text-amber-300 text-xs">
              <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0" />
              <span>Mandatory Medical Disclaimer</span>
            </div>
            <p className="text-[11px] leading-relaxed">
              This application is built solely for educational, academic, portfolio, and research evaluation. It is NOT intended for clinical diagnosis, medical decision making, or primary radiological assessment.
            </p>
          </div>
        </div>

        <div className="pt-4 border-t border-slate-800/80 flex flex-col sm:flex-row items-center justify-between gap-4 text-[11px] text-slate-500">
          <div>
            &copy; {new Date().getFullYear()} MedVision AI. Developed for NVIDIA ML Engineer &amp; Computer Vision Portfolio.
          </div>
          <div className="flex items-center gap-4 font-mono">
            <span>License: Apache-2.0</span>
            <span>•</span>
            <span>Docker Ready</span>
            <span>•</span>
            <span>PyTorch 2.x</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
