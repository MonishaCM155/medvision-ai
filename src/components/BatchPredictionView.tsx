import React, { useState } from 'react';
import { Layers, UploadCloud, Play, CheckCircle2, RefreshCw, FileSpreadsheet, AlertCircle } from 'lucide-react';

export const BatchPredictionView: React.FC = () => {
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [batchResults, setBatchResults] = useState<any[]>([]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const filesArr = Array.from(e.target.files);
      setBatchFiles(filesArr);
      setBatchResults([]);
      setProgress(0);
    }
  };

  const handleExportCsv = () => {
    if (batchResults.length === 0) return;
    // RFC 4180: quote any cell containing a delimiter, quote, or newline; double embedded quotes
    const cell = (v: unknown) => {
      const s = String(v ?? '');
      return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const headers = ['File Name', 'Top Diagnosis', 'Confidence (%)', 'Severity', 'Latency (ms)', 'Status'].map(cell).join(',');
    const rows = batchResults
      .map((r) => [r.fileName, r.topDiagnosis, (r.confidence * 100).toFixed(1), r.severity, r.inferenceMs, r.status].map(cell).join(','))
      .join('\r\n');
    // UTF-8 BOM so Excel renders headers/diacritics correctly
    const blob = new Blob(['\uFEFF' + headers + '\r\n' + rows], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `MedVision_Batch_${Date.now()}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleRunBatch = async () => {
    if (batchFiles.length === 0) return;
    setIsProcessing(true);
    setProgress(10);

    // Simulate progressive processing
    for (let i = 1; i <= batchFiles.length; i++) {
      await new Promise((resolve) => setTimeout(resolve, 400));
      setProgress(Math.round((i / batchFiles.length) * 100));
    }

    try {
      const res = await fetch('/api/batch-predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          files: batchFiles.map((f) => ({ name: f.name, size: f.size })),
        }),
      });
      const data = await res.json();
      if (data.batchResults) {
        setBatchResults(data.batchResults);
      }
    } catch (err) {
      console.error('Batch error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6 text-slate-800">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
        <div>
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <Layers className="w-5 h-5 text-indigo-600" />
            <span>High-Throughput Batch Processing</span>
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Process cohort studies and bulk chest radiograph archives with parallel multi-GPU worker queues.
          </p>
        </div>

        {batchResults.length > 0 && (
          <button
            onClick={handleExportCsv}
            className="px-3.5 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold rounded-md shadow-xs flex items-center gap-2 transition-colors cursor-pointer"
          >
            <FileSpreadsheet className="w-4 h-4" />
            <span>Export Batch CSV</span>
          </button>
        )}
      </div>

      {/* Upload Box */}
      <div className="border-2 border-dashed border-slate-300 bg-slate-50 p-6 rounded-lg text-center space-y-3">
        <input
          type="file"
          multiple
          accept="image/*,.dcm"
          onChange={handleFileChange}
          id="batch-file-input"
          className="hidden"
        />
        <label htmlFor="batch-file-input" className="cursor-pointer inline-block">
          <div className="w-12 h-12 rounded-full bg-white border border-slate-200 shadow-xs flex items-center justify-center mx-auto mb-2 text-indigo-600">
            <UploadCloud className="w-6 h-6" />
          </div>
          <p className="text-xs font-semibold text-slate-700">
            Select Multiple Chest X-rays <span className="text-indigo-600 underline">Browse Files</span>
          </p>
          <p className="text-[11px] text-slate-500 mt-1">Batch size up to 100 images per request queue.</p>
        </label>

        {batchFiles.length > 0 && (
          <div className="pt-2">
            <span className="inline-flex items-center gap-1.5 bg-indigo-50 text-indigo-700 border border-indigo-200 px-3 py-1 rounded-full text-xs font-mono font-medium">
              <CheckCircle2 className="w-3.5 h-3.5 text-indigo-600" />
              <span>Selected {batchFiles.length} files</span>
            </span>
          </div>
        )}
      </div>

      {/* Run Action Bar */}
      {batchFiles.length > 0 && (
        <div className="space-y-3 bg-slate-50 p-4 rounded-lg border border-slate-200">
          <div className="flex items-center justify-between">
            <div className="text-xs font-semibold text-slate-700">
              Queue Progress: <span className="font-mono text-indigo-600">{progress}%</span>
            </div>
            <button
              onClick={handleRunBatch}
              disabled={isProcessing}
              className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs rounded-md shadow-xs flex items-center gap-2 cursor-pointer transition-colors disabled:opacity-50"
            >
              {isProcessing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Processing Queue...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  <span>Start Batch Inference ({batchFiles.length} Images)</span>
                </>
              )}
            </button>
          </div>

          <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden border border-slate-300">
            <div className="bg-indigo-600 h-full transition-all duration-300" style={{ width: `${progress}%` }}></div>
          </div>
        </div>
      )}

      {/* Batch Results Table */}
      {batchResults.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Batch Queue Execution Summary</h3>
          <div className="overflow-x-auto border border-slate-200 rounded-lg shadow-2xs">
            <table className="w-full text-left text-xs text-slate-700">
              <thead className="bg-slate-50 text-slate-500 font-bold text-[11px] uppercase tracking-widest border-b border-slate-200">
                <tr>
                  <th className="p-3">#</th>
                  <th className="p-3">File Name</th>
                  <th className="p-3">Primary Diagnosis</th>
                  <th className="p-3">Confidence</th>
                  <th className="p-3">Severity</th>
                  <th className="p-3">Latency</th>
                  <th className="p-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 bg-white font-mono text-xs">
                {batchResults.map((res, i) => (
                  <tr key={i} className="hover:bg-slate-50">
                    <td className="p-3 text-slate-400">{i + 1}</td>
                    <td className="p-3 font-semibold text-slate-900">{res.fileName}</td>
                    <td className="p-3 text-indigo-700 font-bold">{res.topDiagnosis}</td>
                    <td className="p-3">{(res.confidence * 100).toFixed(1)}%</td>
                    <td className="p-3">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          res.severity === 'High' ? 'bg-rose-50 text-rose-800 border border-rose-200' : 'bg-emerald-50 text-emerald-800 border border-emerald-200'
                        }`}
                      >
                        {res.severity}
                      </span>
                    </td>
                    <td className="p-3 text-slate-500">{res.inferenceMs}ms</td>
                    <td className="p-3 text-emerald-600 font-bold">{res.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
