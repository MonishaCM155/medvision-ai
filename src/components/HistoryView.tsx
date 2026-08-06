import React, { useState } from 'react';
import { History, Search, Trash2, Download, Eye, Calendar, Tag, ShieldAlert, Star, Bookmark } from 'lucide-react';
import { PredictionResult, SeverityLevel } from '../types';
import { cn } from '../utils/cn';

interface HistoryViewProps {
  history: PredictionResult[];
  bookmarkedIds?: Set<string>;
  onToggleBookmark?: (id: string) => void;
  onSelectResult: (result: PredictionResult) => void;
  onDeleteResult: (id: string) => void;
}

export const HistoryView: React.FC<HistoryViewProps> = ({
  history,
  bookmarkedIds = new Set(),
  onToggleBookmark,
  onSelectResult,
  onDeleteResult,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [severityFilter, setSeverityFilter] = useState<string>('All');
  const [bookmarkFilter, setBookmarkFilter] = useState<'all' | 'bookmarked'>('all');

  const filteredHistory = history.filter((item) => {
    const matchesSearch =
      item.imageName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.topDiagnosis.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesSeverity = severityFilter === 'All' || item.severity === severityFilter;
    const matchesBookmark = bookmarkFilter === 'all' || bookmarkedIds.has(item.id);
    return matchesSearch && matchesSeverity && matchesBookmark;
  });

  const bookmarkCount = history.filter((h) => bookmarkedIds.has(h.id)).length;

  const handleExportCsv = () => {
    const headers = 'ID,Timestamp,ImageName,ModelUsed,TopDiagnosis,Confidence,Severity,SeverityScore,InferenceMs\n';
    const rows = history
      .map(
        (h) =>
          `"${h.id}","${h.timestamp}","${h.imageName}","${h.modelUsed}","${h.topDiagnosis}",${(h.topConfidence * 100).toFixed(1)},"${h.severity}",${h.severityScore},${h.inferenceTimeMs}`
      )
      .join('\n');

    const blob = new Blob([headers + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `medvision_history_export_${Date.now()}.csv`;
    link.click();
  };

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6 text-slate-800">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
        <div>
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <History className="w-5 h-5 text-indigo-600" />
            <span>Inference History &amp; Audit Logs</span>
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Stored radiological diagnostic runs with execution latency, model selection, and severity scores.
          </p>
        </div>

        <button
          onClick={handleExportCsv}
          disabled={history.length === 0}
          className="px-3.5 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold rounded-md shadow-xs flex items-center gap-2 transition-colors cursor-pointer disabled:opacity-50"
        >
          <Download className="w-4 h-4" />
          <span>Export CSV</span>
        </button>
      </div>

      {/* Filters Bar */}
      <div className="grid grid-cols-1 sm:grid-cols-12 gap-3">
        <div className="sm:col-span-8 relative">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search by image name or top diagnosis..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-white border border-slate-300 text-slate-800 text-xs rounded-md pl-9 pr-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 shadow-2xs"
          />
        </div>

        <div className="sm:col-span-4 grid grid-cols-2 gap-3">
          <select
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            className="w-full bg-white border border-slate-300 text-slate-800 text-xs rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-mono shadow-2xs dark:bg-[var(--surface-2)] dark:border-[var(--border)] dark:text-[var(--text)]"
          >
            <option value="All">All Severity Levels</option>
            <option value="Low">Low</option>
            <option value="Moderate">Moderate</option>
            <option value="High">High</option>
            <option value="Critical">Critical</option>
          </select>
          <button
            onClick={() => setBookmarkFilter((f) => (f === 'all' ? 'bookmarked' : 'all'))}
            className={cn(
              'flex items-center justify-center gap-1.5 rounded-md border px-3 py-2 text-xs font-semibold transition-colors cursor-pointer',
              bookmarkFilter === 'bookmarked'
                ? 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/40'
                : 'bg-white text-slate-600 border-slate-300 hover:border-amber-400 dark:bg-[var(--surface-2)] dark:border-[var(--border)] dark:text-[var(--text)]'
            )}
            title={bookmarkFilter === 'bookmarked' ? 'Show all records' : 'Show bookmarked records only'}
          >
            <Bookmark className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Bookmarked</span>
            <span className="font-mono text-[10px]">{bookmarkCount}</span>
          </button>
        </div>
      </div>

      {/* Table */}
      {filteredHistory.length === 0 ? (
        <div className="text-center py-12 bg-slate-50 rounded-lg border border-slate-200 text-slate-500 text-xs">
          No history records match the filter criteria.
        </div>
      ) : (
        <div className="overflow-x-auto border border-slate-200 rounded-lg shadow-2xs">
          <table className="w-full text-left text-xs text-slate-700">
            <thead className="bg-slate-50 text-slate-500 font-bold text-[11px] uppercase tracking-widest border-b border-slate-200">
              <tr>
                <th className="p-3">Thumbnail</th>
                <th className="p-3">Image Name</th>
                <th className="p-3">Top Diagnosis</th>
                <th className="p-3">Confidence</th>
                <th className="p-3">Severity</th>
                <th className="p-3">Model</th>
                <th className="p-3">Latency</th>
                <th className="p-3 text-right">Actions</th>
                <th className="p-3 w-12"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white">
              {filteredHistory.map((item) => (
                <tr key={item.id} className="hover:bg-slate-50 transition-colors">
                  <td className="p-3">
                    <div className="w-10 h-10 bg-slate-900 rounded border border-slate-300 overflow-hidden">
                      <img src={item.originalImageUrl} alt="Thumb" className="w-full h-full object-cover" />
                    </div>
                  </td>
                  <td className="p-3 font-mono font-medium text-slate-900">{item.imageName}</td>
                  <td className="p-3 font-bold text-indigo-700">{item.topDiagnosis}</td>
                  <td className="p-3 font-mono">{(item.topConfidence * 100).toFixed(1)}%</td>
                  <td className="p-3">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold font-mono border ${
                        item.severity === 'Critical'
                          ? 'bg-rose-50 text-rose-800 border-rose-200'
                          : item.severity === 'High'
                          ? 'bg-orange-50 text-orange-800 border-orange-200'
                          : item.severity === 'Moderate'
                          ? 'bg-amber-50 text-amber-800 border-amber-200'
                          : 'bg-emerald-50 text-emerald-800 border-emerald-200'
                      }`}
                    >
                      {item.severity}
                    </span>
                  </td>
                  <td className="p-3 font-mono text-slate-500 text-[11px]">{item.modelUsed.split(' ')[0]}</td>
                  <td className="p-3 font-mono text-slate-500 text-[11px]">{item.inferenceTimeMs}ms</td>
                  <td className="p-3 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <button
                        onClick={() => onSelectResult(item)}
                        className="p-1.5 text-slate-500 hover:text-indigo-600 hover:bg-indigo-50 rounded transition-colors cursor-pointer"
                        title="Inspect Result"
                      >
                        <Eye className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => onDeleteResult(item.id)}
                        className="p-1.5 text-slate-500 hover:text-rose-600 hover:bg-rose-50 rounded transition-colors cursor-pointer"
                        title="Delete Record"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </td>
                  <td className="p-3 text-center">
                    {onToggleBookmark && (
                      <button
                        onClick={() => onToggleBookmark(item.id)}
                        className={cn(
                          'p-1.5 rounded transition-all cursor-pointer',
                          bookmarkedIds.has(item.id)
                            ? 'text-amber-500 hover:text-amber-600 scale-110'
                            : 'text-slate-300 hover:text-amber-400'
                        )}
                        title={bookmarkedIds.has(item.id) ? 'Remove bookmark' : 'Bookmark record'}
                      >
                        <Star className={cn('w-4 h-4', bookmarkedIds.has(item.id) && 'fill-amber-400 text-amber-400')} />
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
