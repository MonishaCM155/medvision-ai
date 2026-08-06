import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Users, Search, Filter, UserPlus, Activity, AlertTriangle, X, Phone, Mail,
  Droplets, ClipboardList, CalendarClock, Pill, Stethoscope, History, FileText,
  ChevronRight, ArrowLeft, HeartPulse, ShieldCheck, MapPin, Download,
  ArrowUp, ArrowDown, ArrowUpDown,
} from 'lucide-react';
import { useApp } from '../contexts/AppContext';
import { api } from '../services/api';
import { PatientDetail, PatientRecord, SeverityLevel } from '../types';
import { GlassCard, SectionHeader, SeverityPill, LiveBadge } from '../components/ui/GlassCard';
import { StatCard } from '../components/ui/StatCard';
import { cn } from '../utils/cn';

const STATUS_STYLES: Record<PatientRecord['status'], string> = {
  Admitted: 'bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border-indigo-500/25',
  Outpatient: 'bg-sky-500/10 text-sky-600 dark:text-sky-400 border-sky-500/25',
  ICU: 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/25',
  Discharged: 'bg-slate-500/10 text-slate-500 dark:text-slate-400 border-slate-500/25',
  Critical: 'bg-red-500/15 text-red-600 dark:text-red-400 border-red-500/30',
};

const KINDS = {
  visit: { label: 'Visit', cls: 'bg-indigo-500/10 text-indigo-500 border-indigo-500/25' },
  report: { label: 'AI Report', cls: 'bg-sky-500/10 text-sky-500 border-sky-500/25' },
  lab: { label: 'Lab', cls: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/25' },
  medication: { label: 'Medication', cls: 'bg-amber-500/10 text-amber-500 border-amber-500/25' },
};

export const Patients: React.FC = () => {
  const { pushNotification } = useApp();
  const [patients, setPatients] = useState<PatientRecord[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<PatientDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [query, setQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('All');
  const [deptFilter, setDeptFilter] = useState<string>('All');
  const [sortKey, setSortKey] = useState<'name' | 'riskScore' | 'lastVisit' | 'status' | null>(null);
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  const mountedRef = useRef(true);

  // ESC closes the profile drawer
  useEffect(() => {
    if (!selectedId) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSelectedId(null);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [selectedId]);

  useEffect(() => {
    mountedRef.current = true;
    api.getPatients().then((p) => {
      if (!mountedRef.current) return;
      setPatients(p);
      setLoading(false);
    });
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const openPatient = async (id: string) => {
    setSelectedId(id);
    setDetailLoading(true);
    const d = await api.getPatient(id);
    if (!mountedRef.current) return;
    setDetail(d);
    setDetailLoading(false);
  };

  const filtered = useMemo(() => {
    const list = patients.filter((p) => {
      const q = query.toLowerCase();
      const matchesQuery =
        !q ||
        p.name.toLowerCase().includes(q) ||
        p.id.toLowerCase().includes(q) ||
        (p.topDiagnosis ?? '').toLowerCase().includes(q);
      const matchesStatus = statusFilter === 'All' || p.status === statusFilter;
      const matchesDept = deptFilter === 'All' || p.department === deptFilter;
      return matchesQuery && matchesStatus && matchesDept;
    });
    if (!sortKey) return list;
    const dir = sortDir === 'asc' ? 1 : -1;
    return [...list].sort((a, b) => {
      const va = a[sortKey];
      const vb = b[sortKey];
      if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * dir;
      return String(va ?? '').localeCompare(String(vb ?? '')) * dir;
    });
  }, [patients, query, statusFilter, deptFilter, sortKey, sortDir]);

  const toggleSort = (key: typeof sortKey) => {
    if (!key) return;
    setSortKey(key);
    setSortDir((d) => (sortKey === key && d === 'asc' ? 'desc' : 'asc'));
  };

  const departments = useMemo(() => Array.from(new Set(patients.map((p) => p.department))), [patients]);

  const handleAddPatient = () => {
    pushNotification({ kind: 'info', title: 'Patient intake', body: 'New patient registration form opened. (Demo)' });
  };

  const handleExportCsv = () => {
    if (filtered.length === 0) return;
    const cell = (v: unknown) => {
      const s = String(v ?? '');
      return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const headers = ['ID', 'Name', 'Age', 'Sex', 'Blood Group', 'Department', 'Status', 'Top Diagnosis', 'Risk Score', 'Last Visit'].map(cell).join(',');
    const rows = filtered
      .map((p) =>
        [p.id, p.name, p.age, p.sex, p.bloodGroup, p.department, p.status, p.topDiagnosis ?? '', p.riskScore, p.lastVisit].map(cell).join(',')
      )
      .join('\r\n');
    const blob = new Blob(['\uFEFF' + headers + '\r\n' + rows], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `MedVision_Patients_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
    pushNotification({ kind: 'success', title: 'Registry exported', body: `${filtered.length} patient records written to CSV.` });
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-black tracking-tight text-slate-900 dark:text-white">Patient Management</h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Complete EHR-style patient records, visits, and AI-assisted timelines</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleExportCsv}
            disabled={filtered.length === 0}
            className="inline-flex items-center gap-1.5 text-xs font-bold px-3.5 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white/70 dark:bg-slate-800/70 text-slate-600 dark:text-slate-300 hover:border-primary-400 hover:text-primary-600 dark:hover:text-primary-400 transition-all cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
            title="Export the filtered registry as CSV"
          >
            <Download className="w-4 h-4" />
            Export CSV
          </button>
          <button
            onClick={handleAddPatient}
            className="btn-gradient text-white text-xs font-bold px-4 py-2.5 rounded-xl flex items-center gap-2 cursor-pointer"
          >
            <UserPlus className="w-4 h-4" />
            Register Patient
          </button>
        </div>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total Records" value={patients.length ? patients.length.toLocaleString() : '—'} icon={<Users className="w-5 h-5" />} trend="8420 system-wide" trendDirection="flat" accent="from-indigo-500 to-violet-500" />
        <StatCard label="Critical / ICU" value={patients.filter((p) => p.status === 'Critical' || p.status === 'ICU').length.toLocaleString()} icon={<AlertTriangle className="w-5 h-5" />} trend="Priority triage" trendDirection="up" trendPositive={false} accent="from-rose-500 to-red-600" />
        <StatCard label="Avg Risk Score" value="65.2" suffix="/100" icon={<HeartPulse className="w-5 h-5" />} trend="−2.1% this week" trendDirection="down" accent="from-emerald-500 to-teal-600" />
        <StatCard label="Follow-ups Due" value="14" icon={<CalendarClock className="w-5 h-5" />} trend="Next 7 days" trendDirection="flat" accent="from-amber-500 to-orange-500" />
      </div>

      {/* Filters */}
      <GlassCard className="!p-4">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-3">
          <div className="md:col-span-5 relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by name, patient ID, or diagnosis…"
              className="w-full bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 text-slate-800 dark:text-slate-200 text-xs rounded-xl pl-9 pr-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-primary-400/50"
            />
          </div>
          <div className="md:col-span-3">
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="w-full bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 text-slate-800 dark:text-slate-200 text-xs rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-primary-400/50 font-mono"
            >
              <option value="All">All Statuses</option>
              <option value="Admitted">Admitted</option>
              <option value="Outpatient">Outpatient</option>
              <option value="ICU">ICU</option>
              <option value="Critical">Critical</option>
              <option value="Discharged">Discharged</option>
            </select>
          </div>
          <div className="md:col-span-3">
            <select
              value={deptFilter}
              onChange={(e) => setDeptFilter(e.target.value)}
              className="w-full bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 text-slate-800 dark:text-slate-200 text-xs rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-primary-400/50 font-mono"
            >
              <option value="All">All Departments</option>
              {departments.map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>
          <div className="md:col-span-1 flex items-center justify-end">
            <span className="inline-flex items-center gap-1.5 text-[11px] font-mono text-slate-500 bg-slate-100 dark:bg-slate-800 rounded-lg px-2.5 py-2">
              <Filter className="w-3.5 h-3.5" />
              {filtered.length}
            </span>
          </div>
        </div>
      </GlassCard>

      {/* Patient table */}
      {loading ? (
        <div className="space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="skeleton h-16" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <GlassCard className="text-center py-16">
          <div className="w-14 h-14 mx-auto rounded-2xl bg-slate-100 dark:bg-slate-800 flex items-center justify-center text-slate-400 mb-3">
            <Search className="w-6 h-6" />
          </div>
          <p className="text-sm font-bold text-slate-700 dark:text-slate-200">No patients match your filters</p>
          <p className="text-xs text-slate-500 mt-1">Try adjusting the search query or clearing filters.</p>
        </GlassCard>
      ) : (
        <GlassCard className="!p-0 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-700 dark:text-slate-300">
              <thead className="bg-slate-50 dark:bg-slate-800/60 text-slate-500 dark:text-slate-400 font-bold text-[10px] uppercase tracking-widest border-b border-slate-100 dark:border-slate-800">
                <tr>
                  {([
                    { key: 'name' as const, label: 'Patient' },
                    { key: 'status' as const, label: 'Status' },
                    { key: 'riskScore' as const, label: 'Risk' },
                    { key: 'lastVisit' as const, label: 'Last Visit' },
                  ] as const).map(({ key, label }) => {
                    const active = sortKey === key;
                    return (
                      <th key={key} className="p-3.5">
                        <button
                          onClick={() => toggleSort(key)}
                          className="inline-flex items-center gap-1 hover:text-primary-600 dark:hover:text-primary-400 transition-colors cursor-pointer"
                          title={`Sort by ${label}`}
                        >
                          {label}
                          {active ? (sortDir === 'asc' ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />) : <ArrowUpDown className="w-3 h-3 opacity-40" />}
                        </button>
                      </th>
                    );
                  })}
                  <th className="p-3.5">ID / Blood</th>
                  <th className="p-3.5">Department</th>
                  <th className="p-3.5">Top Diagnosis</th>
                  <th className="p-3.5 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60">
                {filtered.map((p) => {
                  const riskColor = p.riskScore > 75 ? 'text-rose-500' : p.riskScore > 50 ? 'text-amber-500' : 'text-emerald-500';
                  return (
                    <tr key={p.id} onClick={() => openPatient(p.id)} className="hover:bg-primary-50/40 dark:hover:bg-primary-900/10 transition-colors cursor-pointer group">
                      <td className="p-3.5">
                        <div className="flex items-center gap-2.5">
                          <span
                            className="w-8 h-8 rounded-full text-white text-[11px] font-bold flex items-center justify-center shrink-0 shadow"
                            style={{ background: `linear-gradient(135deg, hsl(${p.avatarHue} 70% 55%), hsl(${p.avatarHue + 40} 70% 45%))` }}
                          >
                            {p.name.split(' ').map((w) => w[0]).slice(0, 2).join('')}
                          </span>
                          <div>
                            <p className="font-bold text-slate-900 dark:text-white">{p.name}</p>
                            <p className="text-[10px] text-slate-400">{p.age} y/o · {p.sex}</p>
                          </div>
                        </div>
                      </td>
                      <td className="p-3.5">
                        <p className="font-mono font-semibold text-indigo-600 dark:text-indigo-400">{p.id}</p>
                        <p className="text-[10px] text-slate-400">Blood {p.bloodGroup}</p>
                      </td>
                      <td className="p-3.5 text-slate-600 dark:text-slate-300">{p.department}</td>
                      <td className="p-3.5">
                        <span className={cn('inline-flex px-2 py-0.5 rounded-full border text-[10px] font-bold font-mono', STATUS_STYLES[p.status])}>
                          {p.status}
                        </span>
                      </td>
                      <td className="p-3.5 font-semibold text-slate-800 dark:text-slate-200">{p.topDiagnosis ?? '—'}</td>
                      <td className="p-3.5">
                        <div className="flex items-center gap-2">
                          <div className="w-14 h-1.5 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                            <div
                              className={cn('h-full rounded-full', riskColor)}
                              style={{ width: `${p.riskScore}%` }}
                            />
                          </div>
                          <span className={cn('font-mono font-bold text-[11px]', riskColor)}>{p.riskScore}</span>
                        </div>
                      </td>
                      <td className="p-3.5 font-mono text-[11px] text-slate-500">{p.lastVisit}</td>
                      <td className="p-3.5 text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            openPatient(p.id);
                          }}
                          className="p-2 rounded-lg text-slate-400 hover:text-primary-600 hover:bg-primary-50 dark:hover:bg-primary-900/20 transition-colors cursor-pointer"
                          title="Open patient record"
                        >
                          <ChevronRight className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </GlassCard>
      )}

      {/* Profile drawer */}
      {selectedId && (
        <div className="fixed inset-0 z-50 bg-slate-950/50 backdrop-blur-sm flex justify-end animate-fade-in" onClick={() => setSelectedId(null)}>
          <div
            className="w-full max-w-2xl h-full bg-white dark:bg-slate-900 shadow-2xl overflow-y-auto animate-slide-in-right"
            onClick={(e) => e.stopPropagation()}
          >
            {detailLoading || !detail ? (
              <div className="p-6 space-y-4">
                <div className="skeleton h-40" />
                <div className="skeleton h-64" />
              </div>
            ) : (
              <PatientProfile detail={detail} onClose={() => setSelectedId(null)} />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

const PatientProfile: React.FC<{ detail: PatientDetail; onClose: () => void }> = ({ detail, onClose }) => {
  const { pushNotification } = useApp();
  const p = detail;
  return (
    <div>
      {/* Drawer header */}
      <div className="sticky top-0 z-10 glass-strong px-6 py-4 flex items-center justify-between border-b border-slate-200/60 dark:border-slate-800">
        <button onClick={onClose} className="flex items-center gap-1.5 text-xs font-semibold text-slate-500 hover:text-primary-600 transition-colors cursor-pointer">
          <ArrowLeft className="w-4 h-4" /> Back to list
        </button>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-slate-400 flex items-center gap-1">
            <ShieldCheck className="w-3 h-3 text-emerald-500" /> EHR #{p.id.replace('PAT-', '')}
          </span>
          <button
            onClick={() => pushNotification({ kind: 'success', title: 'Record updated', body: `${p.id} — ${p.name} record saved to EHR.` })}
            className="text-[11px] font-bold bg-indigo-600 text-white px-3 py-1.5 rounded-lg hover:bg-indigo-700 transition-colors cursor-pointer"
          >
            Save
          </button>
          <button onClick={onClose} className="p-2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors cursor-pointer">
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Identity card */}
        <div className="surface-card ring-gradient p-5">
          <div className="flex items-center gap-4">
            <span
              className="w-16 h-16 rounded-2xl text-white text-xl font-black flex items-center justify-center shadow-lg"
              style={{ background: `linear-gradient(135deg, hsl(${p.avatarHue} 70% 55%), hsl(${p.avatarHue + 40} 70% 45%))` }}
            >
              {p.name.split(' ').map((w) => w[0]).slice(0, 2).join('')}
            </span>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <h2 className="text-lg font-black text-slate-900 dark:text-white">{p.name}</h2>
                <span className={cn('px-2 py-0.5 rounded-full border text-[10px] font-bold font-mono', STATUS_STYLES[p.status])}>{p.status}</span>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 font-mono">{p.id}</p>
              <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-[11px] text-slate-600 dark:text-slate-300">
                <span className="inline-flex items-center gap-1"><MapPin className="w-3 h-3 text-slate-400" />{p.department}</span>
                <span className="inline-flex items-center gap-1"><Phone className="w-3 h-3 text-slate-400" />{p.phone}</span>
                <span className="inline-flex items-center gap-1"><Mail className="w-3 h-3 text-slate-400" />{p.email}</span>
              </div>
            </div>
            <div className="text-right shrink-0">
              <p className="text-[10px] font-mono text-slate-400 uppercase tracking-widest">Risk Score</p>
              <p className={cn('text-2xl font-black font-mono', p.riskScore > 75 ? 'text-rose-500' : p.riskScore > 50 ? 'text-amber-500' : 'text-emerald-500')}>
                {p.riskScore}
              </p>
              <SeverityPill level={p.lastSeverity ?? 'Low'} />
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-5 pt-4 border-t border-slate-100 dark:border-slate-800 text-center">
            {[
              { label: 'Age', value: `${p.age} y/o` },
              { label: 'Sex', value: p.sex },
              { label: 'Blood Group', value: p.bloodGroup },
              { label: 'Admitted', value: p.admissionDate },
            ].map((s) => (
              <div key={s.label} className="bg-slate-50 dark:bg-slate-800/60 rounded-xl py-2.5 px-2">
                <p className="text-[9px] font-mono uppercase tracking-widest text-slate-400">{s.label}</p>
                <p className="text-xs font-bold text-slate-800 dark:text-slate-100 mt-0.5">{s.value}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Medical summary */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="surface-card p-4 md:col-span-1">
            <h3 className="text-xs font-bold text-slate-800 dark:text-slate-100 flex items-center gap-1.5 mb-2.5">
              <ClipboardList className="w-4 h-4 text-indigo-500" /> Medical History
            </h3>
            <ul className="space-y-1.5">
              {p.medicalHistory.map((m, i) => (
                <li key={i} className="text-[11px] text-slate-600 dark:text-slate-300 flex items-start gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 mt-1 shrink-0" />{m}
                </li>
              ))}
            </ul>
          </div>
          <div className="surface-card p-4">
            <h3 className="text-xs font-bold text-slate-800 dark:text-slate-100 flex items-center gap-1.5 mb-2.5">
              <AlertTriangle className="w-4 h-4 text-rose-500" /> Allergies
            </h3>
            <ul className="space-y-1.5">
              {p.allergies.map((a, i) => (
                <li key={i} className="text-[11px] text-rose-600 dark:text-rose-400 flex items-start gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-rose-400 mt-1 shrink-0" />{a}
                </li>
              ))}
            </ul>
          </div>
          <div className="surface-card p-4">
            <h3 className="text-xs font-bold text-slate-800 dark:text-slate-100 flex items-center gap-1.5 mb-2.5">
              <Pill className="w-4 h-4 text-emerald-500" /> Current Medications
            </h3>
            <ul className="space-y-1.5">
              {p.medications.map((m, i) => (
                <li key={i} className="text-[11px] text-slate-600 dark:text-slate-300 flex items-start gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1 shrink-0" />{m}
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Visits */}
        <div className="space-y-2.5">
          <h3 className="text-xs font-bold text-slate-800 dark:text-slate-100 flex items-center gap-1.5">
            <Stethoscope className="w-4 h-4 text-sky-500" /> Visit History
          </h3>
          {p.visits.length === 0 && <p className="text-xs text-slate-400">No recorded visits.</p>}
          {p.visits.map((v) => (
            <div key={v.id} className="surface-card p-4 hover:border-primary-300 dark:hover:border-primary-700 transition-colors">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-600 dark:text-sky-400 border border-sky-500/25 font-mono">{v.type}</span>
                  <span className="text-[11px] font-mono text-slate-400">{v.date}</span>
                  <SeverityPill level={v.severity} />
                </div>
                <span className="text-[10px] text-slate-400">{v.physician}</span>
              </div>
              <p className="text-xs text-slate-600 dark:text-slate-300 mt-2 leading-relaxed">{v.findings}</p>
              {v.reports.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-2">
                  {v.reports.map((r, i) => (
                    <span key={i} className="inline-flex items-center gap-1 text-[10px] font-mono text-indigo-600 dark:text-indigo-400 bg-indigo-500/5 border border-indigo-500/20 px-2 py-0.5 rounded-full">
                      <FileText className="w-3 h-3" />{r}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Timeline */}
        <div className="space-y-2.5">
          <h3 className="text-xs font-bold text-slate-800 dark:text-slate-100 flex items-center gap-1.5">
            <History className="w-4 h-4 text-violet-500" /> Patient Timeline
          </h3>
          <div className="relative pl-5 border-l-2 border-slate-100 dark:border-slate-800 space-y-4">
            {p.timeline.map((t, i) => {
              const kind = KINDS[t.kind] ?? KINDS.visit;
              return (
                <div key={i} className="relative">
                  <span className={cn('absolute -left-[1.6rem] top-0.5 w-4 h-4 rounded-full border-2 border-white dark:border-slate-900 flex items-center justify-center', kind.cls.split(' ')[0])}>
                    <span className="w-1.5 h-1.5 rounded-full bg-current" />
                  </span>
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-[11px] font-bold text-slate-800 dark:text-slate-100">{t.title}</span>
                    <span className={cn('text-[9px] font-bold px-1.5 py-0.5 rounded-full border', kind.cls)}>{kind.label}</span>
                    <span className="text-[10px] font-mono text-slate-400">{t.date}</span>
                  </div>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5 leading-snug">{t.detail}</p>
                </div>
              );
            })}
          </div>
        </div>

        {/* Follow-ups */}
        <div className="space-y-2.5">
          <h3 className="text-xs font-bold text-slate-800 dark:text-slate-100 flex items-center gap-1.5">
            <CalendarClock className="w-4 h-4 text-amber-500" /> Follow-up Schedule
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {p.followUps.map((f, i) => (
              <div key={i} className={cn('surface-card p-3.5', !f.completed && 'border-amber-300/60 dark:border-amber-600/50')}>
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono font-bold text-slate-600 dark:text-slate-300">{f.date}</span>
                  <span className={cn('text-[9px] font-bold px-1.5 py-0.5 rounded-full border', f.completed ? 'text-emerald-500 border-emerald-500/30 bg-emerald-500/10' : 'text-amber-500 border-amber-500/30 bg-amber-500/10')}>
                    {f.completed ? 'Completed' : 'Pending'}
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1.5 leading-snug">{f.note}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
