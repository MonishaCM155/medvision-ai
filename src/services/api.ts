import {
  DashboardStats,
  DatasetInfo,
  EngineStatus,
  HubModel,
  MonitoringSnapshot,
  PatientDetail,
  PatientRecord,
  SafetyReport,
  TrainingRun,
} from '../types';
import {
  DASHBOARD_STATS,
  DATASETS,
  HUB_MODELS,
  PATIENTS,
  TRAINING_RUNS,
  getPatientDetail,
} from '../data/mockEnterprise';

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return (await res.json()) as T;
}

export const api = {
  getDashboardStats: (): Promise<DashboardStats> =>
    fetchJson<DashboardStats>('/api/dashboard').catch(() => DASHBOARD_STATS),

  getPatients: (): Promise<PatientRecord[]> =>
    fetchJson<PatientRecord[]>('/api/patients').catch(() => PATIENTS),

  getPatient: (id: string): Promise<PatientDetail> =>
    fetchJson<PatientDetail>(`/api/patients/${id}`).catch(() => {
      const detail = getPatientDetail(id);
      if (!detail) throw new Error('Patient not found');
      return detail;
    }),

  getModelHub: (): Promise<HubModel[]> =>
    fetchJson<HubModel[]>('/api/models-hub').catch(() => HUB_MODELS),

  getDatasets: (): Promise<DatasetInfo[]> =>
    fetchJson<DatasetInfo[]>('/api/datasets').catch(() => DATASETS),

  getTrainingRuns: (): Promise<TrainingRun[]> =>
    fetchJson<TrainingRun[]>('/api/training/runs').catch(() => TRAINING_RUNS),

  getEngineStatus: (): Promise<EngineStatus> => fetchJson<EngineStatus>('/api/engine'),

  /** AI Safety Gate — image-type classification + OOD + quality scoring. */
  validateImage: (payload: {
    imageName?: string;
    imageData?: string;
    clientValidation?: { passed: boolean; score: number };
  }): Promise<SafetyReport> =>
    fetchJson<SafetyReport>('/api/validate', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  /**
   * Genuine similar-case retrieval: DenseNet-121 feature embeddings + cosine
   * similarity (FastAPI engine). 503 when the engine is offline — callers
   * fall back to clearly-labelled demo similarities.
   */
  getSimilarCases: (queryImage: string, references: { id: string; title: string; label?: string; imageData: string }[]): Promise<{
    method: string;
    embedding: string;
    device?: string;
    disclaimer: string;
    cases: { case_id: string; title?: string; label?: string; similarity: number }[];
  }> =>
    fetchJson('/api/similar-cases', {
      method: 'POST',
      body: JSON.stringify({ queryImage, references }),
    }),

  /** Live operational telemetry (uptime, memory, engine, rejections…). */
  getMonitoring: (): Promise<MonitoringSnapshot> =>
    fetchJson<MonitoringSnapshot>('/api/monitoring').catch(() => ({
      service: 'MedVision AI Server',
      version: '—',
      status: 'offline',
      uptimeSec: 0,
      requests: 0,
      requestsPerMinute: 0,
      errors: 0,
      rejectedImages: 0,
      predictions: 0,
      avgInferenceMs: null,
      engine: { status: 'offline', source: 'unknown', device: 'unknown' },
      modelVersion: '—',
      storage: '—',
      memory: { rssMb: 0, heapUsedMb: 0, heapTotalMb: 0 },
      cpu: { user: 0, system: 0 },
      timestamp: new Date().toISOString(),
    })),

};
