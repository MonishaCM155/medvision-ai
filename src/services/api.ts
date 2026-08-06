import {
  AuthSession,
  AuthUser,
  DashboardStats,
  DatasetInfo,
  EngineStatus,
  HubModel,
  PatientDetail,
  PatientRecord,
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

  login: (name: string): Promise<AuthSession> =>
    fetchJson<AuthSession>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ name }),
    }),

  loginWithPassword: (email: string, password: string): Promise<AuthSession> =>
    fetchJson<AuthSession>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),

  getMe: (token: string): Promise<{ success: boolean; user: AuthUser }> =>
    fetchJson<{ success: boolean; user: AuthUser }>('/api/auth/me', {
      headers: { Authorization: `Bearer ${token}` },
    }),

  logout: (): Promise<{ success: boolean }> =>
    fetchJson<{ success: boolean }>('/api/auth/logout', { method: 'POST' }),
};
