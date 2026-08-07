/**
 * MedVision AI Types
 */

export interface DiseaseProbability {
  disease: string;
  probability: number; // 0.0 to 1.0
  severityContribution: number; // weight
  category: 'lung_opacity' | 'infection' | 'structural' | 'pleural' | 'normal';
  description: string;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  confidence: number;
}

export interface GradCamRegion {
  name: string;
  intensity: number; // 0-100%
  bbox?: BoundingBox;
  interpretation: string;
}

export interface RadiologyReport {
  patientId: string;
  patientAge?: number;
  patientSex?: 'M' | 'F' | 'Other';
  studyDate: string;
  indication: string;
  technique: string;
  findings: string[];
  impression: string;
  recommendations: string[];
  disclaimer: string;
}

export type SeverityLevel = 'Low' | 'Moderate' | 'High' | 'Critical';

export interface PredictionResult {
  id: string;
  timestamp: string;
  imageName: string;
  originalImageUrl: string;
  heatmapOverlayUrl: string;
  processedImageUrl?: string;
  claheApplied: boolean;
  noiseRemovalApplied: boolean;
  modelUsed: string;
  inferenceTimeMs: number;
  diseases: DiseaseProbability[];
  topDiagnosis: string;
  topConfidence: number;
  severity: SeverityLevel;
  severityScore: number; // 0 - 100
  gradCamRegions: GradCamRegion[];
  report: RadiologyReport;
  keyMetrics: {
    snr: number;
    resolution: string;
    meanIntensity: number;
    contrastRatio: number;
  };
  /** AI safety pipeline metadata (additive) */
  calibration?: CalibrationInfo;
  uncertainty?: UncertaintyInfo;
  typeCheck?: ImageTypeResult;
  quality?: { score: number; threshold?: number };
  ood?: { score: number; verdict: 'in' | 'borderline' | 'out'; method: string } | null;
  /** Authoritative provenance — who validated and whether inference actually ran. */
  workflow?: 'upload' | 'sample';
  validationSource?: 'fastapi' | 'sample-demo' | 'demo' | string;
  predictionGenerated?: boolean;
  reportAllowed?: boolean;
  engine?: {
    engineMode?: string;
    source?: string;
    modelName?: string;
    modelVersion?: string;
    weightsLoaded?: boolean;
    predictionSource?: string;
    device?: string;
    proxied?: boolean;
    reason?: string;
  };
}

export interface ModelMetadata {
  id: string;
  name: string;
  architecture: string;
  parameters: string;
  auroc: number;
  f1Score: number;
  accuracy: number;
  latencyFp32Ms: number;
  latencyFp16Ms: number;
  latencyOnnxMs: number;
  flopsGiga: number;
  description: string;
  recommendedFor: string;
}

export interface HistoryItem extends PredictionResult {
  notes?: string;
  isBookmarked?: boolean;
}

export interface BatchPredictionItem {
  id: string;
  fileName: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress: number;
  result?: PredictionResult;
  error?: string;
}

export interface SystemStats {
  totalPredictions: number;
  avgInferenceMs: number;
  severityDistribution: Record<SeverityLevel, number>;
  topDiseasesCount: Record<string, number>;
  modelsUsageCount: Record<string, number>;
}

/* ------------------------------------------------------------------ */
/* Enterprise additions (additive only)                                 */
/* ------------------------------------------------------------------ */

export interface SessionUser {
  name: string;
  initials: string;
  department: string;
  lastActive: string;
}

export interface AppNotification {
  id: string;
  kind: 'success' | 'info' | 'warning' | 'critical';
  title: string;
  body: string;
  time: string;
  read: boolean;
}

export interface ActivityEvent {
  id: string;
  type: 'scan' | 'report' | 'model' | 'system' | 'patient' | 'training';
  actor: string;
  action: string;
  detail: string;
  time: string;
  severity?: SeverityLevel;
}

export interface DashboardStats {
  aiStatus: 'online' | 'degraded' | 'offline';
  todayScans: number;
  pendingReports: number;
  emergencyCases: number;
  totalPatients: number;
  aiAccuracy: number;
  avgLatencyMs: number;
  activeModels: number;
  queuedPredictions: number;
  weeklyTrend: { day: string; scans: number; reports: number; aiAccuracy: number }[];
  monthlyTrend: { month: string; scans: number }[];
  diseaseStats: { disease: string; count: number; color: string }[];
  severityDistribution: { name: SeverityLevel; value: number; color: string }[];
  systemHealth: { component: string; usage: number; status: 'healthy' | 'warning' | 'critical' }[];
  queue: { id: string; patient: string; study: string; model: string; etaSec: number; progress: number }[];
}

export interface PatientRecord {
  id: string;
  name: string;
  age: number;
  sex: 'M' | 'F' | 'Other';
  bloodGroup: string;
  phone: string;
  email: string;
  department: string;
  status: 'Admitted' | 'Outpatient' | 'ICU' | 'Discharged' | 'Critical';
  admissionDate: string;
  lastVisit: string;
  medicalHistory: string[];
  allergies: string[];
  medications: string[];
  topDiagnosis?: string;
  lastSeverity?: SeverityLevel;
  riskScore: number; // 0-100
  avatarHue: number;
}

export interface PatientVisit {
  id: string;
  date: string;
  type: 'X-Ray' | 'CT' | 'Consultation' | 'Lab' | 'Follow-up';
  findings: string;
  severity: SeverityLevel;
  physician: string;
  reports: string[];
}

export interface PatientDetail extends PatientRecord {
  timeline: { date: string; title: string; detail: string; kind: 'visit' | 'report' | 'lab' | 'medication' }[];
  visits: PatientVisit[];
  previousReports: string[];
  followUps: { date: string; note: string; completed: boolean }[];
}

export type ModelTask = 'Classification' | 'Detection' | 'Segmentation' | 'Regression' | 'Traditional ML' | 'Ensemble';

export interface HubModel {
  id: string;
  name: string;
  family: string;
  task: ModelTask;
  framework: string;
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  sensitivity: number;
  f1: number;
  auroc: number;
  parameters: string;
  flops: string;
  size: string;
  latencyMs: number;
  color: string;
  description: string;
  bestFor: string;
  deployed: boolean;
  /** Data provenance: published benchmark vs demo estimate — no fake metrics. */
  source: 'published' | 'estimated' | 'synthetic';
  reference?: string;
}

export interface DatasetInfo {
  id: string;
  name: string;
  task: 'Classification' | 'Detection' | 'Segmentation' | 'Multi-label';
  images: number;
  labels: number;
  classes: string[];
  sizeGb: string;
  missingLabels: number;
  duplicates: number;
  trainSplit: number;
  valSplit: number;
  testSplit: number;
  source: string;
  format: string;
  description: string;
  color: string;
}

export interface TrainingRun {
  id: string;
  name: string;
  model: string;
  dataset: string;
  status: 'RUNNING' | 'FINISHED' | 'FAILED' | 'PAUSED' | 'QUEUED';
  epoch: number;
  totalEpochs: number;
  batchSize: number;
  lr: string;
  optimizer: string;
  scheduler: string;
  loss: number;
  valLoss: number;
  valAuroc: number;
  gpuUtil: number;
  startedAt: string;
  duration: string;
}

export interface ExplainabilityMethod {
  id: string;
  name: string;
  description: string;
  filter: string; // CSS filter simulating the method rendering
  accent: string;
}

export interface EngineStatus {
  status: 'ready' | 'offline';
  source: string; // 'pytorch-checkpoint' | 'pytorch-backbone' | 'demo'
  device: string;
  checkedAt?: number;
  baseUrl?: string;
  note?: string;
}

export interface CalibrationInfo {
  rawTopConfidence: number;
  calibratedTopConfidence: number;
  temperature: number;
  applied: boolean;
  method: string;
}

export interface UncertaintyInfo {
  score: number;
  level: 'low' | 'moderate' | 'high';
  method: string;
  samples?: number;
}

export interface ImageTypeResult {
  predicted: string;
  confidences: Record<string, number>;
  method: string;
}

export interface SafetyReport {
  passed: boolean;
  source?: string;
  proxied?: boolean;
  type?: ImageTypeResult;
  quality?: { score: number; threshold?: number };
  ood?: { score: number; verdict: 'in' | 'borderline' | 'out'; method: string } | null;
  calibration?: { temperature?: number };
  note?: string;
}

export interface MonitoringSnapshot {
  service: string;
  version: string;
  status: string;
  uptimeSec: number;
  requests: number;
  requestsPerMinute: number;
  errors: number;
  rejectedImages: number;
  predictions: number;
  avgInferenceMs: number | null;
  engine: { status: 'ready' | 'offline'; source: string; device: string };
  modelVersion: string;
  storage: string;
  memory: { rssMb: number; heapUsedMb: number; heapTotalMb: number };
  cpu: { user: number; system: number };
  timestamp: string;
}
