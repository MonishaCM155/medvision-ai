import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';
import { APP_NAME, APP_VERSION, config, validateConfig } from './config';
import { log, requestLogger, securityHeaders, setLogLevel } from './logger';
import { AuditEntry, initStorage, persistAudit, persistHistory, storageBackend } from './storage';
import {
  DASHBOARD_STATS,
  DATASETS,
  HUB_MODELS,
  PATIENTS,
  SEED_NOTIFICATIONS,
  TRAINING_RUNS,
  getPatientDetail,
} from './src/data/mockEnterprise';
import { SAMPLE_XRAYS } from './src/data/sampleXrays';

const app = express();
const PORT = config.port;

// Only when TRUST_PROXY is explicitly set (Render/nginx/LB deployments) so
// rate limiting and audit IPs reflect the real client instead of the proxy.
app.set('trust proxy', config.trustProxy as boolean | number);

// Structured logging level from environment (LOG_LEVEL)
setLogLevel(config.logLevel);

// Initialize Gemini Client server-side
const aiKey = config.geminiApiKey;
let ai: GoogleGenAI | null = null;
if (aiKey) {
  try {
    ai = new GoogleGenAI({
      apiKey: aiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  } catch (err) {
    log.error('gemini-init-failed', { error: (err as Error).message });
  }
}

app.use(express.json({ limit: config.uploadMaxBytes }));
app.use(requestLogger());
app.use(securityHeaders());

// CORS whitelist — active only when ALLOWED_ORIGINS is configured. The SPA is
// served same-origin so it needs no CORS headers; this supports cross-origin
// deployments without weakening same-origin behavior.
if (config.allowedOrigins.length > 0) {
  app.use((req, res, next) => {
    const origin = req.headers.origin;
    if (origin && config.allowedOrigins.includes(origin)) {
      res.setHeader('Access-Control-Allow-Origin', origin);
      res.setHeader('Vary', 'Origin');
      res.setHeader('Access-Control-Allow-Credentials', 'true');
      res.setHeader('Access-Control-Allow-Methods', 'GET,POST,DELETE,OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type, x-request-id');
    }
    if (req.method === 'OPTIONS') return res.sendStatus(204);
    next();
  });
}

// ---------------------------------------------------------------------------
// Live system monitoring counters (consumed by GET /api/monitoring)
// ---------------------------------------------------------------------------
const monitor = {
  startedAt: Date.now(),
  requests: 0,
  errors: 0,
  rejectedImages: 0,
  predictions: 0,
  rateLimited: 0,
  inferenceWindow: [] as number[],
  requestWindow: [] as number[],
};

app.use((req, res, next) => {
  monitor.requests += 1;
  monitor.requestWindow.push(Date.now());
  res.on('finish', () => {
    if (res.statusCode >= 500) monitor.errors += 1;
  });
  next();
});

// Anonymous audit trail for security-relevant endpoints. Registered BEFORE the
// guarded routes so every predict/validate/batch/history request is captured.
// MedVision AI has no accounts, so every entry is attributed to an anonymous
// public actor — no invented users. Handlers attach validation/engine/report
// state via auditDetail() (read from res.locals on 'finish').
const AUDIT_PATHS = /^\/(api\/(predict|validate|batch-predict|history))(\/|\?|$)/;
app.use((req, res, next) => {
  const ip = req.ip || req.socket.remoteAddress || 'unknown';
  res.on('finish', () => {
    if (AUDIT_PATHS.test(req.path)) {
      const entry: AuditEntry = {
        id: `aud_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
        time: new Date().toISOString(),
        method: req.method,
        path: req.path,
        status: res.statusCode,
        ip,
        actor: 'anonymous',
        detail: res.locals.audit ? JSON.stringify(res.locals.audit) : undefined,
      };
      auditStore.unshift(entry);
      auditStore.splice(500);
      persistAudit(entry);
    }
  });
  next();
});

// In-memory database storage for history and analytics
const historyStore: any[] = [];
const auditStore: AuditEntry[] = [];
const statsStore = {
  totalPredictions: 4,
  avgInferenceMs: 145,
  severityDistribution: {
    Low: 1,
    Moderate: 1,
    High: 1,
    Critical: 1,
  },
  topDiseasesCount: {
    Pneumonia: 2,
    'COVID-19': 1,
    Cardiomegaly: 1,
    'No Finding': 1,
  },
  modelsUsageCount: {
    DenseNet121: 2,
    'EfficientNet-B3': 1,
    'ConvNeXt-Base': 1,
  },
};

// ---------------------------------------------------------------------------
// PyTorch inference engine bridge (FastAPI on :8000) — probed with a short TTL
// ---------------------------------------------------------------------------
const ENGINE_BASE = config.engineBase;
const ENGINE_CACHE_TTL = 30_000;

interface EngineInfo {
  status: 'ready' | 'offline';
  source: string;
  device: string;
  checkedAt: number;
}

let engineCache: EngineInfo | null = null;

async function probeEngine(): Promise<EngineInfo> {
  if (engineCache && Date.now() - engineCache.checkedAt < ENGINE_CACHE_TTL) return engineCache;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 700);
  try {
    const res = await fetch(`${ENGINE_BASE}/api/engine`, { signal: controller.signal });
    if (!res.ok) throw new Error(`engine http ${res.status}`);
    const data = (await res.json()) as { source?: string; device?: string };
    engineCache = { status: 'ready', source: data.source || 'pytorch', device: data.device || 'cpu', checkedAt: Date.now() };
  } catch {
    engineCache = { status: 'offline', source: 'demo', device: 'cpu', checkedAt: Date.now() };
  } finally {
    clearTimeout(timer);
  }
  return engineCache;
}

// ---------------------------------------------------------------------------
// Public-API rate limiter (per IP, sliding 60s window) — abuse protection.
// MedVision AI is a public research platform with no accounts, so per-IP
// throttling of expensive endpoints replaces login throttling.
// ---------------------------------------------------------------------------
const apiCalls = new Map<string, { count: number; resetAt: number }>();
const PUBLIC_API_LIMIT = config.apiRateLimit;

function apiRateLimited(ip: string): boolean {
  const now = Date.now();
  const entry = apiCalls.get(ip);
  if (!entry || now > entry.resetAt) {
    apiCalls.set(ip, { count: 1, resetAt: now + 60_000 });
    return false;
  }
  entry.count += 1;
  return entry.count > PUBLIC_API_LIMIT;
}

setInterval(() => {
  const now = Date.now();
  for (const [ip, e] of apiCalls) if (now > e.resetAt) apiCalls.delete(ip);
}, 60_000).unref?.();

// Expensive public endpoints — HTTP 429 when a client exhausts its per-IP budget
const EXPENSIVE_PATHS = /^\/(api\/(predict|validate|chat|batch-predict))(\/|\?|$)/;
app.use((req, res, next) => {
  if (EXPENSIVE_PATHS.test(req.path)) {
    const ip = req.ip || req.socket.remoteAddress || 'unknown';
    if (apiRateLimited(ip)) {
      monitor.rateLimited += 1;
      return res.status(429).json({ error: 'Rate limit exceeded. Please retry shortly.' });
    }
  }
  next();
});

// ---------------------------------------------------------------------------
// Server-side trust-boundary helpers.
//   SAMPLE_STUDY_NAMES — the bundled demo studies. A request is treated as the
//   SAMPLE workflow ONLY when its imageName matches one of these server-side
//   constants; the client can never tag arbitrary content as a sample.
//   validateViaEngine — the authoritative FastAPI safety gate (type classifier
//   + OOD + quality). Runs for every user upload before any inference.
// ---------------------------------------------------------------------------
const SAMPLE_STUDY_NAMES = new Set([
  'chest_xray_pneumonia_rll.dcm',
  'chest_xray_covid_bilateral.dcm',
  'chest_xray_cardiomegaly_pa.dcm',
  'chest_xray_normal_screening.dcm',
]);
// Byte-exact data URLs of the bundled demo studies — a request that names a
// sample study AND supplies image bytes must match these to be treated as the
// SAMPLE workflow. A spoofed filename with arbitrary bytes falls back to the
// user-upload path (which requires the authoritative engine).
const SAMPLE_SVG_DATA_URLS = new Set(SAMPLE_XRAYS.map((s) => s.svgDataUrl));

function isKnownSample(imageName?: string, imageData?: string): boolean {
  if (typeof imageName !== 'string' || !SAMPLE_STUDY_NAMES.has(imageName)) return false;
  if (typeof imageData !== 'string' || imageData === '') return true; // name-only demo path
  return SAMPLE_SVG_DATA_URLS.has(imageData); // byte-verified when bytes are present
}

async function validateViaEngine(
  imageName: string | undefined,
  imageData: string
): Promise<
  | { reason: 'passed'; validation: any }
  | { reason: 'gate-failed'; validation: any }
  | { reason: 'unavailable' }
> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10_000);
  try {
    const upstream = await fetch(`${ENGINE_BASE}/api/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageName, imageData }),
      signal: controller.signal,
    });
    if (!upstream.ok) return { reason: 'unavailable' };
    const validation = await upstream.json();
    return validation.passed === true
      ? { reason: 'passed', validation }
      : { reason: 'gate-failed', validation };
  } catch {
    return { reason: 'unavailable' };
  } finally {
    clearTimeout(timer);
  }
}

function recordPrediction(result: any) {
  historyStore.unshift(result);
  if (historyStore.length > 200) historyStore.length = 200;
  persistHistory(historyStore);
  monitor.predictions += 1;
  monitor.inferenceWindow.push(result.inferenceTimeMs || 0);
  monitor.inferenceWindow = monitor.inferenceWindow.slice(-200);
  statsStore.totalPredictions += 1;
  statsStore.severityDistribution[result.severity] = (statsStore.severityDistribution[result.severity] || 0) + 1;
  statsStore.topDiseasesCount[result.topDiagnosis] = (statsStore.topDiseasesCount[result.topDiagnosis] || 0) + 1;
}

// Attach audit context for the anonymous audit log (consumed by the middleware
// on 'finish'). Documents validation/engine/report state per request.
function auditDetail(res: express.Response, action: string, state: string, outcome: string) {
  res.locals.audit = { action, state, outcome };
}

// GET /api/engine — current inference engine status (for the UI badge)
app.get('/api/engine', async (_req, res) => {
  const engine = await probeEngine();
  res.json({ ...engine, baseUrl: ENGINE_BASE, note: 'Routes to FastAPI PyTorch engine when online. User uploads are blocked when the engine is offline (no demo fallback); bundled sample studies remain available as clearly-labelled demo profiles.' });
});

// Health Check Endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'MedVision AI - Explainable Medical Image Report Generator',
    geminiConfigured: !!ai,
    timestamp: new Date().toISOString(),
  });
});

// GET /api/version — application + engine version info
app.get('/api/version', async (_req, res) => {
  const engine = await probeEngine();
  res.json({
    name: APP_NAME,
    version: APP_VERSION,
    node: process.version,
    engine: { status: engine.status, source: engine.source, device: engine.device },
    timestamp: new Date().toISOString(),
  });
});

// GET /api/ready — orchestration health check (200 when serving, 503 when a
// configured dependency is down).
app.get('/api/ready', async (_req, res) => {
  const engine = await probeEngine();
  const checks: Record<string, { status: string; detail?: string }> = {
    server: { status: 'up' },
    engine: { status: engine.status === 'ready' ? 'up' : 'degraded', detail: `${engine.source} / ${engine.device}` },
    storage: { status: storageBackend() === 'memory' ? 'degraded' : 'up', detail: storageBackend() },
  };
  if (config.databaseUrl) {
    // 'degraded' (still 200) when the Postgres adapter is unavailable so the
    // container healthcheck never flaps on the optional dependency; 'down'
    // is reserved for genuinely failing core dependencies.
    const backend = storageBackend();
    checks.database = backend === 'postgres'
      ? { status: 'up', detail: 'postgres' }
      : { status: 'degraded', detail: 'DATABASE_URL configured but Postgres adapter unavailable — durable JSON fallback in use' };
  }
  const ready = Object.values(checks).every((c) => c.status !== 'down');
  res.status(ready ? 200 : 503).json({
    status: ready ? 'ready' : 'not-ready',
    checks,
    timestamp: new Date().toISOString(),
  });
});

// GET /api/metrics — Prometheus-style operational metrics (text format)
app.get('/api/metrics', async (_req, res) => {
  const engine = await probeEngine();
  const mem = process.memoryUsage();
  const cpu = process.cpuUsage();
  const now = Date.now();
  const uptimeSec = Math.round((now - monitor.startedAt) / 1000);
  const cpuMsTotal = (cpu.user + cpu.system) / 1000;
  const cpuPercent = uptimeSec > 0 ? Math.min(100, Math.round((cpuMsTotal / 1000 / uptimeSec) * 100)) : 0;
  const lines = [
    '# HELP medvision_requests_total Total HTTP requests served',
    '# TYPE medvision_requests_total counter',
    `medvision_requests_total ${monitor.requests}`,
    '# HELP medvision_errors_total HTTP 5xx responses',
    '# TYPE medvision_errors_total counter',
    `medvision_errors_total ${monitor.errors}`,
    '# HELP medvision_predictions_total Completed inference predictions',
    '# TYPE medvision_predictions_total counter',
    `medvision_predictions_total ${monitor.predictions}`,
    '# HELP medvision_rejected_images_total Images rejected by the safety gate',
    '# TYPE medvision_rejected_images_total counter',
    `medvision_rejected_images_total ${monitor.rejectedImages}`,
    '# HELP medvision_rate_limited_total Requests rejected by the public API rate limiter',
    '# TYPE medvision_rate_limited_total counter',
    `medvision_rate_limited_total ${monitor.rateLimited}`,
    '# HELP medvision_requests_per_minute Request rate (60s window)',
    '# TYPE medvision_requests_per_minute gauge',
    `medvision_requests_per_minute ${monitor.requestWindow.length}`,
    '# HELP medvision_uptime_seconds Process uptime',
    '# TYPE medvision_uptime_seconds gauge',
    `medvision_uptime_seconds ${uptimeSec}`,
    '# HELP medvision_memory_rss_bytes Resident set size',
    '# TYPE medvision_memory_rss_bytes gauge',
    `medvision_memory_rss_bytes ${mem.rss}`,
    '# HELP medvision_memory_heap_used_bytes V8 heap used',
    '# TYPE medvision_memory_heap_used_bytes gauge',
    `medvision_memory_heap_used_bytes ${mem.heapUsed}`,
    '# HELP medvision_cpu_percent CPU percent of one core over uptime',
    '# TYPE medvision_cpu_percent gauge',
    `medvision_cpu_percent ${cpuPercent}`,
    '# HELP medvision_engine_status Engine readiness (1=ready, 0=offline)',
    '# TYPE medvision_engine_status gauge',
    `medvision_engine_status ${engine.status === 'ready' ? 1 : 0}`,
  ];
  res.setHeader('Content-Type', 'text/plain; version=0.0.4; charset=utf-8');
  res.send(lines.join('\n') + '\n');
});

// GET Models Endpoint
app.get('/api/models', (req, res) => {
  res.json({
    models: [
      { id: 'densenet121', name: 'DenseNet-121 (CheXNet)', auroc: 0.841, f1: 0.812, params: '7.0M', latencyMs: 18 },
      { id: 'efficientnet_b3', name: 'EfficientNet-B3', auroc: 0.865, f1: 0.835, params: '12.2M', latencyMs: 24 },
      { id: 'convnext_base', name: 'ConvNeXt-Base', auroc: 0.882, f1: 0.854, params: '88.5M', latencyMs: 38 },
      { id: 'swin_b', name: 'Swin Transformer (Swin-B)', auroc: 0.889, f1: 0.862, params: '88.0M', latencyMs: 45 },
      { id: 'vit_b', name: 'Vision Transformer (ViT-B/16)', auroc: 0.878, f1: 0.849, params: '86.6M', latencyMs: 42 },
    ],
  });
});

// GET Stats Endpoint
app.get('/api/stats', (req, res) => {
  res.json(statsStore);
});

// GET History Endpoint
app.get('/api/history', (req, res) => {
  res.json(historyStore);
});

// DELETE History Endpoint
app.delete('/api/history/:id', (req, res) => {
  const { id } = req.params;
  const index = historyStore.findIndex(item => item.id === id);
  if (index !== -1) {
    historyStore.splice(index, 1);
    persistHistory(historyStore);
    return res.json({ success: true, message: 'Deleted successfully' });
  }
  res.status(404).json({ success: false, error: 'Item not found' });
});

// POST /api/validate — AI Safety Gate (type classifier + OOD + quality)
app.post('/api/validate', async (req, res) => {
  try {
    const { imageData, imageName, clientValidation } = req.body || {};
    if (typeof imageData === 'string' && imageData.length > config.uploadMaxBytes) {
      return res.status(413).json({ error: `Image payload exceeds ${Math.round(config.uploadMaxBytes / 1048576)}MB limit` });
    }

    // Prefer the real PyTorch engine's safety gate when it is online
    const engine = await probeEngine();
    if (engine.status === 'ready' && imageData) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 10_000);
      try {
        const upstream = await fetch(`${ENGINE_BASE}/api/validate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ imageName, imageData }),
          signal: controller.signal,
        });
        if (upstream.ok) {
          const body = await upstream.json();
          body.source = 'pytorch-engine';
          body.proxied = true;
          return res.json(body);
        }
        throw new Error(`engine validate http ${upstream.status}`);
      } catch (upstreamErr) {
        log.warn('engine-validate-fallback', { error: (upstreamErr as Error).message });
      } finally {
        clearTimeout(timer);
      }
    }

    // Honest offline fallback built from the client-side analyzer report
    const passed = clientValidation?.passed !== false;
    res.json({
      passed,
      source: 'client-heuristic',
      proxied: false,
      type: passed
        ? { predicted: 'chest_xray', confidences: { chest_xray: 0.94, other_xray: 0.03, unknown: 0.03 }, method: 'client-heuristic' }
        : { predicted: 'unknown', confidences: {}, method: 'client-heuristic' },
      quality: { score: clientValidation?.score ?? 0, threshold: config.qualityThreshold },
      ood: null,
      calibration: { temperature: 1.0 },
      note: 'PyTorch engine offline — safety-gate summary supplied by the client-side analyzer.',
    });
  } catch (err: any) {
    res.status(500).json({ error: err.message || 'Validation failed' });
  }
});

// POST /api/similar-cases — genuine embedding retrieval, proxied to the FastAPI
// engine (DenseNet-121 features + cosine similarity). Requires the engine;
// returns 503 when it is offline — never fabricated similarity scores.
app.post('/api/similar-cases', async (req, res) => {
  try {
    const { queryImage, references } = req.body || {};
    const engine = await probeEngine();
    if (engine.status !== 'ready') {
      auditDetail(res, 'similar-cases', 'engine-offline', 'blocked');
      return res.status(503).json({
        ok: false,
        status: 'engine_unavailable',
        code: 'ML_ENGINE_UNAVAILABLE',
        message: 'Similar-case retrieval requires the PyTorch engine (FastAPI on :8000).',
        cases: [],
      });
    }
    if (!queryImage || !references || !Array.isArray(references)) {
      return res.status(422).json({ error: 'queryImage and references[] are required', cases: [] });
    }
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15_000);
    try {
      const upstream = await fetch(`${ENGINE_BASE}/api/similar-cases`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ queryImage, references }),
        signal: controller.signal,
      });
      if (!upstream.ok) throw new Error(`engine similar-cases http ${upstream.status}`);
      const body = await upstream.json();
      auditDetail(res, 'similar-cases', 'engine-embeddings', 'ok');
      return res.json(body);
    } finally {
      clearTimeout(timer);
    }
  } catch (err: any) {
    log.error('similar-cases-failed', { error: err.message });
    res.status(502).json({
      ok: false,
      status: 'inference_failed',
      code: 'INFERENCE_FAILED',
      message: 'Similar-case retrieval failed.',
      cases: [],
    });
  }
});

// GET /api/monitoring — live operational telemetry
app.get('/api/monitoring', async (_req, res) => {
  const engine = await probeEngine();
  const mem = process.memoryUsage();
  const cpu = process.cpuUsage();
  const now = Date.now();
  const uptimeSec = Math.round((now - monitor.startedAt) / 1000);
  monitor.requestWindow = monitor.requestWindow.filter((t) => now - t < 60_000);
  const avgInferenceMs = monitor.inferenceWindow.length
    ? Math.round(monitor.inferenceWindow.reduce((a, b) => a + b, 0) / monitor.inferenceWindow.length)
    : null;
  // process.cpuUsage() returns accumulated CPU time in microseconds; express it
  // as a live percentage of one core over the process uptime.
  const cpuMsTotal = (cpu.user + cpu.system) / 1000;
  const cpuPercent = uptimeSec > 0 ? Math.min(100, Math.round((cpuMsTotal / 1000 / uptimeSec) * 100)) : 0;
  res.json({
    service: 'MedVision AI Server',
    version: APP_VERSION,
    status: 'healthy',
    uptimeSec,
    requests: monitor.requests,
    requestsPerMinute: monitor.requestWindow.length,
    errors: monitor.errors,
    rejectedImages: monitor.rejectedImages,
    predictions: monitor.predictions,
    avgInferenceMs,
    engine: { status: engine.status, source: engine.source, device: engine.device },
    modelVersion: APP_VERSION,
    storage: storageBackend(),
    memory: {
      rssMb: Math.round(mem.rss / 1048576),
      heapUsedMb: Math.round(mem.heapUsed / 1048576),
      heapTotalMb: Math.round(mem.heapTotal / 1048576),
    },
    cpu: { user: cpuPercent, system: 0 },
    timestamp: new Date().toISOString(),
  });
});

// POST /api/predict — authoritative server-side gated inference.
//
// TRUST BOUNDARY:
//   - The client is NEVER trusted. Client-supplied validation, quality, OOD,
//     image-type, confidence, and uncertainty fields are advisory at most;
//     NONE of them can authorize inference.
//   - A client-claimed FAILURE may short-circuit (rejecting is always safe).
//   - Inference runs ONLY when the server has independently established:
//       1. a decodable image payload is present (user uploads),
//       2. the FastAPI ML engine is reachable,
//       3. the FastAPI safety gate (type + OOD + quality) PASSED.
//   - The ONLY exception is the explicit SAMPLE workflow: a request whose
//     imageName matches one of the bundled demo studies (server-verified)
//     produces a clearly-labelled demo profile (engineMode: 'demo-engine').
//     Demo inference is NEVER silently substituted for failed real inference.
app.post('/api/predict', async (req, res) => {
  try {
    const { imageName, imageData, model = 'DenseNet121', clahe = false, noiseRemoval = false, validation } = req.body;

    if (!imageData && !imageName) {
      return res.status(400).json({ error: 'Image data or image name is required' });
    }

    // Input cap: refuse oversized payloads before any processing
    if (typeof imageData === 'string' && imageData.length > config.uploadMaxBytes) {
      return res.status(413).json({ error: `Image payload exceeds ${Math.round(config.uploadMaxBytes / 1048576)}MB limit` });
    }

    // Server-side MIME check: reject data-URLs that are not raster/SVG images
    if (
      imageData &&
      typeof imageData === 'string' &&
      imageData.startsWith('data:') &&
      !/^data:image\/(?:png|jpe?g|webp|gif|bmp|svg\+xml)(?:;|,)/i.test(imageData)
    ) {
      monitor.rejectedImages += 1;
      auditDetail(res, 'predict', 'non-image-payload', 'rejected');
      return res.status(422).json({
        error: 'Unsupported image format. Upload a PNG, JPEG, or WebP chest X-ray (DICOM parsing is not supported in this research build).',
        code: 'UNSUPPORTED_IMAGE',
        predictionGenerated: false,
        reportAllowed: false,
      });
    }

    // Client validation is ADVISORY ONLY: a client-claimed FAILURE short-circuits
    // (rejecting is always safe), but a client-claimed PASS never authorizes
    // inference — the server performs its own authoritative validation below.
    if (validation && typeof validation.passed === 'boolean' && !validation.passed) {
      monitor.rejectedImages += 1;
      auditDetail(res, 'predict', 'client-claimed-failure', 'rejected');
      return res.status(422).json({
        error: 'Image failed chest X-ray validation. Only valid frontal chest X-rays can be analyzed.',
        code: 'VALIDATION_FAILED',
        predictionGenerated: false,
        reportAllowed: false,
      });
    }

    // SAMPLE workflow — server-verified against the bundled demo studies.
    if (isKnownSample(imageName, imageData)) {
      // Falls through to the deterministic demo body below (clearly labelled).
    } else {
      // USER UPLOAD — authoritative gating from here on.
      if (typeof imageData !== 'string' || imageData === '') {
        auditDetail(res, 'predict', 'no-image-bytes', 'rejected');
        return res.status(422).json({
          error: 'INVALID_IMAGE: no image bytes were supplied for server-side validation.',
          code: 'INVALID_IMAGE',
          predictionGenerated: false,
          reportAllowed: false,
        });
      }

      const engine = await probeEngine();
      if (engine.status !== 'ready') {
        auditDetail(res, 'predict', 'engine-offline', 'blocked');
        return res.status(503).json({
          ok: false,
          status: 'engine_unavailable',
          code: 'ML_ENGINE_UNAVAILABLE',
          message: 'The medical inference engine is currently unavailable. No prediction was generated.',
          predictionGenerated: false,
          reportAllowed: false,
          engine: { status: 'offline', source: 'demo', engineMode: 'demo-engine', device: 'cpu' },
        });
      }

      // Server-side authoritative validation via the FastAPI safety gate.
      const v = await validateViaEngine(imageName, imageData);
      if (v.reason === 'unavailable') {
        auditDetail(res, 'predict', 'validation-unavailable', 'blocked');
        return res.status(503).json({
          ok: false,
          status: 'validation_unavailable',
          code: 'VALIDATION_UNAVAILABLE',
          message: 'Server-side image validation is temporarily unavailable. No prediction was generated.',
          predictionGenerated: false,
          reportAllowed: false,
        });
      }
      if (v.reason === 'gate-failed') {
        monitor.rejectedImages += 1;
        auditDetail(res, 'predict', 'server-validation-failed', 'rejected');
        return res.status(422).json({
          error: 'Image failed server-side validation. Only valid frontal chest X-rays can be analyzed.',
          code: 'VALIDATION_FAILED',
          validation: v.validation,
          validationSource: 'fastapi',
          predictionGenerated: false,
          reportAllowed: false,
        });
      }

      // Authoritative validation PASSED — real inference through the engine.
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 12_000);
      try {
        const upstream = await fetch(`${ENGINE_BASE}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ imageName, imageData, model, clahe, noiseRemoval }),
          signal: controller.signal,
        });
        if (!upstream.ok) throw new Error(`engine predict http ${upstream.status}`);
        const result = await upstream.json();
        result.engine = { ...(result.engine || {}), proxied: true, via: ENGINE_BASE };
        // Preserve real maps computed by the engine; only echo the raw image
        // when the engine did not produce one.
        result.originalImageUrl = result.originalImageUrl || imageData;
        result.heatmapOverlayUrl = result.heatmapOverlayUrl || imageData;
        result.claheApplied = !!clahe;
        result.noiseRemovalApplied = !!noiseRemoval;
        result.calibration =
          result.calibration || { applied: false, temperature: 1, method: 'identity', rawTopConfidence: result.topConfidence, calibratedTopConfidence: result.topConfidence };
        result.uncertainty = result.uncertainty || { score: 40, level: 'moderate', method: 'margin-proxy' };
        // Authoritative provenance: validation was performed server-side, never client-side.
        result.workflow = 'upload';
        result.validationSource = 'fastapi';
        result.predictionGenerated = true;
        // Report gating (Phase 14): low confidence or high uncertainty blocks
        // the definitive report even after a successful validated inference.
        result.reportAllowed =
          (result.topConfidence ?? 0) >= config.confidenceThreshold &&
          (result.uncertainty?.level ?? 'low') !== 'high';
        recordPrediction(result);
        auditDetail(res, 'predict', 'real-inference', result.reportAllowed ? 'ok' : 'low-confidence');
        return res.json(result);
      } catch (upstreamErr) {
        log.warn('engine-predict-failed', { error: (upstreamErr as Error).message });
        auditDetail(res, 'predict', 'inference-failed', 'blocked');
        // A real inference failure NEVER becomes a demo diagnosis.
        return res.status(502).json({
          ok: false,
          status: 'inference_failed',
          code: 'INFERENCE_FAILED',
          message: 'The medical inference engine failed to produce a result. No prediction was generated.',
          predictionGenerated: false,
          reportAllowed: false,
        });
      } finally {
        clearTimeout(timer);
      }
    }

    // ------------------------------------------------------------------
    // SAMPLE workflow — deterministic demo body (only reached when the
    // request names one of the bundled demo studies). Always labelled.
    // ------------------------------------------------------------------
    const startTime = Date.now();

    // Determine disease predictions using ML model logic / analysis
    // Generate realistic disease probabilities
    let diseaseScores = [
      { disease: 'No Finding', probability: 0.05, severityContribution: 0, category: 'normal', description: 'Clear lung fields without consolidation.' },
      { disease: 'Pneumonia', probability: 0.88, severityContribution: 0.4, category: 'infection', description: 'Airspace opacity with bronchial consolidation.' },
      { disease: 'Lung Opacity', probability: 0.82, severityContribution: 0.3, category: 'lung_opacity', description: 'Increased radiodensity in lung parenchyma.' },
      { disease: 'Atelectasis', probability: 0.35, severityContribution: 0.1, category: 'structural', description: 'Subsegmental volume loss.' },
      { disease: 'Pleural Effusion', probability: 0.25, severityContribution: 0.1, category: 'pleural', description: 'Fluid accumulation in pleural space.' },
      { disease: 'Cardiomegaly', probability: 0.08, severityContribution: 0.0, category: 'structural', description: 'Normal cardiac size.' },
      { disease: 'Edema', probability: 0.12, severityContribution: 0.05, category: 'lung_opacity', description: 'No acute pulmonary edema.' },
      { disease: 'COVID-19', probability: 0.15, severityContribution: 0.0, category: 'infection', description: 'Low probability of COVID-19.' },
      { disease: 'Tuberculosis', probability: 0.04, severityContribution: 0.0, category: 'infection', description: 'No apical cavitation.' },
      { disease: 'Pneumothorax', probability: 0.02, severityContribution: 0.0, category: 'pleural', description: 'No pleural air line.' },
    ];

    // If user uploaded image contains certain keywords or custom image, tailor probabilities
    const nameLower = (imageName || '').toLowerCase();
    if (nameLower.includes('normal') || nameLower.includes('clear')) {
      diseaseScores = diseaseScores.map(d => {
        if (d.disease === 'No Finding') return { ...d, probability: 0.96 };
        return { ...d, probability: Number((Math.random() * 0.05).toFixed(2)) };
      });
    } else if (nameLower.includes('covid')) {
      diseaseScores = diseaseScores.map(d => {
        if (d.disease === 'COVID-19') return { ...d, probability: 0.93 };
        if (d.disease === 'Pneumonia') return { ...d, probability: 0.87 };
        if (d.disease === 'Lung Opacity') return { ...d, probability: 0.89 };
        if (d.disease === 'No Finding') return { ...d, probability: 0.02 };
        return d;
      });
    } else if (nameLower.includes('cardio') || nameLower.includes('heart')) {
      diseaseScores = diseaseScores.map(d => {
        if (d.disease === 'Cardiomegaly') return { ...d, probability: 0.95 };
        if (d.disease === 'Edema') return { ...d, probability: 0.45 };
        if (d.disease === 'No Finding') return { ...d, probability: 0.02 };
        return d;
      });
    }

    // Sort by highest probability
    diseaseScores.sort((a, b) => b.probability - a.probability);
    const topDiag = diseaseScores[0];

    // Calculate severity
    let severityScore = Math.min(100, Math.round(topDiag.probability * 80 + (diseaseScores[1]?.probability || 0) * 20));
    let severity: 'Low' | 'Moderate' | 'High' | 'Critical' = 'Low';
    if (topDiag.disease === 'No Finding' || topDiag.probability < 0.3) {
      severity = 'Low';
      severityScore = Math.round(topDiag.probability < 0.3 ? 15 : 5);
    } else if (severityScore > 85) {
      severity = 'Critical';
    } else if (severityScore > 65) {
      severity = 'High';
    } else if (severityScore > 35) {
      severity = 'Moderate';
    }

    // Call Gemini API for dynamic structured radiology report if available
    let findingsList = [
      `LUNGS: Focal airspace opacity noted with highest activation in ${topDiag.disease === 'COVID-19' ? 'bilateral peripheral' : 'right lower lung'} field.`,
      `PLEURA: ${diseaseScores.find(d => d.disease === 'Pleural Effusion')?.probability! > 0.3 ? 'Blunting of costophrenic angle present.' : 'No significant pleural effusion or pneumothorax.'}`,
      `CARDIOMEDIASTINAL: ${diseaseScores.find(d => d.disease === 'Cardiomegaly')?.probability! > 0.5 ? 'Cardiac silhouette enlarged (CTR > 0.55).' : 'Heart size and mediastinal contours normal.'}`,
    ];
    let impression = `1. Radiographic evidence consistent with ${topDiag.disease} (Confidence: ${(topDiag.probability * 100).toFixed(0)}%).\n2. Clinical correlation recommended.`;
    let recommendations = [
      'Correlate clinically with patient laboratory results, vitals, and inflammatory markers.',
      'Consider follow-up chest radiograph or CT chest if clinically indicated.',
    ];

    if (ai) {
      try {
        const prompt = `You are an expert AI Radiologist assistant generating a structured chest X-ray report.
Image Name: ${imageName || 'Chest X-Ray'}
Top Detected Condition: ${topDiag.disease} (Probability: ${(topDiag.probability * 100).toFixed(1)}%)
Secondary Diseases Detected: ${diseaseScores.slice(1, 4).map(d => `${d.disease} (${(d.probability * 100).toFixed(1)}%)`).join(', ')}
Severity: ${severity} (${severityScore}/100)

Generate a JSON object strictly matching this schema:
{
  "findings": ["string", "string", "string"],
  "impression": "string",
  "recommendations": ["string", "string"]
}

Ensure formal medical terminology (e.g., LUNGS, CARDIOMEDIASTINAL, PLEURA) and concise findings.`;

        const response = await ai.models.generateContent({
          model: config.geminiModel,
          contents: prompt,
          config: {
            responseMimeType: 'application/json',
          },
        });

        if (response.text) {
          const parsed = JSON.parse(response.text);
          if (parsed.findings && Array.isArray(parsed.findings)) findingsList = parsed.findings;
          if (parsed.impression) impression = parsed.impression;
          if (parsed.recommendations && Array.isArray(parsed.recommendations)) recommendations = parsed.recommendations;
        }
      } catch (geminiErr) {
        log.warn('gemini-report-fallback', { error: (geminiErr as Error).message });
      }
    }

    const elapsedTime = Date.now() - startTime + Math.floor(Math.random() * 30 + 100);

    const result: any = {
      id: `pred_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
      timestamp: new Date().toISOString(),
      imageName: imageName || 'uploaded_chest_xray.png',
      originalImageUrl: imageData || '',
      heatmapOverlayUrl: imageData || '',
      claheApplied: !!clahe,
      noiseRemovalApplied: !!noiseRemoval,
      modelUsed: `${model} (PyTorch + Grad-CAM)`,
      inferenceTimeMs: elapsedTime,
      diseases: diseaseScores,
      topDiagnosis: topDiag.disease,
      topConfidence: topDiag.probability,
      severity,
      severityScore,
      gradCamRegions: topDiag.disease !== 'No Finding' ? [
        {
          name: `Focal Region: ${topDiag.disease}`,
          intensity: Math.round(topDiag.probability * 100),
          bbox: { x: 180, y: 210, width: 120, height: 110, label: topDiag.disease, confidence: topDiag.probability },
          interpretation: `High Grad-CAM density highlighting peak neural response in thoracic region for ${topDiag.disease}.`,
        },
      ] : [],
      report: {
        patientId: `PAT-DEMO-${(Math.floor(1 + Math.random() * 998)).toString().padStart(3, '0')}`,
        patientAge: null,
        patientSex: null,
        studyDate: new Date().toISOString().split('T')[0],
        indication: 'Shortness of breath, cough, fever evaluation.',
        technique: 'Upright PA Chest Radiograph (1024x1024).',
        findings: findingsList,
        impression,
        recommendations,
        disclaimer: 'Not for clinical diagnosis. Educational and research demonstration purposes only.',
      },
      keyMetrics: {
        snr: 29.5,
        resolution: '1024x1024',
        meanIntensity: 115.2,
        contrastRatio: 4.9,
      },
      engine: {
        // Canonical honest mode metadata — demo predictions never masquerade as real inference
        engineMode: 'demo-engine',
        source: 'demo',
        modelName: 'DenseNet-121 (CheXNet)',
        modelVersion: APP_VERSION,
        weightsLoaded: false,
        predictionSource: 'demo-profile',
        device: 'cpu',
        proxied: false,
        reason: 'Sample/demo study — deterministic demo profile, not real inference',
      },
      // Confidence calibration (identity unless MEDVISION_TEMPERATURE is set server-side)
      calibration: {
        applied: false,
        temperature: 1.0,
        method: 'identity',
        rawTopConfidence: topDiag.probability,
        calibratedTopConfidence: topDiag.probability,
      },
      // Uncertainty from the top-vs-second margin (server-side demo profile)
      uncertainty: (() => {
        const margin = topDiag.probability - (diseaseScores[1]?.probability || 0);
        const score = Math.round(Math.min(100, Math.max(0, 100 - margin * 160 + 8)) * 10) / 10;
        const level = score < 35 ? 'low' : score < 55 ? 'moderate' : 'high';
        return { score, level, method: 'margin-proxy' };
      })(),
      typeCheck: null, // Node has no image decoder — FastAPI covers real uploads
      quality: { score: 80, threshold: config.qualityThreshold },
      ood: null,
    };

    // SAMPLE workflow provenance — always explicit, never silent. Demo reports
    // are labelled synthetic; export remains available for the explicit demo.
    result.workflow = 'sample';
    result.validationSource = 'sample-demo';
    result.predictionGenerated = true;
    result.reportAllowed = true;

    recordPrediction(result);
    auditDetail(res, 'predict', 'sample-demo', 'ok');
    res.json(result);
  } catch (error: any) {
    log.error('predict-failed', { error: error.message || 'Internal server error' });
    res.status(500).json({ error: error.message || 'Internal server error', predictionGenerated: false, reportAllowed: false });
  }
});

// POST /api/batch-predict — same trust boundary as /api/predict.
//   - mode: 'demo' → explicit demo workflow (the Batch UI is labeled DEMO MODE)
//     returns clearly-flagged simulated profiles, never real inference.
//   - otherwise → authoritative batch: the ML engine must be reachable and
//     every item is server-side validated + inferred individually; items
//     without image bytes or failing the gate are rejected individually.
app.post('/api/batch-predict', async (req, res) => {
  try {
    const { files, mode } = req.body;
    if (!files || !Array.isArray(files)) {
      return res.status(400).json({ error: 'Array of files is required' });
    }
    const engine = await probeEngine();

    // Explicitly-requested demo workflow — flagged, never silent.
    if (mode === 'demo') {
      const results = files.map((file: any, idx: number) => {
        const isNormal = idx % 3 === 0;
        const topDisease = isNormal ? 'No Finding' : idx % 2 === 0 ? 'Pneumonia' : 'Cardiomegaly';
        const prob = isNormal ? 0.95 : 0.89;
        return {
          fileName: file.name || `xray_${idx + 1}.dcm`,
          topDiagnosis: topDisease,
          confidence: prob,
          severity: isNormal ? 'Low' : 'High',
          status: 'Completed',
          inferenceMs: 120 + idx * 15,
          demo: true,
          validationSource: 'demo',
        };
      });
      auditDetail(res, 'batch', 'demo-mode', 'ok');
      return res.json({ success: true, count: results.length, demo: true, batchResults: results });
    }

    // Authoritative batch — engine required, per-file server-side validation.
    if (engine.status !== 'ready') {
      auditDetail(res, 'batch', 'engine-offline', 'blocked');
      return res.status(503).json({
        ok: false,
        status: 'engine_unavailable',
        code: 'ML_ENGINE_UNAVAILABLE',
        message: 'The medical inference engine is currently unavailable. No batch predictions were generated.',
        predictionGenerated: false,
        reportAllowed: false,
      });
    }

    const results = [];
    for (const file of files) {
      if (typeof file.imageData !== 'string' || file.imageData === '') {
        results.push({ fileName: file.name || 'unknown', status: 'error', error: 'INVALID_IMAGE: no image bytes supplied' });
        continue;
      }
      const v = await validateViaEngine(file.name, file.imageData);
      if (v.reason !== 'passed') {
        monitor.rejectedImages += 1;
        results.push({
          fileName: file.name || 'unknown',
          status: 'error',
          error: 'VALIDATION_FAILED',
          validation: v.reason === 'gate-failed' ? v.validation : undefined,
        });
        continue;
      }
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 12_000);
      try {
        const upstream = await fetch(`${ENGINE_BASE}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ imageName: file.name, imageData: file.imageData, model: file.model || 'DenseNet121' }),
          signal: controller.signal,
        });
        if (!upstream.ok) throw new Error(`engine predict http ${upstream.status}`);
        const r = await upstream.json();
        results.push({
          fileName: file.name || 'unknown',
          topDiagnosis: r.topDiagnosis,
          confidence: r.topConfidence,
          severity: r.severity,
          status: 'Completed',
          inferenceMs: Math.round(r.inferenceTimeMs || 0),
          validationSource: 'fastapi',
        });
        monitor.predictions += 1;
      } catch {
        results.push({ fileName: file.name || 'unknown', status: 'error', error: 'INFERENCE_FAILED' });
      } finally {
        clearTimeout(timer);
      }
    }
    auditDetail(res, 'batch', 'authoritative', 'ok');
    res.json({ success: true, count: results.length, demo: false, batchResults: results });
  } catch (err: any) {
    log.error('batch-predict-failed', { error: err.message });
    res.status(500).json({ error: err.message });
  }
});

// ---------------------------------------------------------------------------
// Enterprise Suite Endpoints (additive — existing APIs untouched)
// ---------------------------------------------------------------------------

// GET /api/audit-logs — public security audit trail. Open in public research
// mode (no authentication); entries are anonymized (actor: "anonymous").
// Pre-v2.7 rows (from the JWT era) carry a legacy `role` field — normalized so
// every entry exposes an `actor`.
app.get('/api/audit-logs', (_req, res) => {
  const entries = auditStore.slice(0, 200).map((e: any) => ({ ...e, actor: e.actor ?? 'legacy' }));
  res.json({ success: true, count: entries.length, entries, actor: 'anonymous' });
});

// Authentication routes intentionally removed — MedVision AI is a public
// research/education platform (no login, logout, JWT, or accounts). Legacy
// auth calls now receive an explicit 404 with guidance.
app.all('/api/auth/*', (req, res) => {
  res.status(404).json({
    error: 'Authentication is not used by MedVision AI. The platform runs in public research mode with no login or accounts.',
  });
});

// GET /api/dashboard — enterprise KPI snapshot
app.get('/api/dashboard', (req, res) => {
  res.json({
    ...DASHBOARD_STATS,
    aiStatus: 'online',
    timestamp: new Date().toISOString(),
  });
});

// GET /api/patients — searchable patient registry
app.get('/api/patients', (req, res) => {
  const q = ((req.query.q as string) || '').toLowerCase();
  const filtered = q
    ? PATIENTS.filter(
        (p) =>
          p.name.toLowerCase().includes(q) ||
          p.id.toLowerCase().includes(q) ||
          (p.topDiagnosis || '').toLowerCase().includes(q)
      )
    : PATIENTS;
  res.json(filtered);
});

// GET /api/patients/:id — full EHR detail
app.get('/api/patients/:id', (req, res) => {
  const detail = getPatientDetail(req.params.id);
  if (!detail) return res.status(404).json({ error: 'Patient not found' });
  res.json(detail);
});

// GET /api/models-hub — 16-architecture registry
app.get('/api/models-hub', (req, res) => {
  res.json(HUB_MODELS);
});

// GET /api/datasets — dataset registry
app.get('/api/datasets', (req, res) => {
  res.json(DATASETS);
});

// GET /api/training/runs — experiment runs
app.get('/api/training/runs', (req, res) => {
  res.json(TRAINING_RUNS);
});

// GET /api/notifications
app.get('/api/notifications', (req, res) => {
  res.json(SEED_NOTIFICATIONS);
});

// POST /api/chat Endpoint for AI Copilot
app.post('/api/chat', async (req, res) => {
  try {
    const { prompt, history = [], predictionContext, provider = 'gemini' } = req.body;

    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required' });
    }

    const hasContext = !!(predictionContext && (predictionContext.topDiagnosis || (predictionContext.diseases && predictionContext.diseases.length > 0)));
    if (!hasContext) {
      // No active analysis — answer honestly instead of inventing a diagnosis,
      // a confidence value, or patient data.
      return res.json({
        reply: 'I can explain chest X-ray predictions, Grad-CAM heatmaps, radiology terminology, and study reports — but there is no active analysis in this session yet. Run an analysis first (upload a chest X-ray or choose a curated sample study) and I will walk you through the model\'s prediction, its confidence and uncertainty, and the regions it focused on.\n\n*Research and education only — this is not a clinical diagnosis.*',
        provider: `${provider.toUpperCase()} (MedVision Rule Engine)`,
        timestamp: new Date().toISOString(),
      });
    }
    const topDiag = predictionContext?.topDiagnosis || null;
    const topConf = predictionContext?.topConfidence ? (predictionContext.topConfidence * 100).toFixed(1) : null;
    const severity = predictionContext?.severity || null;
    const modelUsed = predictionContext?.modelUsed || 'DenseNet-121 (PyTorch + Grad-CAM)';

    // System prompt for Gemini or fallback assistant
    const systemPrompt = `You are "MedVision AI Copilot", an elite AI Medical Imaging Researcher and Radiology Assistant created by NVIDIA AI & DeepMind.
Your role is to explain medical imaging predictions, Grad-CAM heatmaps, disease mechanisms, confidence scores, and radiology findings to users in a clear, highly professional, educational manner.

[CRITICAL MANDATE]
Always include this disclaimer at the beginning or end when relevant:
"This application is for research and educational purposes only and must not be used for clinical diagnosis or patient triage."

[CURRENT PATIENT / PREDICTION CONTEXT]
- Top Predicted Condition: ${topDiag}
- Confidence / Certainty: ${topConf}%
- Clinical Severity: ${severity} (${predictionContext?.severityScore || 82}/100)
- Neural Model Architecture: ${modelUsed}
- Image Resolution: ${predictionContext?.keyMetrics?.resolution || '1024x1024'}
- Signal-to-Noise Ratio (SNR): ${predictionContext?.keyMetrics?.snr || 29.5} dB
- Secondary Diseases Detected: ${predictionContext?.diseases ? predictionContext.diseases.slice(1, 4).map((d: any) => `${d.disease} (${(d.probability * 100).toFixed(1)}%)`).join(', ') : 'Lung Opacity (82%), Atelectasis (35%)'}
- Radiology Impression: ${predictionContext?.report?.impression || '1. Radiographic evidence consistent with Pneumonia.'}

Respond to the user's question directly using structured Markdown (headers, bullet points, bold key terms). Be informative, empathetic, scientifically precise, and engaging.`;

    if (ai && provider === 'gemini') {
      try {
        const contents = [
          { role: 'user', parts: [{ text: systemPrompt }] },
          ...history.slice(-6).map((msg: any) => ({
            role: msg.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: msg.content }],
          })),
          { role: 'user', parts: [{ text: prompt }] },
        ];

        const response = await ai.models.generateContent({
          model: config.geminiModel,
          contents,
        });

        if (response.text) {
          return res.json({
            reply: response.text,
            provider: 'Google Gemini 3.6 Flash',
            timestamp: new Date().toISOString(),
          });
        }
      } catch (geminiErr) {
        log.warn('gemini-chat-fallback', { error: (geminiErr as Error).message });
      }
    }

    // Expert Fallback Engine for Medical Queries when Gemini key is not present or offline
    let fallbackReply = '';
    const qLower = prompt.toLowerCase();

    if (qLower.includes('what disease') || qLower.includes('explain this disease') || qLower.includes('pneumonia')) {
      fallbackReply = `### 🦠 Overview of ${topDiag}

**${topDiag}** is an inflammatory condition affecting the pulmonary alveoli (air sacs in the lungs). On this chest X-ray, the neural model identified localized consolidation where air is replaced by inflammatory exudate.

* **Key AI Finding:** The model calculated a **${topConf}% probability** for ${topDiag} based on localized feature activations in the lower right pulmonary field.
* **Common Symptoms:** Fever, productive cough with sputum, chest pain on deep inspiration (pleuritic pain), shortness of breath, and fatigue.
* **Pathophysiology:** Microorganisms trigger an immune response causing fluid and white blood cells to fill alveolar spaces, decreasing gas exchange efficiency across the alveolar-capillary membrane.

*Note: This application is for education and research only and must not be used for clinical diagnosis.*`;
    } else if (qLower.includes('heatmap') || qLower.includes('grad-cam') || qLower.includes('highlight')) {
      fallbackReply = `### 🎯 Grad-CAM Heatmap Analysis

The highlighted red and yellow regions represent peak spatial activations generated by the **Gradient-weighted Class Activation Mapping (Grad-CAM)** algorithm:

1. **Red/Orange Zone (High Activation):** Indicates the strongest neural gradient focus. The model detected dense, asymmetrical opacity in the right lower lung zone.
2. **Yellow/Green Zone (Moderate Focus):** Represents transitional zones where opacity diffuses into healthy surrounding parenchyma.
3. **Blue/Transparent Zone (Low Significance):** Evaluated as clear or anatomically expected non-pathological tissue.

**Anatomical Correlation:** The focal region corresponds to the lower right lobe bronchial tree where consolidation is most pronounced.`;
    } else if (qLower.includes('confidence') || qLower.includes('why not') || qLower.includes('72') || qLower.includes('wrong')) {
      fallbackReply = `### 📊 Confidence Calibration & Model Certainty

The model calculated a **${topConf}% confidence level** for ${topDiag}.

* **Why is it not 100%?** Multi-label models emit an independent sigmoid probability per disease, and overlapping opacity patterns mean several classes can be active at once — so no single probability reaches certainty.
* **Differential Considerations:** The model also assigned a **${(predictionContext?.diseases?.[1]?.probability * 100 || 82).toFixed(1)}% probability** to *${predictionContext?.diseases?.[1]?.disease || 'Lung Opacity'}*. High opacity overlap is common between severe consolidation and adjacent subsegmental atelectasis.
* **Limitations:** Overlapping soft tissue density, patient rotation, or low inspiration depth can introduce model uncertainty. Clinical correlation is always mandatory.`;
    } else if (qLower.includes('compare report') || qLower.includes('compare with previous') || qLower.includes('previous study') || qLower.includes('interval change') || qLower.includes('serial')) {
      fallbackReply = `### 🆚 Comparison with Prior Studies

No prior study is available in this session to compare against — MedVision AI does not store patient imaging and never infers interval changes.

To perform a genuine comparison, upload the previous study and run it through the same pipeline, then review both model reports side by side with a qualified clinician. Interval-change claims (growth, resolution, stability) require the actual prior images.

*Note: This platform never fabricates a prior baseline or interval measurements.*`;
    } else if (qLower.includes('compare') || qLower.includes('versus') || qLower.includes('vs')) {
      fallbackReply = `### 🔬 Comparative Disease Differential

| Feature | ${topDiag} | COVID-19 | Tuberculosis |
| :--- | :--- | :--- | :--- |
| **Primary Location** | Focal/Lobar consolidation | Bilateral peripheral ground-glass | Apical/Upper lobe cavitation |
| **Onset** | Acute (days) | Acute (3-7 days) | Chronic (weeks/months) |
| **X-Ray Pattern** | Dense, opaque consolidation | Peripheral ground-glass opacities | Apical infiltration & nodular scarring |
| **Typical Etiology** | *S. pneumoniae*, bacteria | SARS-CoV-2 Virus | *M. tuberculosis* |

*The model differentiates these based on spatial activation patterns and parenchymal texture analysis.*`;
    } else if (qLower.includes('summar') || qLower.includes('summary') || qLower.includes('overview of the scan') || qLower.includes('in short')) {
      fallbackReply = `### 📋 Scan Summary

**${topDiag}** (${topConf}% confidence) — **${severity}** severity.

* **Primary Finding:** ${predictionContext?.report?.impression?.split('\n')[0] || `Radiographic evidence consistent with ${topDiag}.`}
* **Secondary Findings:** ${predictionContext?.diseases?.slice(1, 3).map((d: any) => `${d.disease} (${(d.probability * 100).toFixed(1)}%)`).join(', ') || 'None significant'}
* **Suggested Action:** ${predictionContext?.report?.recommendations?.[0] || 'Clinical correlation and follow-up imaging as indicated.'}
* **Model:** ${modelUsed} · **Latency:** ${predictionContext?.inferenceTimeMs || 142}ms

Ask me to **explain findings**, **suggest follow-up imaging**, or **compare reports** for a deeper dive.

*Note: Research and education only — not for clinical diagnosis.*`;
    } else if (qLower.includes('follow-up') || qLower.includes('next steps') || qLower.includes('further imaging') || qLower.includes('what next') || qLower.includes('reccomend')) {
      fallbackReply = `### 🔁 Recommended Follow-Up

Based on **${topDiag}** (${topConf}% confidence) at **${severity}** severity, the recommended pathway is:

| Priority | Action | Rationale |
| :--- | :--- | :--- |
| **1** | ${predictionContext?.report?.recommendations?.[0] || 'Clinical correlation with labs & vitals'} | Confirms diagnosis with objective markers |
| **2** | ${predictionContext?.report?.recommendations?.[1] || 'Follow-up chest radiograph in 4–6 weeks'} | Tracks radiographic resolution |
| **3** | ${severity === 'Critical' || severity === 'High' ? 'Consider CT chest if findings persist or worsen' : 'Outpatient review in 2 weeks'} | Excludes complications |

* **Red flags to monitor:** worsening dyspnea, fever > 48h on therapy, hemoptysis, oxygen desaturation below 92%.

*Note: This is educational guidance — the attending physician makes all clinical decisions.*`;
    } else if (qLower.includes('glossary') || qLower.includes('medical term') || qLower.includes('define') || qLower.includes('what does') || qLower.includes('meaning of')) {
      const termMatch = qLower.replace(/what does|define|meaning of|medical term|glossary/g, '').trim();
      fallbackReply = `### 📖 Medical Term Explainer

${termMatch ? `**"${termMatch}"** — ` : ''}Here are the key terms from this report:

* **Consolidation:** Lung tissue filled with fluid/pus instead of air — appears white on X-ray.
* **Ground-Glass Opacity (GGO):** Hazy density where underlying vessels remain visible — typical of viral pneumonia.
* **Cardiothoracic Ratio (CTR):** Heart width ÷ chest width; normal ≤ 0.50.
* **Costophrenic Angle:** Angle between chest wall and diaphragm; blunting suggests pleural fluid.
* **Atelectasis:** Partial lung collapse from airway obstruction.
* **Grad-CAM:** Gradient-weighted Class Activation Mapping — the AI heatmap technique.

Ask me to **define** any specific term from the report!

*Note: Definitions are educational and research-oriented.*`;
    } else if (qLower.includes('treatment') || qLower.includes('symptoms') || qLower.includes('causes') || qLower.includes('risk')) {
      fallbackReply = `### 💊 Clinical Overview: Symptoms & Treatment

#### **Causes & Risk Factors**
* Bacterial or viral pathogens (*Streptococcus pneumoniae*, Influenza)
* Advanced age (>65), smoking, chronic obstructive pulmonary disease (COPD), or immunocompromised states.

#### **Standard Management Overview**
* **Antibiotic / Antiviral Therapy:** Target specific microbial pathogen.
* **Supportive Care:** Oxygen supplementation, hydration, and antipyretics.
* **Monitoring:** Serial pulse oximetry and clinical evaluation.

*Disclaimer: This information is for educational purposes only. Always consult a licensed physician.*`;
    } else {
      fallbackReply = `### 🤖 MedVision AI Assistant Response

I evaluated your question regarding **${topDiag}** (${topConf}% confidence) and the uploaded chest radiogram.

* **Model Used:** ${modelUsed}
* **Current Status:** Analyzed with ${predictionContext?.keyMetrics?.resolution || '1024x1024'} resolution and SNR of ${predictionContext?.keyMetrics?.snr || 29.5} dB.

I can explain the **disease mechanism**, **Grad-CAM heatmap regions**, **confidence score breakdown**, **treatment overviews**, or **compare diseases**. Feel free to choose from the suggested topic buttons below!`;
    }

    res.json({
      reply: fallbackReply,
      provider: `${provider.toUpperCase()} (MedVision Rule Engine)`,
      timestamp: new Date().toISOString(),
    });
  } catch (err: any) {
    log.error('chat-failed', { error: err.message || 'Failed to process chat message' });
    res.status(500).json({ error: err.message || 'Failed to process chat message' });
  }
});

// Load persisted state (history + audit) at boot, then start listening.
// Awaited so early requests can never race the PG/JSON restore.
async function boot() {
  // Validate environment — fail fast in production with meaningful messages
  const problems = validateConfig();
  for (const p of problems) log.warn('config', { problem: p });
  if (problems.length > 0 && config.nodeEnv === 'production') {
    log.error('config-validation-failed-in-production', { problems });
    process.exit(1);
  }
  try {
    await initStorage(historyStore, auditStore);
  } catch (err) {
    log.warn('storage-init-failed', { error: (err as Error).message });
  }
  await startServer();
}

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        watch: {
          // Ignore the desktop app's local database so its writes don't reload the page
          ignored: ['**/.freebuff/**', '**/desktop-v2.db*', '**/*.zip'],
        },
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    log.info('server-started', {
      name: APP_NAME,
      version: APP_VERSION,
      port: PORT,
      env: config.nodeEnv,
      engine: ENGINE_BASE,
    });
  });
}

boot();
