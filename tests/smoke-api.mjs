/**
 * MedVision AI — Express API smoke test (zero dependencies).
 *
 * Requires the dev server to be running, then:
 *   npm run dev          # terminal 1
 *   npm run test:api     # terminal 2
 *
 * Verifies the critical API contract end-to-end: health, engine status,
 * public research mode (no auth endpoints), predict (real-engine proxy shape),
 * chat fallback, enterprise endpoints, safety gate, rate limiting, and
 * observability (version / ready / metrics / monitoring / audit).
 */

const BASE = process.env.MEDVISION_BASE || 'http://127.0.0.1:3000';

let passed = 0;
let failed = 0;

function ok(cond, label, extra = '') {
  if (cond) {
    passed += 1;
    console.log(`  ✓ ${label}`);
  } else {
    failed += 1;
    console.error(`  ✗ ${label}${extra ? ` — ${extra}` : ''}`);
  }
}

async function get(path, headers = {}) {
  const res = await fetch(`${BASE}${path}`, { headers, signal: AbortSignal.timeout(5000) });
  return { status: res.status, body: await res.json().catch(() => ({})) };
}

async function post(path, payload, headers = {}) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(5000),
  });
  return { status: res.status, body: await res.json().catch(() => ({})) };
}

console.log(`\nMedVision API smoke test → ${BASE}\n`);

// 0. Reachability — friendly failure when the server isn't running
let health;
try {
  health = await get('/api/health');
} catch (err) {
  console.error(`\nCannot reach ${BASE} — is the dev server running? Start it with:  npm run dev\n`);
  console.error(String(err.message || err).slice(0, 200));
  process.exit(1);
}

// 1. Health
ok(health.status === 200 && health.body.status === 'ok', 'GET /api/health returns ok');

// 2. Engine status (must exist and report an honest source)
const engine = await get('/api/engine');
ok(engine.status === 200 && ['ready', 'offline'].includes(engine.body.status), 'GET /api/engine reports status');
ok(['pytorch-checkpoint', 'pytorch-backbone', 'demo'].includes(engine.body.source), 'engine source is one of the three modes', engine.body.source);

// 3. Authentication removed — public research mode (no login, JWT, or accounts)
const legacyLogin = await post('/api/auth/login', { email: 'radiologist@medvision.ai', password: 'rad123' });
ok(legacyLogin.status === 404, 'POST /api/auth/login is removed — 404, no accounts');

const legacyMe = await get('/api/auth/me');
ok(legacyMe.status === 404, 'GET /api/auth/me is removed — 404, no sessions');

// 7. Predict — user upload without image bytes → rejected, no prediction
const noBytes = await post('/api/predict', { imageName: 'covid_sample.png' });
ok(noBytes.status === 422 && noBytes.body.code === 'INVALID_IMAGE' && noBytes.body.predictionGenerated === false, 'upload without image bytes is rejected — no prediction', String(noBytes.status));

// 8. Adversarial: fabricated client validation must NEVER authorize inference.
// Engine offline → 503; engine online → the authoritative server-side gate
// still refuses the non-CXR payload with 422. Either way: NO prediction, no
// demo fallback, regardless of what the client claims.
const attack = await post('/api/predict', {
  imageName: 'photo.jpg',
  imageData: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
  validation: { passed: true, valid: true, validated: true, safe: true, quality: 100, ood: false, imageType: 'chest-xray', confidence: 0.99 },
});
ok((attack.status === 503 && attack.body.code === 'ML_ENGINE_UNAVAILABLE' || attack.status === 422) && attack.body.predictionGenerated === false, 'fabricated client validation cannot authorize inference (503/422, no prediction)');

// 9. Known bundled sample study → explicit, clearly-labelled demo workflow
const sample = await post('/api/predict', { imageName: 'chest_xray_covid_bilateral.dcm' });
ok(sample.status === 200 && sample.body.workflow === 'sample' && sample.body.validationSource === 'sample-demo', 'known sample study → explicit sample-demo workflow', String(sample.status));
ok(sample.body.topDiagnosis === 'COVID-19' && sample.body.engine?.engineMode === 'demo-engine' && sample.body.engine?.predictionSource === 'demo-profile', 'sample demo profile is deterministic and demo-labelled');
ok(sample.body.diseases?.length === 10, 'sample demo returns all 10 disease probabilities');
ok(!!sample.body.originalImageUrl || sample.body.engine?.proxied !== undefined, 'predict returns image-echo fields (frontend contract)');

// 9b. Adversarial: spoofed sample filename + arbitrary bytes must NOT be treated
// as a sample — byte-verification sends it down the user-upload path (503 offline)
const spoof = await post('/api/predict', { imageName: 'chest_xray_covid_bilateral.dcm', imageData: 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciLz4=' });
ok((spoof.status === 503 || spoof.status === 422) && spoof.body.predictionGenerated === false, 'spoofed sample filename + wrong bytes → upload path, no demo diagnosis', String(spoof.status));

// 10. Sample normal → No Finding / Low
const sampleNormal = await post('/api/predict', { imageName: 'chest_xray_normal_screening.dcm' });
ok(sampleNormal.status === 200 && sampleNormal.body.topDiagnosis === 'No Finding' && sampleNormal.body.severity === 'Low', 'known sample (normal) → No Finding / Low demo');

// 9. Chat fallback engine (works with or without Gemini key)
const chat = await post('/api/chat', { prompt: 'Summarize the findings', predictionContext: { topDiagnosis: 'Pneumonia', topConfidence: 0.94, severity: 'High', diseases: [{ disease: 'Pneumonia', probability: 0.94 }, { disease: 'Lung Opacity', probability: 0.8 }] } });
ok(chat.status === 200 && typeof chat.body.reply === 'string' && chat.body.reply.length > 50, 'POST /api/chat returns a substantive reply');

// 10. Enterprise endpoints
const dashboard = await get('/api/dashboard');
ok(dashboard.status === 200 && dashboard.body.aiStatus, 'GET /api/dashboard returns KPI snapshot');

const patients = await get('/api/patients');
ok(patients.status === 200 && Array.isArray(patients.body) && patients.body.length > 0, 'GET /api/patients returns registry');

const patientDetail = await get('/api/patients/PAT-883921');
ok(patientDetail.status === 200 && Array.isArray(patientDetail.body.visits), 'GET /api/patients/:id returns EHR detail');

const modelsHub = await get('/api/models-hub');
ok(modelsHub.status === 200 && modelsHub.body.length >= 14, 'GET /api/models-hub returns ≥14 architectures');

const datasets = await get('/api/datasets');
ok(datasets.status === 200 && datasets.body.length >= 5, 'GET /api/datasets returns the 5 standard datasets');

const training = await get('/api/training/runs');
ok(training.status === 200 && Array.isArray(training.body), 'GET /api/training/runs returns runs');

// 11. Batch — authoritative mode never fabricates. Engine offline → 503;
// engine online with no image bytes → every file reports an explicit error.
const batchStrict = await post('/api/batch-predict', { files: [{ name: 'a.png', size: 100 }, { name: 'b.png', size: 200 }] });
const batchOk =
  (batchStrict.status === 503 && batchStrict.body.code === 'ML_ENGINE_UNAVAILABLE') ||
  (batchStrict.status === 200 && Array.isArray(batchStrict.body.batchResults) &&
    batchStrict.body.batchResults.length === 2 &&
    batchStrict.body.batchResults.every((r) => r.status === 'error'));
ok(batchOk, 'authoritative batch never fabricates (503 offline, or per-file errors online)', String(batchStrict.status));

// 12. Batch — explicit demo mode is allowed and flagged
const batchDemo = await post('/api/batch-predict', { files: [{ name: 'a.png', size: 100 }, { name: 'b.png', size: 200 }], mode: 'demo' });
ok(batchDemo.status === 200 && batchDemo.body.demo === true && batchDemo.body.batchResults?.length === 2, 'explicit demo batch returns flagged simulated profiles');

// 12. Legacy endpoints preserved
const stats = await get('/api/stats');
ok(stats.status === 200 && typeof stats.body.totalPredictions === 'number', 'GET /api/stats legacy endpoint preserved');

// 13. AI Safety Gate — /api/validate honours the client-side report
const valOk = await post('/api/validate', { imageName: 'cxr.png', clientValidation: { passed: true, score: 88 } });
ok(valOk.status === 200 && valOk.body.passed === true && typeof valOk.body.quality?.score === 'number', 'POST /api/validate returns the safety-gate contract');

const valFail = await post('/api/validate', { imageName: 'photo.jpg', clientValidation: { passed: false, score: 12 } });
ok(valFail.status === 200 && valFail.body.passed === false, 'POST /api/validate honours a failing client report');

// 14. Predict honours the safety gate — failed validation → 422, no fallback inference
const gated = await post('/api/predict', {
  imageName: 'photo.jpg',
  imageData: 'data:image/png;base64,AAAA',
  validation: { passed: false, score: 10 },
});
ok(gated.status === 422, 'POST /api/predict rejects a failed validation report with 422');

// 15. Predict returns calibrated confidence + uncertainty metadata (sample demo)
ok(typeof sample.body.calibration?.calibratedTopConfidence === 'number' && sample.body.calibration?.temperature === 1, 'predict returns calibration metadata');
ok(sample.body.uncertainty && ['low', 'moderate', 'high'].includes(sample.body.uncertainty.level), 'predict returns an uncertainty estimate');

// 16. Live monitoring telemetry
const monitoring = await get('/api/monitoring');
ok(monitoring.status === 200 && typeof monitoring.body.uptimeSec === 'number' && monitoring.body.engine?.status, 'GET /api/monitoring returns live telemetry');

// 17. Audit log is public and anonymized in public research mode
const audit = await get('/api/audit-logs');
ok(audit.status === 200 && audit.body.actor === 'anonymous', 'GET /api/audit-logs is public with anonymous actors', String(audit.status));

// 18. Model hub provenance — every architecture declares its metric source
ok(modelsHub.body.every((m) => ['published', 'estimated', 'synthetic'].includes(m.source)), 'every hub model declares metric provenance');

// 19. Version endpoint
const version = await get('/api/version');
ok(version.status === 200 && version.body.name === 'medvision-ai' && typeof version.body.version === 'string', 'GET /api/version reports app name + version');

// 20. Readiness endpoint — 200 with dependency checks
const ready = await get('/api/ready');
ok(ready.status === 200 && ready.body.status === 'ready' && ready.body.checks?.server?.status === 'up', 'GET /api/ready reports readiness + checks');

// 21. Prometheus metrics endpoint (text/plain, so fetch text directly)
const metricsRes = await fetch(`${BASE}/api/metrics`, { signal: AbortSignal.timeout(5000) });
const metricsText = await metricsRes.text();
ok(metricsRes.status === 200 && metricsText.includes('medvision_requests_total') && metricsText.includes('medvision_engine_status'), 'GET /api/metrics exposes Prometheus-style counters');

// 22. Security headers on responses
const headers = await fetch(`${BASE}/api/health`, { signal: AbortSignal.timeout(5000) });
ok(headers.headers.get('x-content-type-options') === 'nosniff' && headers.headers.get('x-frame-options') === 'DENY', 'responses carry security headers (nosniff, frame deny)');
const reqId = headers.headers.get('x-request-id');
ok(!!reqId && reqId.length > 0, 'responses carry an x-request-id for tracing');

// 23. Public API rate limiting — a burst on an expensive endpoint returns 429
// (runs last so the per-IP budget reset does not affect earlier checks)
let saw429 = false;
for (let i = 0; i < 70; i++) {
  const r = await fetch(`${BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: 'rate-limit probe' }),
    signal: AbortSignal.timeout(5000),
  });
  if (r.status === 429) {
    saw429 = true;
    break;
  }
}
ok(saw429, 'public API rate limiter returns HTTP 429 on a burst');

console.log(`\n${passed} passed · ${failed} failed\n`);
// Let Node drain keep-alive handles naturally to avoid Windows libuv exit noise
process.exitCode = failed === 0 ? 0 : 1;
