/**
 * MedVision AI — Express API smoke test (zero dependencies).
 *
 * Requires the dev server to be running, then:
 *   npm run dev          # terminal 1
 *   npm run test:api     # terminal 2
 *
 * Verifies the critical API contract end-to-end: health, engine status,
 * JWT auth (valid + invalid), predict (real-engine proxy shape), chat
 * fallback, enterprise endpoints, and batch inference.
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

// 3. Auth: invalid credentials rejected
const bad = await post('/api/auth/login', { email: 'nobody@medvision.ai', password: 'wrong' });
ok(bad.status === 401, 'POST /api/auth/login rejects bad credentials');

// 4. Auth: valid login issues a JWT
const login = await post('/api/auth/login', { email: 'radiologist@medvision.ai', password: 'rad123' });
ok(login.status === 200 && !!login.body.token, 'POST /api/auth/login issues a token');
const token = login.body.token || '';
ok(token.split('.').length === 3, 'token is a 3-part JWT');

// 5. /api/auth/me validates the token
const me = await get('/api/auth/me', { Authorization: `Bearer ${token}` });
ok(me.status === 200 && me.body.user?.role === 'Radiologist', 'GET /api/auth/me returns the session role', me.body.user?.role);

// 6. Tampered token rejected
const tampered = await get('/api/auth/me', { Authorization: `Bearer ${token}x` });
ok(tampered.status === 401, 'GET /api/auth/me rejects a tampered token');

// 7. Predict — COVID profile
const covid = await post('/api/predict', { imageName: 'covid_sample.png' });
ok(covid.status === 200 && covid.body.topDiagnosis === 'COVID-19', 'POST /api/predict detects COVID-19 profile');
ok(covid.body.diseases?.length === 10, 'predict returns all 10 disease probabilities');
ok(covid.body.engine && ['pytorch-checkpoint', 'pytorch-backbone', 'demo'].includes(covid.body.engine.source), 'predict includes engine metadata');
ok(!!covid.body.originalImageUrl || covid.body.engine?.proxied !== undefined, 'predict returns image-echo fields (frontend contract)');

// 8. Predict — normal profile
const normal = await post('/api/predict', { imageName: 'normal_cxr.png' });
ok(normal.status === 200 && normal.body.topDiagnosis === 'No Finding' && normal.body.severity === 'Low', 'POST /api/predict normal → No Finding / Low');

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

// 11. Batch predict
const batch = await post('/api/batch-predict', { files: [{ name: 'a.png', size: 100 }, { name: 'b.png', size: 200 }] });
ok(batch.status === 200 && batch.body.batchResults?.length === 2, 'POST /api/batch-predict processes the queue');

// 12. Legacy endpoints preserved
const stats = await get('/api/stats');
ok(stats.status === 200 && typeof stats.body.totalPredictions === 'number', 'GET /api/stats legacy endpoint preserved');

console.log(`\n${passed} passed · ${failed} failed\n`);
// Let Node drain keep-alive handles naturally to avoid Windows libuv exit noise
process.exitCode = failed === 0 ? 0 : 1;
