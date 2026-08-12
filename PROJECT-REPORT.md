# MedVision AI — Project Report

**Version:** 2.7.0 · **Branch:** `main` · **Generated:** August 8, 2026
**License:** Apache-2.0 · **Purpose:** Explainable medical image report generator (chest X-ray analysis)

> ⚠️ **Clinical disclaimer:** Research, educational, and portfolio use only. Not for clinical diagnosis or patient triage. All AI outputs require radiologist validation.

---

## 1. Executive Summary

MedVision AI is a full-stack, end-to-end **explainable AI medical imaging platform** for chest X-ray (CXR) analysis. It detects 10 multi-label thoracic pathologies, produces **Grad-CAM** activation heatmaps, generates structured AI radiology reports (Findings / Impression / Recommendations) via **Gemini 3.6 Flash**, and exports hospital-grade **PDF/DOCX reports with QR verification**.

The platform operates in **public research mode** — no accounts, no JWT, no login — while retaining defense-in-depth security (rate limiting, payload caps, security headers, anonymized audit logging, server-side ML safety gates).

**Validation status:** TypeScript compiles clean (`tsc --noEmit` → exit 0). Working tree is clean. 2 commits in history (initial enterprise suite + hardening).

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, TypeScript, Vite 6, Tailwind v4, Recharts, Lucide, jsPDF, html2canvas, Motion |
| Server | Express 4 (TypeScript, `tsx`), Uvicorn/FastAPI (Python 3.10) |
| ML / Vision | PyTorch 2.2, TorchVision, Captum, Grad-CAM reference implementation |
| LLM | `@google/genai` — Gemini 3.6 Flash (server-side, optional key) |
| DevOps / MLOps | Docker Compose, MLflow, GitHub Actions CI |
| Persistence | Postgres (optional) → JSON file (`data/medvision-store.json`) → memory |

---

## 3. Codebase Size & Composition

| Metric | Value |
|---|---|
| TypeScript/TSX source | ~12,800 LOC |
| Python (backend + training) | ~880 LOC |
| Source files (`src/`) | 45 |
| React components | 26 |
| Lazy-loaded pages | 6 page modules → 15 workspace views |
| Git history | 2 commits (initial + hardening) |

---

## 4. System Architecture

```
React 19 SPA (src/) — 15 workspace views, command palette, toasts, notifications
        │  fetch (same origin)
Express server (server.ts · :3000)
  • Gemini report + copilot synthesis (optional key, rule-engine fallback)
  • Enterprise REST API + engine proxy + rate limiting + audit log
        │  HTTP 127.0.0.1:8000 (30s-cached probe)
FastAPI engine (backend/)          MLflow (docker-compose · :5000)
  • DenseNet-121 + Grad-CAM         • experiment tracking
  • 3-tier fallback: checkpoint → backbone → demo
```

**Inference path:** Upload → `/api/predict` → engine online? → real forward pass with `engine` metadata → Grad-CAM + report + copilot. Engine offline? → deterministic simulation clearly tagged `engine.source: "demo"`.

---

## 5. Feature Inventory — 15 Workspace Views

| # | View | Highlights |
|---|---|---|
| 1 | **Dashboard** | 8 KPI cards, throughput charts, severity donut, live activity feed, real telemetry (`/api/monitoring`, 5s poll) |
| 2 | **Inference** | Drag-drop upload, CLAHE preview, 5 sample X-rays, Grad-CAM viewer (colormaps, opacity, ROI), explainability explorer (6 methods), confidence chart, structured report card, PDF/DOCX export, QR verification |
| 3 | **Batch Processing** | Up to 100 files, animated progress, RFC 4180 CSV export |
| 4 | **Analytics & GPU** | ROC/PR/calibration curves, confusion matrix, cohort analytics |
| 5 | **DICOM Studio** | Zoom/pan/rotate, window-level presets, Grad-CAM fusion, segmentation overlay, measurements, annotations |
| 6 | **Knowledge Hub** | Disease knowledge base + clinical glossary |
| 7 | **History** | In-memory prediction log, bookmarks, filters |
| 8 | **Model Comparison** | DenseNet-121 → Swin-B head-to-head benchmarks |
| 9 | **MLOps** | ECE calibration, Platt scaling, MLflow, model registry |
| 10 | **Docs** | Architecture documentation |
| 11 | **Patients** | Searchable EHR registry, profile drawer, timeline, CSV export |
| 12 | **Model Hub** | 17 architectures with metric provenance badges (published/estimated/demo — no fake metrics) |
| 13 | **Training Studio** | Hyperparameters, GPU telemetry, simulated loss/AUROC curves, experiment runs |
| 14 | **Dataset Registry** | NIH, CheXpert, MIMIC-CXR, VinDr-CXR, RSNA with quality gates |
| 15 | **Settings** | Theme, AI models, notifications, accessibility, research-mode disclaimers |

---

## 6. API Surface (Express · Port 3000)

**Core inference & safety:**
- `POST /api/predict` — server-gated inference (422 invalid / 503 engine offline — never silent demo fallback for uploads)
- `POST /api/batch-predict` — per-file server-side validation
- `POST /api/validate` — AI Safety Gate (image type, OOD, quality)
- `POST /api/chat` — AI copilot (Gemini or rule-engine fallback)

**Observability:** `/api/health` · `/api/ready` · `/api/version` · `/api/metrics` (Prometheus) · `/api/monitoring` · `/api/audit-logs`

**Enterprise suite:** `/api/dashboard` · `/api/patients(/:id)` · `/api/models-hub` · `/api/datasets` · `/api/training/runs` · `/api/notifications` · `/api/engine` · `/api/history` (legacy)

**Auth:** `/api/auth/*` → deliberately removed (404). Public research mode.

---

## 7. AI Safety Pipeline (server-side, defense in depth)

```
UPLOAD → FILE VALIDATION → IMAGE VALIDATION → QUALITY SCORE →
IMAGE-TYPE CLASSIFIER (11 classes) → OOD DETECTION → CHEST X-RAY? →
DISEASE DETECTOR → EXPLAINABILITY → CALIBRATION → UNCERTAINTY → REPORT
```

- **Trust boundary:** the client is never trusted — fabricated client validation can never authorize inference (verified by smoke test).
- **Error contract:** `422 VALIDATION_FAILED` / `503 ML_ENGINE_UNAVAILABLE` / `502 INFERENCE_FAILED`, always with `predictionGenerated: false`.
- **Calibration:** temperature scaling (default 1.0 = honest identity); raw vs calibrated confidence shown.
- **Uncertainty:** Monte-Carlo dropout (8 passes) when a fine-tuned head exists; honest margin/quality proxy otherwise.
- **Only exception:** bundled demo samples (server-verified filenames) return clearly-labelled `demo-engine` profiles.

**Real ML engine (FastAPI) — 3-tier honesty chain:**
1. `pytorch-checkpoint` — fine-tuned DenseNet-121 checkpoint → full real inference
2. `pytorch-backbone` — pretrained backbone, real forward pass, calibrated head (disclosed)
3. `demo` — deterministic simulation (only when torch is absent)

---

## 8. Testing & Validation Status

| Check | Status |
|---|---|
| `tsc --noEmit` (lint) | ✅ Pass (exit 0) |
| `npm run build` (vite + server bundle) | ✅ Pass |
| Python API tests (`pytest tests/test_api.py`) | ✅ 19/19 pass (health, engine, predict, validate, safety gate, CORS, explainability, similar-cases) |
| End-to-end engine smoke (`tests/e2e_smoke.py`) | ✅ 20/20 (engine → health → validation → prediction → explainability → report → similar-cases) |
| API smoke tests (`tests/smoke-api.mjs`) | 23 scenario groups — requires running server |
| CI (`.github/workflows/ci.yml`) | lint + build + pytest + e2e smoke on push/PR to `main`/`dev` |
| Training pipeline (`--synthetic-sanity`) | ✅ Runs end-to-end on CPU (device detection, split, class weights, checkpointing) |

---

## 9. Security Posture

- ✅ Security headers (nosniff, frame-deny, no-referrer, COOP/CORP), request IDs
- ✅ Per-IP rate limiting on expensive endpoints (60/min → 429)
- ✅ 27 MB payload cap (413), MIME/format validation, CORS whitelist
- ✅ Anonymized audit trail (`actor: "anonymous"`), persisted
- ✅ No PHI — all patient data is curated synthetic/demo
- ✅ Secrets via `.env` (git-ignored)
- ✅ Public research mode keeps security: "removing auth didn't remove security"

---

## 10. Observations & Recommendations

**Strengths**
1. Exceptionally thorough **safety-first inference design** — honest fallbacks, never fabricated diagnoses.
2. Broad feature surface (15 views) with consistent enterprise design system and accessibility (ARIA, focus rings, reduced motion).
3. Honest ML posture: provenance badges, calibration metadata, uncertainty disclosure.
4. Strong test coverage of the critical API contract (23 smoke scenarios including adversarial probes).

**Changes made in this hardening pass**
1. **Real explainability wired end-to-end** — Grad-CAM / Grad-CAM++ (fine-tuned head) or class-agnostic feature activation (backbone) now run inside the FastAPI engine; `/api/predict` returns real `heatmapUrl` / `heatmapOverlayUrl` data-URLs, a genuine peak-region bounding box with thoracic-zone label, and `explainability` metadata. Simulated UI methods are clearly labelled.
2. **Structured server-side validation** — `/api/validate` and every prediction return a `checks[]` report (PASS/WARN/FAIL) for format/resolution/grayscale/contrast/brightness/sharpness/orientation/chest_xray, with env-configurable thresholds.
3. **Real training pipeline** — `training/train.py` now trains (config, patient-level split, class weights, schedulers, early stopping, checkpoints, resume, CUDA/MPS/CPU detection), `training/evaluate.py` computes AUROC/AUPRC/precision/recall/F1/specificity/ECE + plots, and `--synthetic-sanity` validates the loop without real data. **No checkpoint ships and none is claimed.**
4. **Real similar-case retrieval** — `/api/similar-cases` computes genuine DenseNet-121 feature embeddings + cosine similarity (engine online); the UI falls back to clearly-labelled demo similarities offline.
5. **No fabricated patient data** — demo reports use `PAT-DEMO-*` ids; age/sex display "n/a" instead of invented 52/M; the copilot fallback no longer invents prior-baseline or interval-change numbers.
6. **Honesty fixes** — DICOM parsing claims removed (PNG/JPEG/WebP only), model AUROC values labelled published-reference, engine-offline note corrected, Gemini model id env-configurable.
7. **CI + tests** — pytest runs (19/19), `tests/e2e_smoke.py` added (20/20), `.gitignore` covers `artifacts/` and `training/checkpoints/`.

**Remaining opportunities**
1. **Fine-tuned checkpoint** — obtain NIH ChestX-ray14 (or CheXpert) legitimately, train with `training/train.py`, place the checkpoint under `training/checkpoints/`; the engine then runs full real inference with disease-specific Grad-CAM.
2. **CI server smoke** — start the dev server in CI and run `npm run test:api` (currently requires a running server).
3. **Version pinning** — package.json uses caret ranges; pin exact versions for fully reproducible production builds.

---

## 11. Quick Start

```bash
npm install && npm run dev     # http://localhost:3000 — full suite, simulated engine
# Optional real engine:
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --port 8000
# Production:
npm run build && NODE_ENV=production npm run dev
# Docker (app + MLflow):
docker-compose up --build -d
```

---

*Report generated from live codebase inspection — LOC counts, endpoint lists, and validation status are measured, not estimated.*
