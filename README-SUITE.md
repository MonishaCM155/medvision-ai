# MedVision AI — Enterprise Suite Handoff Guide

> **Version 2.7.0 · Enterprise Edition**
> An explainable AI medical imaging platform for chest X-ray analysis — built for hospitals, radiologists, researchers, medical students, healthcare startups, hackathons, and AI/ML portfolios.
>
> ⚠️ **Research & educational use only. Not for clinical diagnosis. All AI outputs require radiologist validation.**

---

## Table of Contents

1. [What This Is](#1-what-this-is)
2. [Quick Start](#2-quick-start)
3. [Architecture](#3-architecture)
4. [The 15 Workspace Pages](#4-the-15-workspace-pages)
5. [Global Platform Features](#5-global-platform-features)
6. [API Reference (Express · Port 3000)](#6-api-reference-express--port-3000)
7. [Public Research Mode](#7-public-research-mode)
8. [Real ML Inference (FastAPI · Port 8000)](#8-real-ml-inference-fastapi--port-8000)
9. [Environment Variables](#9-environment-variables)
10. [Project Structure](#10-project-structure)
11. [Production & Deployment](#11-production--deployment)
12. [Security & Compliance Notes](#12-security--compliance-notes)

---

## 1. What This Is

MedVision AI is a **full-stack, end-to-end explainable medical imaging suite**:

- **React 19 + Vite + TypeScript + Tailwind v4** frontend with a premium enterprise design system (glassmorphism, dark/light themes, motion, skeletons, empty/error states).
- **Express server** (TypeScript, `tsx`) that serves the app, orchestrates Gemini-powered radiology reports, hosts the AI copilot, and **proxies inference to a real PyTorch engine** when available. The platform runs in **public research mode — no login, no accounts, no JWT**.
- **FastAPI + PyTorch engine** (`backend/`) that performs genuine DenseNet-121 inference with a graceful three-tier fallback chain.
- **Grad-CAM explainability**, a multi-method **explainability explorer**, **differential diagnosis**, **QR-verified PDF/DOCX reports**, an **AI radiology copilot** (voice in/out), a **command palette**, and **MLOps tooling** — all in one codebase.

Every screen, endpoint, and workflow was designed to feel like enterprise hospital software (NVIDIA Clara / Philips IntelliSpace class) while remaining a self-contained demo you can run in seconds.

---

## 2. Quick Start

### A. Frontend + Express (everything simulated)

```bash
npm install
npm run dev          # http://localhost:3000  (Express + Vite HMR)
```

No backend required — prediction, reports, and the copilot all work with built-in fallbacks.

### B. Enable the real PyTorch inference engine (optional)

```bash
pip install -r backend/requirements.txt   # fastapi, uvicorn, torch, torchvision, captum…
uvicorn backend.app.main:app --port 8000
```

The header badge flips from **`DEMO ENGINE`** (amber) to **`PYTORCH · CUDA/CPU`** (emerald) within ~30 seconds, and `/api/predict` starts routing through the real model. If the engine is ever offline, prediction falls back to a **clearly-labeled demo profile** (`engine.source: "demo"`, amber `DEMO ENGINE` badge) — nothing breaks, and the fallback is never presented as real model inference.

### C. Production build

```bash
npm run build        # bundles to dist/
NODE_ENV=production npm run dev   # Express serves the static bundle
```

### D. Docker

```bash
docker-compose up --build -d       # app + MLflow tracking server
```

---

## 3. Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  React 19 + Vite + Tailwind v4 (src/)                             │
│  • 15 workspace pages (lazy-loaded / code-split)                  │
│  • Command palette · toasts · notifications                       │
│  • AppContext: theme, engine status, live feed (no accounts)      │
└───────────────┬────────────────────────────────────────────────────┘
                │  fetch (same origin)
┌───────────────▼────────────────────────────────────────────────────┐
│  Express server (server.ts · :3000)                               │
│  • Static SPA / Vite middleware                                   │
│  • Gemini report + copilot synthesis (server-side, optional key)  │
│  • Public research mode — no auth; rate limiting + security       │
│  • Enterprise REST API (dashboard, patients, models, datasets…)   │
│  • Engine proxy: probe → forward → silent fallback to simulation  │
└───────┬──────────────────────────────────────┬─────────────────────┘
        │  HTTP (127.0.0.1:8000, cached 30s probe)
┌───────▼──────────────────────┐   ┌───────────▼──────────────────────┐
│  FastAPI engine (backend/)   │   │  MLflow (docker-compose · :5000) │
│  • DenseNet-121 + Grad-CAM   │   └──────────────────────────────────┘
│  • pytorch-checkpoint >      │
│    pytorch-backbone > demo   │
└──────────────────────────────┘
```

**Inference path:** Upload → Express `/api/predict` → *engine online?* → FastAPI real forward pass → response with `engine` metadata → Grad-CAM viewer + report card + copilot context. *Engine offline?* → deterministic simulation (unchanged legacy path) tagged `engine.source: "demo"`.

---

## 4. The 15 Workspace Pages

### 1. Enterprise Dashboard (`dashboard`)
Operational command center:
- **8 KPI cards** — today's scans, pending reports, emergency cases, total patients, AI accuracy, mean latency, active models, prediction queue
- **Weekly Clinical Throughput** (scans vs reports, 7-day bars)
- **Severity Distribution** donut (Low/Moderate/High/Critical)
- **Disease Statistics** (30-day detection counts)
- **Live Activity Feed** — streaming scan/report/model/system events
- **Monthly Scan Volume** trend
- **Model & System Health** — GPU/CPU/Memory/Queue/Gemini/Storage utilization
- **Live System Monitor** — real telemetry from `GET /api/monitoring` (uptime, requests, errors, rejected images, memory, engine status, model version) polled every 5s — not simulated
- **Prediction Queue** with a public **Flush** action (demo — no permissions)

### 2. Inference (`inference`) — the core workflow
- **XrayUploader** — drag & drop, live CLAHE preprocessing preview, 5 curated SVG sample X-rays, model selector
- **GradCamViewer** — overlay / split / heatmap modes, colormap (Jet/Turbo/Viridis/Inferno), opacity, focal bounding box, downloadable map, **Explainability Method Explorer** (Grad-CAM, Grad-CAM++, IG, Saliency, Occlusion, Attention), **AI Decision Flow**, **Top Feature Contributions**, **Similar Case Retrieval**
- **ConfidenceChart** — Recharts probability bars for all 10 pathologies
- **ReportCard** — structured findings, impression, severity score, differential diagnosis, suggested tests, recommendations, **QR verification**, AI explanation
- **PdfExportModal** — paginated multi-page **PDF** (jsPDF) + **DOCX** export
- Graceful **inference error banner** + local fallback if the backend drops

### 3. Batch Processing (`batch`)
- Multi-file queue (up to 100), animated progress, results table
- **Real RFC 4180 CSV export** (quoted cells, escaped quotes, UTF-8 BOM)

### 4. Analytics & GPU (`analytics`)
- KPI telemetry + **model evaluation suite**: ROC curve, PR curve, calibration curve, **confusion matrix**, precision/recall/F1/AUROC
- **Cohort analytics**: disease distribution, **age distribution**, **feature importance**, monthly trends

### 5. DICOM Studio (`doctor`)
- **Zoom, pan (pointer-drag), rotate**, brightness/contrast sliders, **Window/Level presets** (Lung / Mediastinum / Bone), invert
- **Grad-CAM fusion** overlay, **U-Net segmentation overlay** (Segment toggle)
- **Measurement ruler** (px readouts), **annotation** tool, split comparison view
- Patient EHR sidebar + physician notes

### 6. Knowledge Hub (`knowledge`)
- Disease knowledge base (description, causes, symptoms, severity, risk factors, prevention, treatment, WHO/reference links) + clinical glossary

### 7. History (`history`)
- In-memory prediction log with select/delete, **bookmark (favorite) toggle + filter**

### 8. Models & Benchmarks (`models`)
- Head-to-head model comparison (DenseNet-121 → Swin-B), AUROC/F1/latency

### 9. MLOps (`mlops`)
- ECE calibration curves, Platt scaling, MLflow integration, model registry, deployment views

### 10. Docs (`docs`)
- Architecture documentation

### 11. Patients (`patients`)
- Searchable, filterable (status + department), sortable **EHR registry**
- **Profile drawer**: identity card, medical history, allergies, medications, visit history, **timeline**, follow-up schedule, risk score
- **Real CSV export** of the filtered registry

### 12. Model Hub (`modelhub`)
- **17 architectures**: CNN-family (ResNet-50, DenseNet-121, EfficientNet, MobileNet-V3, ConvNeXt), Transformers (ViT-B/16, Swin-B), Detection (YOLOv8, YOLOv11, Faster R-CNN), Segmentation (U-Net, Attention U-Net, DeepLabV3+), Traditional ML (XGBoost, LightGBM, Random Forest) and a stacked ensemble
- Per-model: accuracy, precision, recall, **specificity, sensitivity**, F1, AUROC, parameters, FLOPS, size, latency + **interactive comparison** (bars + radar)
- **Metric provenance badges** on every model: `PUBLISHED` (peer-reviewed/official benchmark, with citation tooltip — e.g. CheXNet 0.841 AUROC), `ESTIMATED` (extrapolated from comparable published models — no direct paper), `DEMO` (internal placeholder — not a benchmark). **No fake metrics.**

### 13. Training Studio (`training`)
- Hyperparameters (epochs, batch size, LR, optimizer, scheduler), **GPU/accelerator telemetry**, **simulated live loss/AUROC streaming curves**, early stopping, checkpoints, experiment run table

### 14. Dataset Registry (`datasets`)
- NIH ChestX-ray14, CheXpert, MIMIC-CXR, VinDr-CXR, RSNA Pneumonia
- Label statistics, missing-label detection, duplicate detection, train/val/test splits, quality gates, upload

### 15. Settings (`settings`)
- **Appearance** (light/dark/system), **AI models** (report engine + backbone), **Notifications**, **Export preferences**, **Accessibility & Language**, and **Public Research Mode** (mode explanation + medical disclaimers — see §7)

---

## 5. Global Platform Features

| Feature | How |
|---|---|
| **Command palette** | `Ctrl/Cmd+K` — fuzzy search across pages, patients, models, datasets, and actions (New Scan, Toggle Copilot, Theme) |
| **Keyboard shortcuts** | `c` copilot · `t` theme · `n` new scan · `/` focus search (or open palette on small screens) · `?` shortcuts help · `Esc` closes modals/drawers |
| **Notifications** | Header bell with unread badge, dropdown panel, simulated live feed |
| **Toasts** | Success/info/warning/critical with auto-dismiss progress bars |
| **Error boundary** | Enterprise crash screen with reload, wrapped around the app |
| **Theme** | Premium light & dark modes, persisted, legacy-class override layer |
| **Public mode** | Opens straight into the workspace — no login, no roles, no sessions |
| **Engine badge** | Live indicator in header: `PYTORCH · CUDA` vs `DEMO ENGINE` |
| **Lazy loading** | All enterprise pages code-split via `React.lazy` + skeletons |
| **Accessibility** | ARIA labels, keyboard navigation, focus rings, reduced-motion support, high-contrast option |

---

## 6. API Reference (Express · Port 3000)

### Inference & Copilot
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/predict` | Authoritative server-gated inference. Body: `{ imageName, imageData, model?, clahe?, noiseRemoval? }`. The server validates **itself** (FastAPI safety gate) and refuses when the engine is offline: `422` (invalid image) or `503 ML_ENGINE_UNAVAILABLE` — **never a demo fallback for uploads**. Known sample studies (`chest_xray_*.dcm`) return a clearly-labelled demo profile. Returns `PredictionResult` + `engine` + `validationSource`/`predictionGenerated`/`reportAllowed`. |
| `POST` | `/api/batch-predict` | `{ files: [{name,size,imageData?}], mode? }`. `mode: 'demo'` → flagged simulated profiles; otherwise engine-required with **per-file server-side validation** (offline → `503`). |
| `POST` | `/api/similar-cases` | Real retrieval: `{ queryImage, references: [{id,title,label,imageData}] }` → DenseNet-121 feature embeddings + cosine similarity (engine required; `503` offline). Similarity is **not** diagnostic evidence. |
| `POST` | `/api/chat` | Copilot. `{ prompt, history?, predictionContext?, provider? }` → Gemini reply or rule-engine fallback |
| `GET` | `/api/health` | Service + Gemini-configured status |
| `GET` | `/api/ready` | **Readiness probe** for orchestration: `200` with per-dependency checks (server / engine / storage / database), `503` when a configured dependency is down |
| `GET` | `/api/version` | App name + version + Node + engine info |
| `GET` | `/api/metrics` | **Prometheus-style text metrics**: requests, errors, predictions, rejected images, rate-limited, req/min, uptime, RSS/heap bytes, CPU %, engine status |

### Authentication (removed)
| Method | Path | Description |
|---|---|---|
| `*` | `/api/auth/*` | **404** — authentication was removed. MedVision AI is a public research/education platform: no login, logout, JWT, or accounts. |

### Enterprise Suite
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/dashboard` | KPI snapshot (falls back client-side to mock data) |
| `GET` | `/api/patients?q=` | Searchable registry |
| `GET` | `/api/patients/:id` | Full EHR detail |
| `GET` | `/api/models-hub` | 16-architecture registry |
| `GET` | `/api/datasets` | Dataset registry |
| `GET` | `/api/training/runs` | Experiment runs |
| `GET` | `/api/notifications` | Seed notification feed |
| `GET` | `/api/engine` | Inference-engine status (`ready`/`offline`, source, device) |
| `POST` | `/api/validate` | **AI Safety Gate** — `{ imageName, imageData?, clientValidation? }` → image-type classification, OOD verdict + quality score. Proxies the FastAPI engine when online; honest client-heuristic fallback otherwise |
| `GET` | `/api/monitoring` | **Live telemetry** — uptime, requests/min, errors, rejected images, predictions, rate-limited, avg inference ms, engine source, memory, CPU |
| `GET` | `/api/audit-logs` | **Public** anonymized security audit trail (predict/validate/batch/history events, last 200, `actor: "anonymous"`) |
| `GET` | `/api/models` · `/api/stats` · `/api/history` · `DELETE /api/history/:id` | Legacy analytics/history |

> **Compatibility contract:** all endpoints are additive and backward-compatible. The frontend service layer (`src/services/api.ts`) gracefully falls back to typed mock data whenever a server is unavailable.

---

## 7. Public Research Mode

**MedVision AI does not require user authentication.** There is no login, no logout, no JWT, no passwords, no user accounts, and no role-based access. The application opens **directly into the workspace** — Dashboard → use immediately.

- **Who it is for:** research, education, demonstration, and portfolio. It is **not** a substitute for professional medical diagnosis, and it stores **no PHI** — the patient registry, datasets, and reports are clearly labeled **synthetic/demo data**.
- **Removing authentication did not remove security:** the public API keeps its CORS whitelist, per-IP rate limiting, payload caps, MIME/format validation, security headers, request IDs, anonymized audit logging (`actor: "anonymous"`), and the ML safety gates.
- Legacy `/api/auth/*` endpoints return `404` with guidance; the frontend ships no login UI.

---

## 8. Real ML Inference (FastAPI · Port 8000)

`backend/app/main.py` exposes a genuine inference engine with a **three-tier honesty chain**:

| Mode | Condition | What actually happens |
|---|---|---|
| `pytorch-checkpoint` | A fine-tuned 10/14-class DenseNet-121 checkpoint exists in `training/checkpoints/*.pt` | Full real inference: image decoded → 256→224 ImageNet preprocessing → forward pass → **actual sigmoid probabilities** for each pathology |
| `pytorch-backbone` | torch/torchvision installed, no fine-tuned head | **Real forward pass** through the pretrained DenseNet-121 (real device latency measured); probability head uses the calibrated profile until a checkpoint is supplied (explicitly disclosed in the response) |
| `demo` | torch not installed | Deterministic, keyword-calibrated simulation — the same behavior as the Express fallback |

### FastAPI endpoints
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/predict` | Same schema as Express. `engine` field reports honest mode metadata: `engineMode` (`real-model` / `backbone-live` / `demo-engine`), `weightsLoaded`, `predictionSource`, `modelName`, `modelVersion`, plus `source`, `device`, `checkpoint`. Also returns `calibration`, `uncertainty`, `typeCheck`, `quality`, `ood` (see §8b). |
| `POST` | `/api/validate` | AI Safety Gate — real pixel analysis (PIL): 11-class type classifier, OOD verdict, 0–100 quality score. Returns 422 for non-image payloads. |
| `GET` | `/api/engine` | Engine readiness (probed by Express with a 30s cache) |
| `GET` | `/api/health` · `/api/models` · `/` | Health, model registry, root banner |

### 8a. Training & Evaluation (`training/`)

A real, reproducible training pipeline ships in `training/`:

```bash
python training/prepare_dataset.py --check-images                    # validate NIH layout + write reproducible patient-level splits
python training/train.py --config training/configs/train.yaml        # train on NIH ChestX-ray14
python training/train.py --config ... --synthetic-sanity             # validate pipeline, no data needed
python training/select_thresholds.py --checkpoint training/checkpoints/best.pt   # validation-set max-F1 thresholds
python training/evaluate.py --checkpoint training/checkpoints/best.pt --config training/configs/train.yaml
python training/predict_demo.py --checkpoint training/checkpoints/best.pt        # real predictions + Grad-CAM overlays
```

- **Transparent label mapping**: `Lung Opacity` ← NIH `Consolidation` ∪ `Infiltration`; `COVID-19` / `Tuberculosis` marked **unavailable** (absent from NIH ChestXray14) — kept as head slots, never reported as findings.
- Patient-level train/val/test split (persisted to `training/splits/` by `prepare_dataset.py`; no patient appears in two partitions), augmentation, inverse-frequency class weighting, cosine/step LR, early stopping, AMP (CUDA), best/last checkpoints, resume, experiment metadata JSON.
- **Per-class decision thresholds are selected on the validation set only** (max-F1), frozen into `best.pt` metadata + `training/results/thresholds.json`; the test set is never used to tune thresholds.
- Device auto-detection **CUDA → MPS → CPU** — CPU training works everywhere.
- `evaluate.py` writes AUROC (per-class/macro/micro), AUPRC, precision/recall/F1, sensitivity/specificity, ECE calibration and plots to `artifacts/evaluation/`, plus the test classification report to `training/results/classification_report.json`. Checkpoint metadata (class order, dataset, frozen thresholds, unavailable classes, epoch, val AUROC) is read back so operating metrics use the exact thresholds the engine applies.
- If the dataset/checkpoint is absent, every script reports honestly — **no fabricated metrics, no fake training**.
- `--synthetic-sanity` writes to `training/checkpoints/sanity/` so sanity weights are never mistaken for real model weights by the engine.
- **Checkpoint provenance in the engine**: when `training/checkpoints/best.pt` (or another `.pt`/`.pth`) exists with a 10/14-class head, `/api/engine` and every prediction report `engineMode: real-model`, `checkpointFile`, `dataset`, `trainedClasses` / `unavailableClasses`, `thresholds` and `thresholdPolicy` — the UI/API never present `backbone-live` as `fine-tuned`.

### 8b. AI Safety Pipeline (medically responsible by design)

**Trust boundary — the client is never trusted.** Frontend validation, quality scores, OOD verdicts, image-type labels, and confidence values are **advisory UI only**; none of them can authorize inference. Inference runs only when the **server** has independently established that (1) an image payload is present, (2) the FastAPI ML engine is reachable, and (3) the FastAPI safety gate (type + OOD + quality) passed. When authoritative validation or the ML engine is unavailable, `/api/predict` returns `503` (`ML_ENGINE_UNAVAILABLE` / `VALIDATION_UNAVAILABLE`) with `predictionGenerated: false` — **no demo disease prediction is ever silently substituted for failed real inference**, and a real inference failure returns `502 INFERENCE_FAILED` rather than a synthetic diagnosis. The only exception is the explicit **sample workflow**: requests naming one of the bundled demo studies (server-verified filename) receive a clearly-labelled demo profile (`workflow: "sample"`, `validationSource: "sample-demo"`, `engineMode: "demo-engine"`).

The platform refuses to diagnose random images. Every prediction passes through a gated pipeline, and **any failed stage stops inference immediately**:

```
UPLOAD → FILE VALIDATION → IMAGE VALIDATION → QUALITY SCORE →
IMAGE-TYPE CLASSIFIER (11 classes) → OOD DETECTION → CHEST X-RAY? →
DISEASE DETECTOR → EXPLAINABILITY → CALIBRATION → UNCERTAINTY → REPORT
```

- **Stage 1–2 (client, `src/utils/imageValidation.ts`)** — format/corruption, resolution, grayscale, orientation, exposure, contrast, sharpness (Laplacian), thoracic anatomy. Rejections show per-check PASS/FAIL reasons and **block `/api/predict`** (the Express server returns `422` for failed validation reports — no fallback inference).
- **Server-side per-check report** — `/api/validate` (and every prediction) returns a structured `checks[]` array with **PASS/WARN/FAIL** for `format` / `resolution` / `grayscale` / `contrast` / `brightness` / `sharpness` / `orientation` / `chest_xray`, each with a human-readable reason. All thresholds are env-configurable (`VALIDATION_*`).
- **Stage 3 (engine, `backend/app/main.py`)** — `_pixel_metrics` computes brightness/contrast/sharpness/noise/color + a thoracic structure signature from actual pixels (PIL only, no numpy).
- **Stage 4 — image-type classifier** — 11 classes (chest X-ray, other X-ray, CT, MRI, ultrasound, PET, mammography, photograph, animal, document, unknown). Labeled `heuristic-v1` — a trained CNN checkpoint (`training/checkpoints/image_type.pt`) is the documented upgrade path with an unchanged API contract.
- **Stage 5 — OOD detection** — `feature-proxy-v1` scores deviation from the supported CXR distribution (weak anatomy, poor quality, colour content, extreme framing) → `in` / `borderline` / `out`.
- **Stage 6 — calibration** — temperature scaling on the sigmoid logit (`MEDVISION_TEMPERATURE`, default `1.0` = honest identity). Report displays **raw vs calibrated confidence**.
- **Stage 7 — uncertainty** — **Monte-Carlo dropout** over a fine-tuned head when available (8 stochastic passes), otherwise an honest margin + quality proxy. High uncertainty surfaces a "confidence insufficient for reliable diagnosis" notice.

**Error states (API):** `422 VALIDATION_FAILED` / `UNSUPPORTED_IMAGE` / `INVALID_IMAGE` · `503 ML_ENGINE_UNAVAILABLE` / `VALIDATION_UNAVAILABLE` · `502 INFERENCE_FAILED` — every refusal carries `predictionGenerated: false` and `reportAllowed: false`, so no report, PDF/DOCX/QR, or diagnosis can be produced from an unsafe request.

**Grad-CAM / Grad-CAM++** (research implementations) live in `backend/app/models/gradcam.py`; **DenseNet-121** in `backend/app/models/densenet.py`. When the engine is online, `/api/predict` returns **real** activation maps: `heatmapUrl` + `heatmapOverlayUrl` are PNG data-URLs computed from actual gradients/activations, `gradCamRegions` carries a genuine peak-region bounding box (with thoracic-zone label) derived from the map, and `explainability.method` reports `grad-cam++`/`grad-cam` (fine-tuned head) or `feature-activation` (backbone-only — honestly labelled as not disease-specific). The remaining explainability-explorer methods are simulated CSS previews, explicitly labelled as such in the UI.

**Dependencies:** `pip install -r backend/requirements.txt` (torch/torchvision are optional — the engine degrades gracefully without them).

---

## 9. Environment Variables

Copy `.env.example` → `.env`:

| Variable | Purpose | Default |
|---|---|---|
| `PORT` | Express listen port | `3000` |
| `NODE_ENV` | `development` / `production` (production validates config and fails fast on invalid values) | `development` |
| `LOG_LEVEL` | Structured log verbosity: `debug`/`info`/`warn`/`error` | `info` |
| `ALLOWED_ORIGINS` | Comma-separated CORS whitelist (empty = same-origin SPA, no CORS headers) | — |
| `GEMINI_API_KEY` | Enables Gemini-generated reports + copilot responses (falls back to the rule engine when absent) | — |
| `GEMINI_MODEL` | Gemini model id used for report + copilot synthesis (invalid ids degrade gracefully to the rule engine) | `gemini-3.6-flash` |
| `MEDVISION_CLASS_THRESHOLDS` | Optional JSON per-class decision thresholds (`{"Pneumonia": 0.6, ...}`); default global 0.5 | `{}` |
| `VALIDATION_*` | Per-check validation thresholds: `VALIDATION_MIN_SHORT_EDGE` (256), `VALIDATION_MAX_COLOR_DEVIATION` (18), `VALIDATION_MIN_STD_CONTRAST` (8), `VALIDATION_MIN_SHARPNESS` (15), `VALIDATION_WARN_SHARPNESS` (30), `VALIDATION_MIN_STRUCTURE_SCORE` (45), `VALIDATION_WARN_STRUCTURE_SCORE` (60), `VALIDATION_MIN_BRIGHTNESS` (20), `VALIDATION_MAX_BRIGHTNESS` (235), `VALIDATION_MAX_ASPECT_RATIO` (1.45) | defaults listed |
| `PYTORCH_ENGINE_URL` | FastAPI engine base URL | `http://127.0.0.1:8000` |
| `MEDVISION_TEMPERATURE` | Calibration temperature for confidence scaling (1.0 = identity) | `1.0` |
| `QUALITY_THRESHOLD` | Minimum quality score for the safety gate (0–100) | `55` |
| `MEDVISION_CONFIDENCE_THRESHOLD` | Minimum confidence for a definitive diagnosis (below = indeterminate + exports locked) | `0.75` |
| `MEDVISION_API_RATE_LIMIT` | Public API requests per IP per minute on expensive endpoints (predict/validate/chat/batch) | `60` |
| `MAX_UPLOAD_SIZE` | Max upload payload in bytes (`413` beyond this) | `28311552` (27 MB) |
| `TRUST_PROXY` | Reverse-proxy hop count so rate limiting/audit see real client IPs (off by default) | `0` (off) |
| `MODEL_VERSION` | Engine model-version label reported in metadata | `2.7.0` |
| `DATABASE_URL` | Optional Postgres DSN — activates the durable storage adapter (else JSON file / memory) | — |

All variables are validated on startup in `config.ts`; production refuses to boot on an invalid `PYTORCH_ENGINE_URL` or implausible limits, printing a meaningful message. The FastAPI engine reads `ALLOWED_ORIGINS`/`FRONTEND_URL` for its CORS whitelist and `QUALITY_THRESHOLD`/`MODEL_VERSION`/`MEDVISION_TEMPERATURE` for the safety pipeline.

---

## 10. Project Structure

```
medvision-ai/
├── server.ts                  # Express: API + auth + engine proxy + Gemini + SPA
├── storage.ts                 # Durable persistence: Postgres → JSON file → memory
├── data/medvision-store.json  # Auto-created JSON store (history + audit log)
├── src/
│   ├── App.tsx                # Shell: header, sidebar, lazy pages, palette, copilot
│   ├── pages/                 # Dashboard, Patients, ModelHub, Training, Datasets, Settings
│   ├── components/            # Viewer, copilot, UI kit, palette, toasts, boundary…
│   ├── contexts/AppContext.tsx# theme · engine · notifications (no accounts)
│   ├── services/api.ts        # Typed API layer with mock fallbacks
│   ├── data/                  # sampleXrays, medicalKnowledge, mockEnterprise
│   ├── utils/                 # cn, format, qr (QR verification codes)
│   └── types.ts               # Strict shared types
├── backend/
│   ├── app/main.py            # FastAPI inference engine (3-tier fallback)
│   ├── app/models/            # densenet.py · gradcam.py (reference implementations)
│   └── requirements.txt
├── training/                  # dataset.py · train.py (NIH ChestX-ray14 loop)
├── tests/test_api.py          # API test scaffold
└── docker-compose.yml         # App + MLflow
```

---

## 11. Production & Deployment

- **Build:** `npm run build` → Express serves `dist/` in `NODE_ENV=production`
- **Docker:** `docker-compose up --build -d` (web app + MLflow on :5000)
- **CI:** GitHub Actions workflow (`.github/workflows/ci.yml`)
- **Persistence:** history + audit log now persist to a JSON file (`data/medvision-store.json`) by default; set `DATABASE_URL` (with the optional `pg` package) for PostgreSQL. **Docker:** `docker-compose.yml` includes a `db` service — `DATABASE_URL=postgres://medvision:medvision@db:5432/medvision` switches the whole suite to Postgres with one command.
- **Scaling pointers:** move the engine behind a GPU pool; add a message queue for batch jobs.

---

## 12. Security & Compliance Notes

- **Disclaimer-first design:** every surface (header banner, copilot, reports, footer) carries the research-only mandate — matching medical-demo best practice.
- **Structured logging:** single-line JSON records (timestamp, level, message, request ID, method, path, status, duration) via the `requestLogger` middleware — every response carries an `x-request-id` for tracing. Errors go to stderr.
- **Security headers:** `nosniff`, `X-Frame-Options: DENY`, `no-referrer`, `COOP: same-origin`, `CORP: same-origin`, a permissive CSP-friendly `Permissions-Policy` (microphone deliberately allowed for voice features). Dependency-free Helmet subset.
- **CORS whitelist:** only when `ALLOWED_ORIGINS` is set; same-origin deployments send no CORS headers.
- **No authentication:** public research mode — no login, logout, JWT, passwords, sessions, or roles anywhere in the stack.
- **Rate limiting:** per-IP token bucket on expensive public endpoints (`/api/predict`, `/api/validate`, `/api/chat`, `/api/batch-predict`; default 60/min → `429`).
- **Audit trail:** security-relevant endpoints (predict/validate/batch/history) write an anonymized audit log (`actor: "anonymous"`), public via `GET /api/audit-logs`, persisted with the storage adapter.
- **Input caps:** `/api/predict` and `/api/validate` reject payloads > 27 MB (`413`) and non-image data-URLs (`422`) before any processing.
- **No PHI stored:** all patient data is curated mock data; nothing sensitive is written to disk.
- **Defense in depth:** the ML safety gates are enforced **server-side** (Express + FastAPI reject invalid images before any inference) — the frontend is never trusted.
- **Secrets:** `.env` is git-ignored and never shipped in the archive; never commit real API keys.

---

*MedVision AI Enterprise Suite 2.7.0 — built for education, research, portfolio, and demonstration. Apache-2.0.*
