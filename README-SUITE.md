# MedVision AI — Enterprise Suite Handoff Guide

> **Version 2.5.0 · Enterprise Edition**
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
7. [Authentication & Roles](#7-authentication--roles)
8. [Real ML Inference (FastAPI · Port 8000)](#8-real-ml-inference-fastapi--port-8000)
9. [Environment Variables](#9-environment-variables)
10. [Project Structure](#10-project-structure)
11. [Production & Deployment](#11-production--deployment)
12. [Security & Compliance Notes](#12-security--compliance-notes)

---

## 1. What This Is

MedVision AI is a **full-stack, end-to-end explainable medical imaging suite**:

- **React 19 + Vite + TypeScript + Tailwind v4** frontend with a premium enterprise design system (glassmorphism, dark/light themes, motion, skeletons, empty/error states).
- **Express server** (TypeScript, `tsx`) that serves the app, orchestrates Gemini-powered radiology reports, hosts the AI copilot, issues JWT sessions, and **proxies inference to a real PyTorch engine** when available.
- **FastAPI + PyTorch engine** (`backend/`) that performs genuine DenseNet-121 inference with a graceful three-tier fallback chain.
- **Grad-CAM explainability**, a multi-method **explainability explorer**, **differential diagnosis**, **QR-verified PDF/DOCX reports**, an **AI radiology copilot** (voice in/out), a **command palette**, **role-based access**, and **MLOps tooling** — all in one codebase.

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

The header badge flips from **`DEMO ENGINE`** (amber) to **`PYTORCH · CUDA/CPU`** (emerald) within ~30 seconds, and `/api/predict` starts routing through the real model. If the engine is ever offline, the platform **silently falls back** to simulation — nothing breaks.

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
│  • Command palette · toasts · notifications · role switcher       │
│  • AppContext: theme, session (JWT), engine status, live feed     │
└───────────────┬────────────────────────────────────────────────────┘
                │  fetch (same origin)
┌───────────────▼────────────────────────────────────────────────────┐
│  Express server (server.ts · :3000)                               │
│  • Static SPA / Vite middleware                                   │
│  • Gemini report + copilot synthesis (server-side, optional key)  │
│  • JWT auth (HMAC-SHA256, node crypto)                            │
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
- **Prediction Queue** with admin **Flush** action

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

### 13. Training Studio (`training`)
- Hyperparameters (epochs, batch size, LR, optimizer, scheduler), **GPU/accelerator telemetry**, **simulated live loss/AUROC streaming curves**, early stopping, checkpoints, experiment run table

### 14. Dataset Registry (`datasets`)
- NIH ChestX-ray14, CheXpert, MIMIC-CXR, VinDr-CXR, RSNA Pneumonia
- Label statistics, missing-label detection, duplicate detection, train/val/test splits, quality gates, upload

### 15. Settings (`settings`)
- **Appearance** (light/dark/system), **AI models** (report engine + backbone), **Notifications**, **Export preferences**, **Accessibility & Language**, and **Security & Session** (JWT demo login / sign-out — see §7)

---

## 5. Global Platform Features

| Feature | How |
|---|---|
| **Command palette** | `Ctrl/Cmd+K` — fuzzy search across pages, patients, models, datasets, and actions (New Scan, Toggle Copilot, Theme, Switch Role) |
| **Keyboard shortcuts** | `c` copilot · `t` theme · `n` new scan · `/` focus search (or open palette on small screens) · `?` shortcuts help · `Esc` closes modals/drawers |
| **Notifications** | Header bell with unread badge, dropdown panel, simulated live feed |
| **Toasts** | Success/info/warning/critical with auto-dismiss progress bars |
| **Error boundary** | Enterprise crash screen with reload, wrapped around the app |
| **Theme** | Premium light & dark modes, persisted, legacy-class override layer |
| **Role switcher** | Demo role switching (Admin/Radiologist/Doctor/Researcher/Student) + real JWT sessions |
| **Engine badge** | Live indicator in header: `PYTORCH · CUDA` vs `DEMO ENGINE` |
| **Lazy loading** | All enterprise pages code-split via `React.lazy` + skeletons |
| **Accessibility** | ARIA labels, keyboard navigation, focus rings, reduced-motion support, high-contrast option |

---

## 6. API Reference (Express · Port 3000)

### Inference & Copilot
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/predict` | Chest X-ray prediction. Body: `{ imageName, imageData, model?, clahe?, noiseRemoval? }`. **Routes to the real engine when online; otherwise deterministic simulation.** Returns `PredictionResult` + `engine` metadata. |
| `POST` | `/api/batch-predict` | `{ files: [{name,size}] }` → per-file diagnosis summary |
| `POST` | `/api/chat` | Copilot. `{ prompt, history?, predictionContext?, provider? }` → Gemini reply or rule-engine fallback |
| `GET` | `/api/health` | Service + Gemini-configured status |

### Auth (JWT)
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/auth/login` | `{ email, password }` → `{ token, user }`. Legacy `{ name }` quick-login preserved. |
| `GET` | `/api/auth/me` | Bearer token → current user |
| `POST` | `/api/auth/logout` | Ends session (client also clears the stored token) |

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
| `GET` | `/api/models` · `/api/stats` · `/api/history` · `DELETE /api/history/:id` | Legacy analytics/history |

> **Compatibility contract:** all endpoints are additive and backward-compatible. The frontend service layer (`src/services/api.ts`) gracefully falls back to typed mock data whenever a server is unavailable.

---

## 7. Authentication & Roles

**Mechanism:** HMAC-SHA256 signed tokens (JWT-style, `header.payload.signature`) generated with **Node's built-in `crypto`** — zero new dependencies. Signatures are verified with **constant-time comparison** (`crypto.timingSafeEqual`). Tokens expire after **8 hours**.

**Sessions** persist in `localStorage` (`medvision-token`) and are restored on reload via `/api/auth/me`, guarded against a logout race.

### Demo accounts

| Role | Email | Password |
|---|---|---|
| **Admin** | `admin@medvision.ai` | `admin123` |
| **Radiologist** | `radiologist@medvision.ai` | `rad123` |
| **Doctor** | `doctor@medvision.ai` | `doc123` |
| **Researcher** | `researcher@medvision.ai` | `res123` |
| **Student** | `student@medvision.ai` | `stu123` |

**Where to sign in:** Settings → **Security & Session** (click an account to autofill). Authenticated sessions override the demo role switcher until sign-out. The `can(...roles)` helper in `AppContext` drives role-aware UI.

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
| `POST` | `/api/predict` | Same schema as Express. `engine` field reports `source`, `device`, `pytorchVersion`, `checkpoint`. |
| `GET` | `/api/engine` | Engine readiness (probed by Express with a 30s cache) |
| `GET` | `/api/health` · `/api/models` · `/` | Health, model registry, root banner |

**Grad-CAM** (research reference implementation) lives in `backend/app/models/gradcam.py`; **DenseNet-121** in `backend/app/models/densenet.py`. The UI's heatmap rendering is currently a simulated SVG pipeline — swap `heatmapOverlayUrl` for a computed activation map to go fully real.

**Dependencies:** `pip install -r backend/requirements.txt` (torch/torchvision are optional — the engine degrades gracefully without them).

---

## 9. Environment Variables

Copy `.env.example` → `.env`:

| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Enables Gemini-generated reports + copilot responses (falls back to the rule engine when absent) | — |
| `PYTORCH_ENGINE_URL` | FastAPI engine base URL | `http://127.0.0.1:8000` |
| `JWT_SECRET` | HMAC signing key — **change in any shared deployment** | `medvision-dev-secret-change-me` |

---

## 10. Project Structure

```
medvision-ai/
├── server.ts                  # Express: API + auth + engine proxy + Gemini + SPA
├── src/
│   ├── App.tsx                # Shell: header, sidebar, lazy pages, palette, copilot
│   ├── pages/                 # Dashboard, Patients, ModelHub, Training, Datasets, Settings
│   ├── components/            # Viewer, copilot, UI kit, palette, toasts, boundary…
│   ├── contexts/AppContext.tsx# theme · session · engine · notifications · roles
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
- **Scaling pointers:** swap in-memory stores (history/stats) for Postgres/Redis; move the engine behind a GPU pool; add a message queue for batch jobs.

---

## 12. Security & Compliance Notes

- **Disclaimer-first design:** every surface (header banner, copilot, reports, footer) carries the research-only mandate — matching medical-demo best practice.
- **JWT:** HMAC-SHA256, constant-time signature verification, 8h expiry, client-persisted with server validation via `/api/auth/me`.
- **No PHI stored:** all patient data is curated mock data; nothing sensitive is written to disk.
- **Defense in depth (demo):** role-based `can()` gating is enforced client-side; a production deployment should mirror role checks server-side.
- **Secrets:** `.env` is git-ignored and never shipped in the archive; rotate `JWT_SECRET` and never commit real API keys.

---

*MedVision AI Enterprise Suite 2.5.0 — built for education, research, portfolio, and demonstration. Apache-2.0.*
