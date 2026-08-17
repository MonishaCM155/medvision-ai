# MedVision AI: Explainable Medical Image Report Generator

[![PyTorch](https://img.shields.io/badge/PyTorch-v2.2.0-red.svg?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.109-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.0-61DAFB.svg?logo=react)](https://react.dev/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-v4.0-38BDF8.svg?logo=tailwindcss)](https://tailwindcss.com/)
[![Gemini 3.6](https://img.shields.io/badge/Gemini-3.6_Flash-8E75B2.svg?logo=google)](https://ai.google.dev/)
[![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA-CUDA_12.2-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> **MANDATORY CLINICAL DISCLAIMER:**
> **Not for clinical diagnosis or patient triage.** This software application is developed exclusively for educational, academic, computer vision research, and technical portfolio benchmarking. All predictions and AI reports must be reviewed and verified by a licensed diagnostic radiologist.

---

## 🌟 Executive Summary & Key Highlights

**MedVision AI** is an end-to-end explainable AI platform designed to analyze Chest Radiographs (X-rays), detect multi-label thoracic pathologies, compute local feature activations via **Grad-CAM & Grad-CAM++**, and generate structured **Gemini 3.6 Flash** radiology reports complete with estimated severity scores and hospital-grade PDF exports.

Designed for high-throughput clinical research and portfolio evaluation (NVIDIA, Machine Learning Engineer, Computer Vision AI Researcher), MedVision AI features:

- 🩺 **10 Multi-Label Thoracic Pathologies**: No Finding, Pneumonia, COVID-19, Tuberculosis, Cardiomegaly, Pleural Effusion, Edema, Atelectasis, Pneumothorax, and Lung Opacity.
- 🔍 **Real Grad-CAM / Grad-CAM++ Heatmaps**: when the PyTorch engine is online, activation maps are computed server-side from real gradients/activations and returned as data-URL overlays with genuine peak-region bounding boxes; other explainability methods are clearly-marked simulated previews.
- 📝 **AI Radiology Report Generator**: Dynamic structured reports (**Findings**, **Impression**, and **Recommendations**) powered server-side by `@google/genai` Gemini 3.6 Flash.
- 📊 **Model Comparison & Benchmarks**: Comparative evaluation of DenseNet-121 (CheXNet), EfficientNet-B3, ConvNeXt-Base, Swin Transformer (Swin-B), and Vision Transformer (ViT-B/16).
- ⚙️ **MLOps & Probability Calibration**: Expected Calibration Error (ECE) analysis, Platt scaling reliability curves, and MLflow experiment logging.
- 📑 **Hospital PDF Exporter**: Generate printable, hospital-formatted PDF radiology reports powered by `jspdf` and `html2canvas`.

---

## 🏗️ System Architecture

```
                                  +------------------------------------+
                                  |   Chest Radiograph Upload / Sample  |
                                  +------------------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | Preprocessing (CLAHE + Bilateral)  |
                                  +------------------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | PyTorch Backbone (DenseNet-121)    |
                                  +------------------------------------+
                                      /                            \
                                     v                              v
                    +----------------------------------+  +-----------------------------------+
                    | Multi-Label Sigmoid Classifier   |  | Grad-CAM Feature Activation Hook  |
                    | (10 Pathology Probabilities)     |  | (Target Layer: denseblock4.conv2) |
                    +----------------------------------+  +-----------------------------------+
                                     \                              /
                                      v                            v
                                  +------------------------------------+
                                  | Gemini 3.6 Flash Structured Report |
                                  +------------------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | Interactive Web UI & PDF Exporter  |
                                  +------------------------------------+
```

---

## 🛠️ Tech Stack

- **Frontend**: React 19, TypeScript, Vite, TailwindCSS, Recharts, Lucide Icons, jsPDF, html2canvas.
- **Backend**: Express (Node.js) / FastAPI (Python 3.12), Uvicorn, Pydantic.
- **Machine Learning & Vision**: PyTorch, TorchVision, NumPy, Pillow, Grad-CAM / Grad-CAM++ (reference implementation in `backend/app/models/gradcam.py`). Captum / OpenCV are optional extras in `requirements.txt`.
- **LLM Integration**: `@google/genai` (Gemini, configurable via `GEMINI_MODEL`) server-side proxy with an offline rule-engine fallback.
- **DevOps & MLOps**: Docker, Docker Compose (app + Postgres + MLflow), GitHub Actions CI.

---

## ⚡ Quick Start & Installation

### 1. Local Development (Node / Express + Vite)
```bash
# Clone the repository
git clone https://github.com/your-username/medvision-ai.git
cd medvision-ai

# Install dependencies
npm install

# Start full-stack dev server (runs on http://0.0.0.0:3000)
npm run dev
```

### 2. Docker Compose Deployment
```bash
# Build and launch multi-container stack (App + MLflow)
docker-compose up --build -d

# Check application status at http://localhost:3000
# Access MLflow server at http://localhost:5000
```

---

## 🧪 Training & Evaluation Pipeline

A complete, reproducible training pipeline lives in `training/` (DenseNet-121
multi-label head, patient-level split, augmentation, class weighting, cosine
scheduling, early stopping, checkpoints, resume, device auto-detection
CUDA → MPS → CPU).

**Label mapping (NIH ChestX-ray14 → project classes, fully transparent):**
`No Finding`, `Pneumonia`, `Cardiomegaly`, `Pleural Effusion` (← NIH
`Effusion`), `Edema`, `Atelectasis`, `Pneumothorax` and `Lung Opacity` (← NIH
`Consolidation` ∪ `Infiltration`) are trainable. `COVID-19` and
`Tuberculosis` are **not present in NIH ChestXray14** (2017 dataset; TB is a
separate NIH collection) — they are marked **unavailable**, keep a head slot
for the API contract, and are never reported as model findings.

```bash
# 1. Obtain NIH ChestX-ray14 (https://nihcc.app.box.com/v/ChestXray-NIHCC):
#    Data_Entry_2017.csv + images/ extracted from images_001..012.zip (~42 GB)

# 2. Validate layout, print per-class statistics, write reproducible
#    patient-level train/val/test splits (no patient appears in two partitions)
python training/prepare_dataset.py --check-images

# 3. Train (uses the persisted splits; --synthetic-sanity validates the loop
#    on a tiny synthetic dataset with no real data needed)
python training/train.py --config training/configs/train.yaml

# 4. Per-class decision thresholds are selected on the VALIDATION set only
#    (max-F1) and frozen into best.pt automatically. To re-run standalone:
python training/select_thresholds.py --checkpoint training/checkpoints/best.pt

# 5. Evaluate on the held-out test split (AUROC/AUPRC/precision/recall/F1/
#    specificity/ECE + ROC/PR/calibration plots → artifacts/evaluation/ and
#    training/results/classification_report.json)
python training/evaluate.py --checkpoint training/checkpoints/best.pt \
    --config training/configs/train.yaml

# 6. Inspect real predictions + Grad-CAM overlays on held-out images
python training/predict_demo.py --checkpoint training/checkpoints/best.pt
```

If the dataset is not present, every script reports honestly ("...not
executed because the required dataset and/or checkpoint are unavailable")
rather than pretending. **No checkpoint ships and none is claimed.**

The inference engine scans `training/checkpoints/` (preferring `best.pt`) and
switches to **full real inference** when it finds a fine-tuned 10/14-class
head, loading the checkpoint's provenance metadata — dataset, trained vs
unavailable classes, frozen validation-selected thresholds, epoch, val AUROC,
training date — and exposing it through `/api/engine` and every `/api/predict`
response (`engineMode: real-model`, `checkpointFile`, `trainedClasses`, …).
See README-SUITE §8 for the three-tier honesty chain (fine-tuned →
backbone-live → demo).

---

## 📜 Citation & References

1. **CheXNet**: Rajpurkar et al., *"CheXNet: Radiologist-Level Chest Radiograph Diagnosis Extension with Deep Learning"*, arXiv:1711.05225.
2. **Grad-CAM**: Selvaraju et al., *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"*, IEEE ICCV 2017.
3. **NIH Dataset**: Wang et al., *"ChestX-ray14: Hospital-scale Chest X-ray Database and Benchmarks"*, IEEE CVPR 2017.

---

## 📄 License
Licensed under the [Apache License, Version 2.0](LICENSE).
