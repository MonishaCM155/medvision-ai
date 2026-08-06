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
- 🔍 **Interactive Grad-CAM Heatmaps**: Dual-view overlay with adjustable heat opacity, colormap selection (Jet, Turbo, Viridis, Inferno), and focal Region of Interest (ROI) bounding boxes.
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
- **Backend**: Express (Node.js) / FastAPI (Python 3.10), Uvicorn, Pydantic, SQLAlchemy.
- **Machine Learning & Vision**: PyTorch 2.2, TorchVision, OpenCV, Albumentations, Captum, Grad-CAM, ONNX Runtime.
- **LLM Integration**: `@google/genai` (Gemini 3.6 Flash) server-side proxy.
- **DevOps & MLOps**: Docker, Docker Compose, MLflow, GitHub Actions CI.

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

To train the PyTorch DenseNet-121 backbone on the NIH ChestX-ray14 dataset:

```bash
python training/train.py \
    --data_dir ./datasets/nih_chestxray \
    --architecture densenet121 \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --loss_fn weighted_bce \
    --amp_fp16
```

---

## 📜 Citation & References

1. **CheXNet**: Rajpurkar et al., *"CheXNet: Radiologist-Level Chest Radiograph Diagnosis Extension with Deep Learning"*, arXiv:1711.05225.
2. **Grad-CAM**: Selvaraju et al., *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"*, IEEE ICCV 2017.
3. **NIH Dataset**: Wang et al., *"ChestX-ray14: Hospital-scale Chest X-ray Database and Benchmarks"*, IEEE CVPR 2017.

---

## 📄 License
Licensed under the [Apache License, Version 2.0](LICENSE).
