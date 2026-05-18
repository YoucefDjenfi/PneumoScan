# PneumoScan AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61dafb?style=flat-square&logo=react&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

**A full-stack medical AI pipeline for pneumonia detection in chest X-rays.**

Fine-tuned ResNet-34 classifier · Grad-CAM explainability · PACS-style diagnostic workstation UI

[Architecture](#architecture) · [Features](#features) · [Setup](#local-setup--installation) · [Evaluation](#evaluation) · [Project Structure](#project-structure)

</div>

---

## Overview

PneumoScan AI is an end-to-end, inference-ready system for binary pneumonia classification on chest X-rays. The backend exposes a low-latency FastAPI inference server powered by a fine-tuned ResNet-34 classifier with Grad-CAM explainability. The frontend is a React/Vite diagnostic workstation — a dark-mode, PACS-style UI with DICOM windowing simulation, animated heatmap overlays, and radiology-grade UX conventions.

The system is designed to be cloned, stood up locally in under five minutes, and extended cleanly. Every component — model loading, inference, explainability, and the health endpoint — is isolated and independently testable.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE PIPELINE                             │
│                                                                        │
│  Input Image  (JPEG / PNG / DICOM)                                     │
│       │                                                                │
│       ▼                                                                │
│  ┌─────────────────────────────────────────────────────┐              │
│  │  STAGE 1 · PREPROCESSING  (classifier.py)           │              │
│  │                                                      │              │
│  │  1. DICOM decode (pydicom + VOI LUT) or PIL fallback│              │
│  │  2. Resize  →  224×224 float tensor                 │              │
│  │  3. ImageNet normalisation                          │              │
│  └───────────────────────┬─────────────────────────────┘              │
│                          │  tensor [1, 3, 224, 224]                   │
│                          ▼                                             │
│  ┌─────────────────────────────────────────────────────┐              │
│  │  STAGE 2 · CLASSIFICATION  (classifier.py)          │              │
│  │                                                      │              │
│  │  ResNet-34  (fine-tuned, binary head)               │              │
│  │  forward pass  →  softmax(2)                        │              │
│  │  → diagnosis + confidence probability               │              │
│  └───────────────────────┬─────────────────────────────┘              │
│                          │                                             │
│                          ▼                                             │
│  ┌─────────────────────────────────────────────────────┐              │
│  │  STAGE 3 · EXPLAINABILITY  (classifier.py)          │              │
│  │                                                      │              │
│  │  GradCAM on layer4[-1].conv2                        │              │
│  │  Bilinear upscale  →  224×224                       │              │
│  │  Min-max normalise  →  heatmap [0, 1]               │              │
│  └───────────────────────┬─────────────────────────────┘              │
│                          │                                             │
│                          ▼                                             │
│  { diagnosis, probability, heatmap[224][224] }  →  FastAPI Response   │
└────────────────────────────────────────────────────────────────────────┘
```

**Tech stack:** Python 3.10 · PyTorch 2.2 · FastAPI · torchcam · pydicom · React 18 · Vite · TailwindCSS

---

## Features

### Backend (FastAPI / PyTorch)

- **Fine-Tuned ResNet-34 Classifier** — Binary pneumonia classifier with a custom head, trained on NIH ChestX-14. Weights load at startup; the `/health` endpoint exposes `classifier_ready` so clients can gate on model availability before sending inference requests
- **Grad-CAM Explainability** — GradCAM computed on `layer4[-1].conv2` via `torchcam`, bilinearly upscaled to 224×224 and min-max normalised to `[0, 1]`. Every prediction ships with a heatmap — no separate explainability call required
- **DICOM Support** — `pydicom` with `apply_voi_lut` for correct DICOM windowing; transparent fallback to PIL for JPEG/PNG inputs
- **Sub-second Inference** — Single forward pass + GradCAM averages ~0.24 s/image on CPU. No preprocessing bottlenecks
- **Clean API Surface** — Two endpoints: `POST /predict` (multipart file upload → JSON diagnosis + base64 heatmap) and `GET /health` (liveness + model readiness). Fully documented via FastAPI's auto-generated `/docs`
- **Structured Logging** — Isolated exception handling in `_load_model()` with `traceback.format_exc()` output, surfaced immediately in uvicorn startup logs for rapid diagnosis

### Frontend (React / Vite / TailwindCSS)

- **PACS-Style Diagnostic Workstation** — Dark-mode layout modelled on clinical PACS viewers, with a sidebar navigator, patient demographics panel, and model information panel. Designed to feel like software radiologists already know
- **DICOM Windowing Simulation** — Real-time `contrast()` CSS filter on the image canvas, adjustable via slider (0.5× – 3.0×), simulating the window/level control radiologists use to tune X-ray density visualisation
- **Animated Grad-CAM Reveal** — 4-second `requestAnimationFrame` animation where the heatmap threshold decreases from `0.9 → 0` and opacity increases from `0.15 → 0.85`, progressively materialising the activation map over the image
- **Radiology Crosshairs** — SVG overlay tracking mouse position across the diagnostic canvas with pixel-coordinate readout in `JetBrains Mono`. Implemented as a `pointer-events: none` layer — zero interaction cost
- **Robust Animation State** — `drawCanvasRef` pattern ensures the rAF loop always closes over the latest `drawCanvas` instance, preventing stale `showGradCam`, `heatmapOpacity`, and `windowContrast` values during multi-frame animations

---

## Local Setup & Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- Your trained model weights: `resnet34_pneumonia.pth`

### Backend

```bash
# 1. Clone the repository
git clone https://github.com/your-username/pneumoscan-ai.git
cd pneumoscan-ai/backend

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your model weights in the backend directory
cp /path/to/resnet34_pneumonia.pth .

# 5. Start the inference server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Confirm the server is healthy:

```bash
curl http://localhost:8000/health
# {"status":"ok","classifier_ready":true}
```

The interactive API docs are available at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

Open `http://localhost:5173` in your browser. The workstation connects to the backend at `http://localhost:8000` by default. Upload a chest X-ray (JPEG, PNG, or DICOM) and click **Analyze** to run inference.

---

## Evaluation

The `evaluate_pipeline.py` script runs a systematic batch evaluation against the Kaggle Chest X-Ray Images (Pneumonia) test set.

### Setup

1. Download the [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset
2. Extract the `test/` split to a local directory
3. Set the `DATASET_DIR` path in `evaluate_pipeline.py`
4. Ensure the backend server is running

### Run

```bash
python evaluate_pipeline.py
```

Example output:

```
========================================
📊 PIPELINE EVALUATION REPORT
========================================
Total Images Tested : 624
Accuracy            : 94.87%
False Positives     : 12  (Healthy flagged as pneumonia)
False Negatives     : 20  (Pneumonia missed)
Time Taken          : 148.32 seconds
Avg Inference Time  : 0.24 sec/image
========================================
```

False negatives (missed pneumonia) are the clinically critical failure mode. Use this report to tune the classification threshold before any downstream validation.

---

## Project Structure

```
pneumoscan-ai/
├── backend/
│   ├── main.py                   # FastAPI server — /predict and /health endpoints
│   ├── classifier.py             # PneumoniaDetector: ResNet-34 forward pass + GradCAM
│   ├── evaluate_pipeline.py      # Batch evaluation harness
│   └── resnet34_pneumonia.pth    # Trained weights (not committed — add your own)
├── frontend/
│   └── src/
│       └── ChestXRayAnalyzer.jsx # Full diagnostic workstation UI component
└── README.md
```

---

## Dependency Reference

| Package | Version | Role |
|---|---|---|
| `torch` | `>=2.2.0` | Model runtime and autograd |
| `torchvision` | `>=0.17.0` | ResNet-34 architecture and pretrained weights |
| `torchcam` | `>=0.4.0` | GradCAM implementation |
| `fastapi` | `>=0.111.0` | Inference API server |
| `uvicorn` | `>=0.29.0` | ASGI server |
| `pydicom` | `>=2.4.0` | DICOM image decoding with VOI LUT |
| `Pillow` | `>=10.0.0` | JPEG/PNG fallback image loading |
| `numpy` | `<2.0.0` | Array operations — pinned below 2.x to avoid ABI conflicts with `torchcam` |

---

> **Clinical Disclaimer:** PneumoScan AI is a research prototype and has not been validated for clinical use. All outputs must be reviewed and confirmed by a licensed radiologist before informing any clinical decision. The model was trained on NIH ChestX-14; performance may vary on images acquired under different equipment, patient positioning, or exposure protocols.
