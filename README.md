# PneumoScan AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61dafb?style=flat-square&logo=react&logoColor=black)
![TorchXRayVision](https://img.shields.io/badge/TorchXRayVision-1.0.1+-blueviolet?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

**A production-grade, full-stack medical AI pipeline for pneumonia detection in chest X-rays.**

Engineered to eliminate shortcut learning bias via domain-specific anatomical segmentation — not just a ResNet wrapper.

[Architecture](#architecture) · [The Problem We Solved](#the-shortcut-learning-problem--our-solution) · [Features](#features) · [Setup](#local-setup--installation) · [Evaluation](#evaluation--ood-testing)

</div>

---

## What This Is

PneumoScan AI is a full-stack, inference-ready pipeline for binary pneumonia classification on chest X-rays. It is built around a fine-tuned ResNet-34 classifier, but the engineering effort that differentiates it from a standard classification project lies in the upstream domain-specific preprocessing: **Lung Field Isolation via a TorchXRayVision PSPNet segmenter**.

The system ships a React/Vite diagnostic workstation UI with radiology-grade UX features (DICOM windowing simulation, Grad-CAM overlay controls, radiology crosshairs), a FastAPI inference server, and a dedicated out-of-distribution evaluation harness.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                              │
│                                                                        │
│  Input Image (any aspect ratio — JPEG / PNG / DICOM)                  │
│       │                                                                │
│       ▼                                                                │
│  ┌─────────────────────────────────────────────────────┐              │
│  │  STAGE 1 · LUNG FIELD ISOLATION  (segmentation.py)  │              │
│  │                                                      │              │
│  │  1. Aspect-ratio-preserving letterbox  →  512×512   │              │
│  │  2. TorchXRayVision PSPNet (ChestX-Det weights)     │              │
│  │     14-class anatomical segmentation map            │              │
│  │  3. Union(Left Lung ch.4, Right Lung ch.5)          │              │
│  │  4. Crop letterbox padding from mask                │              │
│  │  5. Morphological refinement:                       │              │
│  │       binary_fill_holes  →  convex_hull per lobe    │              │
│  │  6. Bilinear resize  →  224×224 binary float mask   │              │
│  └───────────────────────┬─────────────────────────────┘              │
│                          │  mask (224×224) {0,1}                      │
│                          ▼                                             │
│  ┌─────────────────────────────────────────────────────┐              │
│  │  STAGE 2 · MASKED CLASSIFICATION  (classifier.py)   │              │
│  │                                                      │              │
│  │  1. Resize input image  →  224×224 float tensor     │              │
│  │  2. Apply lung mask BEFORE ImageNet normalisation:  │              │
│  │       background pixels → channel mean              │              │
│  │       (maps to exactly 0.0 post-normalisation)      │              │
│  │  3. ResNet-34 forward pass  →  softmax(2)           │              │
│  └───────────────────────┬─────────────────────────────┘              │
│                          │  diagnosis + probability                    │
│                          ▼                                             │
│  ┌─────────────────────────────────────────────────────┐              │
│  │  STAGE 3 · EXPLAINABILITY  (classifier.py)          │              │
│  │                                                      │              │
│  │  1. GradCAM on layer4[-1].conv2                     │              │
│  │  2. Bilinear upscale  →  224×224                    │              │
│  │  3. Min-max normalise                               │              │
│  │  4. Multiply by lung mask → zero non-lung regions   │              │
│  │  5. Re-normalise  →  heatmap [0,1]                  │              │
│  └───────────────────────┬─────────────────────────────┘              │
│                          │                                             │
│                          ▼                                             │
│  { diagnosis, probability, heatmap[224][224] }  →  FastAPI Response   │
└────────────────────────────────────────────────────────────────────────┘
```

**Tech stack:** Python 3.10 · PyTorch 2.2 · FastAPI · TorchXRayVision · scikit-image · scipy · React 18 · Vite · TailwindCSS

---

## The Shortcut Learning Problem & Our Solution

### The Problem

A standard ResNet-34 trained end-to-end on chest X-ray datasets is vulnerable to **shortcut learning** — a form of confounding bias where the model learns to classify based on spurious correlations in the training set rather than the pathology of interest.

Auditing our model's Grad-CAM activations revealed it was attending heavily to:
- **Cardiac silhouette size** (cardiomegaly is correlated with severe pneumonia in many datasets)
- **Endotracheal and nasogastric tube artifacts** (intubated patients → ICU → pneumonia)
- **Image border brightness** and **AP vs PA acquisition tag artifacts** (acquisition protocol correlates with disease severity)

These are legitimate statistical signals in the training data but they are not generalizable clinical indicators of pulmonary parenchymal consolidation — the actual radiological finding for pneumonia. A model learning these features will degrade catastrophically on out-of-distribution images.

### Our Solution: Domain-Specific Ensemble with Lung Field Isolation

We architected a two-model ensemble where the segmenter acts as a hard prior over the classifier's visual field.

**Step 1 — Anatomical Segmentation**

We replaced the broken personal-repo U-Net dependency with [`torchxrayvision`](https://github.com/mlmed/torchxrayvision)'s PSPNet: a Pyramid Scene Parsing Network trained on the NIH ChestX-Det dataset, annotated by three board-certified radiologists, with weights hosted on GitHub Releases under institutional maintenance (Cohen et al., MIDL 2022).

Unlike a generic binary lung segmenter, the PSPNet outputs **14 separate anatomical channels** — including dedicated `Left Lung` (channel 4) and `Right Lung` (channel 5) masks. We union these two channels to produce a bilateral lung mask with no mediastinal leakage.

**Step 2 — Morphological Refinement**

The raw PSPNet mask is post-processed with:
- `scipy.ndimage.binary_fill_holes` — closes the hilar gap the network sometimes leaves between the two lobes
- `skimage.morphology.convex_hull_image` per connected component — convexifies each lobe independently, preventing the jagged lobe edges from leaking out-of-distribution background pixels to the classifier

**Step 3 — Mask-Gated Preprocessing**

The binary lung mask is applied **before** ImageNet normalization in the classifier's `preprocess()` method. Background pixels are set to each channel's ImageNet mean value, which maps to exactly **0.0** after normalization — the cleanest possible neutral value for a pre-trained backbone. The classifier's gradient signal is thus entirely confined to lung parenchyma.

**Step 4 — Mask-Gated Grad-CAM**

The final Grad-CAM heatmap is multiplied element-wise by the same lung mask, zeroing any activation outside the lung fields. This ensures the explainability output is also anatomically constrained.

The result is a model that cannot attend to cardiac anatomy, tubes, or image borders during inference — by construction.

---

## Features

### Backend (FastAPI / PyTorch)

- **Lung Field Isolation** — TorchXRayVision PSPNet with 14-class anatomical output, Left+Right lung union, morphological post-processing
- **Aspect-Ratio-Preserving Preprocessing** — Rectangular X-rays (e.g. 1448×1056) are letterboxed (uniform scale + symmetric zero-padding) before segmentation. Thoracic anatomy is never distorted. The padding is cropped from the mask before output. See `_letterbox_image()` in `segmentation.py`
- **Grad-CAM Explainability** — `torchcam` GradCAM on ResNet-34 `layer4[-1].conv2`, lung-mask-gated, re-normalised to [0,1]
- **DICOM Support** — `pydicom` with `apply_voi_lut` for proper DICOM windowing; graceful fallback to PIL for standard image formats
- **Graceful Degradation** — If the segmenter fails to load, the pipeline continues in passthrough mode. The `/health` endpoint exposes `segmenter_ready` so the frontend can surface this to the radiologist
- **Production-Grade Logging** — Staged, isolated exception handling in `_load_model()` with `traceback.format_exc()` to surface root causes immediately in uvicorn startup logs

### Frontend (React / Vite / TailwindCSS)

- **Radiology Diagnostic Workstation UI** — Dark-mode PACS-style layout with sidebar navigation, patient demographics panel, and model information panel
- **DICOM Windowing Simulation** — Real-time `contrast()` CSS filter applied to the canvas, adjustable via slider (0.5× – 3.0×), simulating radiologist window/level adjustment
- **Progressive Grad-CAM Reveal** — 4-second rAF animation where the heatmap threshold decreases from 0.9→0 and opacity increases from 0.15→0.85, creating a cinematic activation map materialisation effect
- **Radiology Crosshairs** — SVG overlay tracks mouse position over the canvas with coordinate readout in JetBrains Mono, zero-cost (pointer-events: none)
- **Lung Isolation HUD Badge** — Live indicator in the canvas HUD showing `SEG: LUNG-ISO ✓` or `SEG: PASSTHROUGH` based on the `/health` endpoint response, so the radiologist always knows whether bias-reduction is active
- **Stale Closure Fix** — `drawCanvasRef` pattern ensures the rAF animation loop always calls the latest `drawCanvas` closure, preventing stale `showGradCam`, `heatmapOpacity`, `windowContrast` values mid-animation
- **Toast Notification System** — Enter/leave animated toasts for API errors, tool activations, and analysis completion
- **OOD Distribution Warning** — Persistent, non-intrusive footer disclaimer about the NIH ChestX-14 training distribution

---

## Local Setup & Installation

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- A trained ResNet-34 checkpoint named `resnet34_pneumonia.pth` in the backend directory

### Backend

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies with strict pins
#
#    CRITICAL: numpy<2.0.0 is required.
#    torchcam and scikit-image both link against numpy C extensions. numpy 2.x
#    introduced ABI-breaking changes that cause diamond dependency conflicts
#    at import time. Pin to <2.0.0 to avoid runtime ImportErrors.
#
#    CRITICAL: tifffile<2024.1.0 is required.
#    scikit-image's internal tifffile dependency shipped a breaking API change
#    in 2024.1.0 that conflicts with the version torchxrayvision expects.
#    Pin it explicitly to prevent silent failures.
#
pip install "numpy<2.0.0"
pip install "tifffile<2024.1.0"
pip install torch>=2.2.0 torchvision>=0.17.0
pip install fastapi>=0.111.0 "uvicorn[standard]>=0.29.0" python-multipart>=0.0.9
pip install torchcam>=0.4.0
pip install torchxrayvision>=1.0.1
pip install "scikit-image>=0.22.0" "scipy>=1.13.0"
pip install "Pillow>=10.3.0" "pydicom>=2.4.0"

# 3. IMPORTANT — file naming (namespace shadowing):
#    The classifier is in classifier.py, NOT model.py.
#    torchxrayvision imports a module named `model` internally. If your
#    classifier file is also named model.py, Python's import system will find
#    your local file first and raise an AttributeError or ImportError deep
#    inside torchxrayvision. Keep the file named classifier.py.

# 4. Place your trained weights in the backend directory
cp /path/to/your/resnet34_pneumonia.pth .

# 5. Start the server
#    TorchXRayVision will download the PSPNet weights (~100MB) to
#    ~/.torchxrayvision on first run. Subsequent starts are instant.
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Verify startup — you should see all three of these lines in the uvicorn log:

```
INFO  torchxrayvision imported (version: x.x.x)
INFO  scikit-image available (version: x.x.x)
INFO  PSPNet ready (ChestX-Det weights, 14 anatomical classes)
INFO  LungSegmenter ready — lung-field isolation active (aspect-ratio-preserving letterboxing enabled).
```

If you see `LungSegmenter running in passthrough mode` instead, check the `ERROR` lines immediately above it — the staged logger will print the exact `traceback.format_exc()` output.

Confirm the pipeline is healthy:

```bash
curl http://localhost:8000/health
# {"status":"ok","classifier_ready":true,"segmenter_ready":true}
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## Evaluation & OOD Testing

The `evaluate_pipeline.py` script provides a systematic out-of-distribution (OOD) evaluation harness against the Kaggle Chest X-Ray Images (Pneumonia) test set — a dataset that differs in acquisition protocol and patient demographics from the NIH ChestX-14 training distribution.

### Setup

1. Download the [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset
2. Extract the `test/` split to a local directory
3. Edit the `DATASET_DIR` path in `evaluate_pipeline.py`
4. Ensure the backend server is running (`uvicorn main:app`)

### Run

```bash
python evaluate_pipeline.py
```

The script evaluates every `.jpeg`/`.jpg` in the `NORMAL/` and `PNEUMONIA/` subdirectories against the live `/predict` endpoint and reports:

```
========================================
📊 PIPELINE EVALUATION REPORT
========================================
Total Images Tested : 624
Accuracy            : 94.87%
False Positives     : 12  (Told healthy patient they are sick)
False Negatives     : 20  (Told sick patient they are healthy!)
Time Taken          : 148.32 seconds
Avg Inference Time  : 0.24 sec/image
========================================
```

False negatives (missed pneumonia) are the clinically critical failure mode. Use this report to tune the classification threshold before any clinical validation exercise.

---

## Project Structure

```
pneumoscan-ai/
├── backend/
│   ├── main.py              # FastAPI server, /predict and /health endpoints
│   ├── classifier.py        # PneumoniaDetector: ResNet-34 + Grad-CAM
│   │                        #   NOTE: named classifier.py, NOT model.py
│   │                        #   (avoids namespace collision with torchxrayvision)
│   ├── segmentation.py      # LungSegmenter: PSPNet + letterboxing + morphology
│   ├── evaluate_pipeline.py # OOD evaluation harness
│   └── resnet34_pneumonia.pth  # trained weights (not committed)
├── frontend/
│   └── src/
│       └── ChestXRayAnalyzer.jsx  # Full diagnostic workstation UI
└── README.md
```

---

## Dependency Reference

| Package | Version Pin | Reason |
|---|---|---|
| `numpy` | `<2.0.0` | **Required.** ABI-breaking changes in 2.x cause diamond dependency conflicts between `torchcam` and `scikit-image` at import time |
| `tifffile` | `<2024.1.0` | **Required.** API breaking change in 2024.1.0 conflicts with `torchxrayvision`'s internal expectations |
| `torchxrayvision` | `>=1.0.1` | PSPNet segmenter — institutionally maintained, GitHub Releases weights |
| `scikit-image` | `>=0.22.0` | `convex_hull_image` for morphological lobe refinement |
| `scipy` | `>=1.13.0` | `binary_fill_holes`, `ndimage.label` |
| `torchcam` | `>=0.4.0` | GradCAM implementation |
| `pydicom` | `>=2.4.0` | DICOM image decoding with VOI LUT application |

---

## References

```bibtex
@inproceedings{Cohen2022xrv,
  title   = {TorchXRayVision: A library of chest X-ray datasets and models},
  author  = {Cohen, Joseph Paul and Viviano, Joseph D. and Bertin, Paul and
             Morrison, Paul and Torabian, Parsa and Guarrera, Matteo and
             Lungren, Matthew P and Chaudhari, Akshay and Brooks, Rupert and
             Hashir, Mohammad and Bertrand, Hadrien},
  booktitle = {Medical Imaging with Deep Learning},
  year    = {2022},
  url     = {https://github.com/mlmed/torchxrayvision}
}

@article{Lian2021chestxdet,
  title   = {A Structure-Aware Relation Network for Thoracic Diseases
             Detection and Segmentation},
  author  = {Lian, Jie and Liu, Jingyu and Zhang, Shu and Gao, Kai and
             Liu, Xiaoqing and Zhang, Dingwen and Yu, Yizhou},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2021},
  doi     = {10.48550/arxiv.2104.10326}
}

@inproceedings{Selvaraju2017gradcam,
  title   = {Grad-CAM: Visual Explanations from Deep Networks via
             Gradient-Based Localization},
  author  = {Selvaraju, Ramprasath R. and Cogswell, Michael and Das, Abhishek
             and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  booktitle = {ICCV},
  year    = {2017}
}
```

---

> **Clinical Disclaimer:** This system is a research prototype and has not been validated for clinical use. All outputs must be reviewed and confirmed by a licensed radiologist before informing any clinical decision. The model was trained on NIH ChestX-14; images acquired under different equipment, patient positioning, or exposure protocols may fall outside the training distribution.
