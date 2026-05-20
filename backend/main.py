"""
main.py
───────
FastAPI inference server for PneumoScan AI.

Pipeline (per request):
  1. Decode uploaded file → PIL RGB image
     • Handles JPEG, PNG, and DICOM (via pydicom → pixel array → PIL)
  2. LungSegmenter  → binary mask (224 × 224)
     • Handles any input aspect ratio via aspect-ratio-preserving letterboxing
  3. PneumoniaDetector.preprocess(image, lung_mask)
     → masked & normalised (1, 3, 224, 224) tensor
  4. PneumoniaDetector.predict()  → class + probability
  5. PneumoniaDetector.compute_gradcam(…, lung_mask)
     → heatmap masked to lung fields only

Response contract (unchanged):
  { "diagnosis": "Pneumonia"|"Normal",
    "probability": float,
    "heatmap": [[float, …], …]   # 224 × 224 nested list
  }

Both models are loaded once at startup and kept in memory.
"""

from __future__ import annotations

import io
import logging

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from classifier   import PneumoniaDetector
from segmentation import LungSegmenter

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="PneumoScan AI — Pneumonia Detection API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins    = ["http://localhost:5173", "http://127.0.0.1:5173",],
    allow_credentials= True,
    allow_methods    = ["*"],
    allow_headers    = ["*"],
)

# ── Model singletons (populated at startup) ───────────────────────────────────
MODEL_PATH = "resnet34_pneumonia.pth"
detector  : PneumoniaDetector | None = None
segmenter : LungSegmenter     | None = None


@app.on_event("startup")
def load_models() -> None:
    global detector, segmenter

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # ── 1. Classifier ─────────────────────────────────────────────────────────
    logger.info("Loading PneumoniaDetector from %s …", MODEL_PATH)
    detector = PneumoniaDetector(model_path=MODEL_PATH, device=device)
    logger.info("PneumoniaDetector ready.")

    # ── 2. Lung segmenter (graceful — falls back to passthrough if unavailable) ─
    logger.info("Loading LungSegmenter …")
    segmenter = LungSegmenter(device=device)
    if segmenter.is_ready:
        logger.info(
            "LungSegmenter ready — lung-field isolation active "
            "(aspect-ratio-preserving letterboxing enabled)."
        )
    else:
        logger.warning(
            "LungSegmenter NOT available — running without lung-field isolation. "
            "Run: pip install torchxrayvision scikit-image"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────
def _decode_image(raw_bytes: bytes, content_type: str) -> Image.Image:
    """
    Decode raw bytes to a PIL RGB Image.

    Supports:
      • Standard images (JPEG, PNG, BMP, TIFF, WebP) via PIL
      • DICOM (.dcm) via pydicom — content_type "application/dicom" or
        "application/octet-stream" with a DICOM magic header
    """
    is_dicom = (
        content_type in ("application/dicom", "application/octet-stream")
        or raw_bytes[:4] == b"\x00\x00\x00\x00"
        or raw_bytes[128:132] == b"DICM"
    )

    if is_dicom:
        try:
            import pydicom
            from pydicom.pixel_data_handlers.util import apply_voi_lut
            ds  = pydicom.dcmread(io.BytesIO(raw_bytes))
            arr = apply_voi_lut(ds.pixel_array.astype(np.float32), ds)
            arr = arr - arr.min()
            peak = arr.max()
            if peak > 0:
                arr = arr / peak
            arr = (arr * 255).astype(np.uint8)
            if arr.ndim == 2:
                pil = Image.fromarray(arr, mode="L").convert("RGB")
            else:
                pil = Image.fromarray(arr).convert("RGB")
            logger.info("DICOM decoded: shape=%s", arr.shape)
            return pil
        except ImportError:
            logger.warning("pydicom not installed — attempting PIL fallback for DICOM bytes")
        except Exception as exc:
            logger.warning("DICOM decode failed (%s) — attempting PIL fallback", exc)

    try:
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unrecognised image format. "
                f"Supported: JPEG, PNG, BMP, TIFF, WebP, DICOM. ({exc})"
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image decode error: {exc}")


def _heatmap_to_list(heatmap: np.ndarray) -> list:
    """Round to 4 dp and serialise a 2-D float array to a JSON-safe nested list."""
    return np.round(heatmap.astype(np.float64), 4).tolist()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict:
    return {
        "status"          : "ok",
        "classifier_ready": detector  is not None,
        "segmenter_ready" : segmenter is not None and segmenter.is_ready,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """
    Accepts a chest X-ray (PNG / JPEG / DICOM), any aspect ratio.

    Returns
    -------
    {
        "diagnosis":   "Pneumonia" | "Normal",
        "probability": float,
        "heatmap":     [[float, …], …]   # 224×224 nested list, values 0–1
    }
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded — try again.")

    ct = file.content_type or ""
    if not (ct.startswith("image/") or ct in ("application/dicom", "application/octet-stream")):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported content-type '{ct}'. "
                f"Send image/jpeg, image/png, or application/dicom."
            ),
        )

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    image = _decode_image(raw, ct)
    logger.info("Image decoded: size=%s, mode=%s", image.size, image.mode)

    # ── Step 1 — Lung segmentation ────────────────────────────────────────────
    # LungSegmenter.segment() handles any aspect ratio internally via
    # letterboxing — no pre-resize needed here.
    lung_mask: np.ndarray | None = None
    if segmenter is not None:
        try:
            lung_mask = segmenter.segment(image)       # (224, 224) float32 {0,1}
            coverage  = lung_mask.mean() * 100
            logger.info("Lung mask computed. Coverage: %.1f%%", coverage)
        except Exception as exc:
            logger.warning(
                "Segmentation failed (%s) — proceeding without mask", exc
            )
            lung_mask = None

    # ── Step 2 — Pre-process (masking + normalisation) ────────────────────────
    input_tensor = detector.preprocess(image, lung_mask=lung_mask)

    # ── Step 3 — Classification ───────────────────────────────────────────────
    pred_class, prob_pneumonia, _ = detector.predict(input_tensor)
    diagnosis = "Pneumonia" if pred_class == 1 else "Normal"
    logger.info("Prediction: %s (p=%.4f)", diagnosis, prob_pneumonia)

    # ── Step 4 — Grad-CAM ────────────────────────────────────────────────────
    heatmap = detector.compute_gradcam(
        input_tensor,
        class_idx = 1,
        lung_mask = lung_mask,
    )

    return {
        "diagnosis"  : diagnosis,
        "probability": round(float(prob_pneumonia), 4),
        "heatmap"    : _heatmap_to_list(heatmap),
    }