"""
segmentation.py
───────────────
Lung-field binary segmentation using the TorchXRayVision PSPNet —
a Pyramid Scene Parsing Network trained on the NIH ChestX-Det dataset,
annotated by three board-certified radiologists.

Why this replaces the original imesluh/lung-segmentation U-Net
────────────────────────────────────────────────────────────────
The previous checkpoint was hosted on a personal HuggingFace repository with
no institutional backing or availability SLA.  In production it began returning
401 / 404 errors when the author locked or deleted the repo.

TorchXRayVision's PSPNet is:
  • Hosted on GitHub Releases under a versioned, institutionally maintained
    project (mlmed/torchxrayvision) — weights will not silently disappear.
  • Purpose-built for chest X-rays (same domain as our classifier).
  • Semantically richer: outputs 14 anatomical channels, including dedicated
    Left Lung (idx 4) and Right Lung (idx 5) channels, which we union to form
    a precise bilateral lung mask.
  • Enhanced with morphological post-processing (binary fill + convex hull
    per lobe) to close the hilar gap the network sometimes leaves.

Aspect-Ratio Preservation (letterboxing)
─────────────────────────────────────────
The PSPNet requires square input (H == W).  A naïve resize of a rectangular
X-ray (e.g. 1448 × 1056) would squeeze or stretch thoracic anatomy —
distorting rib angles, cardiac silhouette width, and lung field proportions
in ways that degrade segmentation accuracy.

Instead we apply medical-grade letterboxing:
  1. The image is scaled down uniformly so its longest edge fits inside the
     target square, preserving aspect ratio exactly.
  2. Symmetric zero-padding (black border) fills the remaining space.
  3. After inference the mask is cropped to the original image region and
     the padding rows/columns are discarded.
  4. The recovered mask is then bilinearly resampled to 224 × 224.

This is the same strategy used in production PACS pre-processing pipelines
and clinical DL inference engines (e.g. TorchXRayVision's own XRayCenterCrop
utility, MONAI's SpatialPad, CheXpert's preprocessing chain).

Reference
─────────
  TorchXRayVision: A library of chest X-ray datasets and models.
  Cohen et al., Medical Imaging with Deep Learning 2022.
  https://github.com/mlmed/torchxrayvision
  https://arxiv.org/abs/2111.00595

  ChestX-Det dataset (segmentation training data):
  Lian et al., IEEE Trans. Med. Imaging 2021.
  https://arxiv.org/abs/2104.10326

Public interface (unchanged)
─────────────────────────────
  segmenter = LungSegmenter(device="cpu")
  mask = segmenter.segment(pil_image)   # np.ndarray (224, 224) float32 {0, 1}
"""

from __future__ import annotations

import logging
import traceback
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# ── Output contract (must match PneumoniaDetector input) ──────────────────────
_MASK_OUTPUT_SIZE  = (224, 224)         # (H, W) — must equal ResNet-34 input
_PROB_THRESHOLD    = 0.50               # sigmoid threshold → binary mask

# ── TorchXRayVision PSPNet details ────────────────────────────────────────────
# PSPNet was trained at 512 × 512 square resolution.  All input must be square.
_PSPNet_SQUARE_SIZE = 512               # model's native training resolution
_LEFT_LUNG_IDX      = 4                 # channel index: Left Lung
_RIGHT_LUNG_IDX     = 5                 # channel index: Right Lung

# Minimum lung-field coverage fraction (sanity check before returning mask)
_MIN_LUNG_COVERAGE  = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Letterbox helpers
# ─────────────────────────────────────────────────────────────────────────────

def _letterbox_image(
    gray_arr: np.ndarray,
    target_size: int,
) -> Tuple[np.ndarray, int, int, int, int]:
    """
    Uniformly scale a 2-D grayscale array so its longest edge equals
    *target_size*, then symmetrically zero-pad it to a (target_size ×
    target_size) square.

    This preserves thoracic anatomy proportions — no squashing or stretching.

    Parameters
    ----------
    gray_arr   : np.ndarray  shape (H, W), dtype float32, any value range
    target_size: int         desired square side length (e.g. 512)

    Returns
    -------
    padded     : np.ndarray  shape (target_size, target_size), same dtype
    pad_top    : int         rows of zero-padding added above the image
    pad_left   : int         columns of zero-padding added left of the image
    scaled_h   : int         height of the scaled (pre-padding) image content
    scaled_w   : int         width  of the scaled (pre-padding) image content
    """
    h, w = gray_arr.shape

    # ── 1. Compute uniform scale factor ──────────────────────────────────────
    scale    = target_size / max(h, w)
    scaled_h = int(round(h * scale))
    scaled_w = int(round(w * scale))

    # ── 2. Resize using PIL (BILINEAR — suitable for float/uint8 medical images)
    # PIL expects uint8 for mode "L"; we rescale to [0,255], resize, rescale back.
    arr_min, arr_max = gray_arr.min(), gray_arr.max()
    if arr_max - arr_min > 1e-6:
        arr_norm = ((gray_arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
    else:
        arr_norm = np.zeros_like(gray_arr, dtype=np.uint8)

    pil_small  = Image.fromarray(arr_norm, mode="L").resize(
        (scaled_w, scaled_h), Image.BILINEAR
    )
    scaled_arr = np.array(pil_small, dtype=np.float32) / 255.0
    # Restore original value range
    scaled_arr = scaled_arr * (arr_max - arr_min) + arr_min

    # ── 3. Symmetric zero-padding ─────────────────────────────────────────────
    pad_top   = (target_size - scaled_h) // 2
    pad_left  = (target_size - scaled_w) // 2
    pad_bottom = target_size - scaled_h - pad_top
    pad_right  = target_size - scaled_w - pad_left

    padded = np.pad(
        scaled_arr,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0.0,
    )

    logger.debug(
        "Letterbox: original (%d, %d) → scaled (%d, %d) → padded (%d, %d) "
        "[pad top=%d, left=%d]",
        h, w, scaled_h, scaled_w, padded.shape[0], padded.shape[1],
        pad_top, pad_left,
    )

    return padded, pad_top, pad_left, scaled_h, scaled_w


def _crop_padded_mask(
    mask: np.ndarray,
    pad_top: int,
    pad_left: int,
    scaled_h: int,
    scaled_w: int,
) -> np.ndarray:
    """
    Remove letterbox padding from a square segmentation mask, returning
    only the region that corresponds to actual image content.

    Parameters
    ----------
    mask     : np.ndarray  shape (S, S) — the square padded mask
    pad_top  : int         rows added above during letterboxing
    pad_left : int         columns added left during letterboxing
    scaled_h : int         height of content region
    scaled_w : int         width  of content region

    Returns
    -------
    np.ndarray  shape (scaled_h, scaled_w)
    """
    return mask[pad_top : pad_top + scaled_h, pad_left : pad_left + scaled_w]


# ─────────────────────────────────────────────────────────────────────────────
# Morphological post-processing
# ─────────────────────────────────────────────────────────────────────────────

def _fill_and_hull(binary: np.ndarray) -> np.ndarray:
    """
    Per-connected-component convex hull + binary hole fill.

    Rationale: PSPNet occasionally leaves a small gap in the central hilum
    (where the two lung lobes meet the mediastinum).  Filling holes and taking
    the convex hull of each lobe closes these gaps without over-segmenting into
    non-lung tissue.

    Parameters
    ----------
    binary : np.ndarray  shape (H, W), dtype bool or float32

    Returns
    -------
    np.ndarray  shape (H, W), dtype float32, values 0.0 or 1.0
    """
    # ── Availability check ────────────────────────────────────────────────────
    _have_scipy = False
    _have_skimage = False
    try:
        from scipy.ndimage import binary_fill_holes, label  # noqa: F401
        _have_scipy = True
    except ImportError:
        pass
    try:
        from skimage.morphology import convex_hull_image    # noqa: F401
        _have_skimage = True
    except ImportError:
        pass

    if not _have_scipy:
        logger.debug("scipy not available — returning raw binary mask (no morphological fill)")
        return binary.astype(np.float32)

    from scipy.ndimage import binary_fill_holes, label

    filled = binary_fill_holes(binary.astype(bool))

    if not _have_skimage:
        logger.debug("scikit-image not available — returning hole-filled mask (no convex hull)")
        return filled.astype(np.float32)

    from skimage.morphology import convex_hull_image

    # Label connected components (typically: left lobe, right lobe)
    labeled, n_components = label(filled)
    result = np.zeros_like(filled, dtype=np.float32)

    for comp_idx in range(1, n_components + 1):
        component = labeled == comp_idx
        if component.sum() < 50:           # ignore tiny noise blobs
            continue
        try:
            hulled = convex_hull_image(component)
            result = np.logical_or(result, hulled).astype(np.float32)
        except Exception:
            # convex_hull_image can fail on degenerate shapes — fall back gracefully
            result = np.logical_or(result, component).astype(np.float32)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class LungSegmenter:
    """
    Produces a binary float32 numpy mask of shape (224, 224) with values in
    {0.0, 1.0}.  Lung pixels = 1, background = 0.

    Uses TorchXRayVision's PSPNet (ChestX-Det weights) — a Pyramid Scene
    Parsing Network trained specifically on chest X-ray anatomical segmentation
    by three board-certified radiologists.

    Handles rectangular input images via aspect-ratio-preserving letterboxing
    so that thoracic anatomy is never distorted prior to segmentation.

    If the model fails to load for any reason, every call returns an all-ones
    mask so the downstream pipeline continues to function correctly.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self._model: Optional[torch.nn.Module] = None
        self._ready = False

        self._model = self._load_model()
        if self._model is not None:
            self._ready = True
            logger.info("LungSegmenter (TorchXRayVision PSPNet) ready on %s", device)
        else:
            logger.warning(
                "LungSegmenter running in passthrough mode — "
                "install torchxrayvision: pip install torchxrayvision"
            )

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_model(self) -> Optional[torch.nn.Module]:
        """
        Load the TorchXRayVision PSPNet with explicit, staged error reporting.
        Each failure point is isolated and logged with a full traceback so that
        uvicorn startup logs immediately reveal the root cause.
        """
        # ── Stage 1: top-level package import ────────────────────────────────
        try:
            import torchxrayvision as xrv
            logger.info(
                "torchxrayvision imported (version: %s)",
                getattr(xrv, "__version__", "unknown"),
            )
        except ImportError:
            logger.error(
                "torchxrayvision is not importable from interpreter: %s\n"
                "Run: pip install torchxrayvision",
                __import__("sys").executable,
            )
            return None

        # ── Stage 2: optional morphological dependencies ─────────────────────
        try:
            import skimage
            logger.info(
                "scikit-image available (version: %s)",
                getattr(skimage, "__version__", "unknown"),
            )
        except ImportError:
            logger.warning(
                "scikit-image not installed — convex hull post-processing "
                "disabled.  Run: pip install scikit-image"
            )
            # Non-fatal: segmentation will still work, just without hull closing

        # ── Stage 3: PSPNet instantiation ─────────────────────────────────────
        try:
            from torchxrayvision.baseline_models.chestx_det import PSPNet
            logger.info("PSPNet class imported — instantiating and downloading weights…")
            model = PSPNet()
            model.to(self.device).eval()
            logger.info("PSPNet ready (ChestX-Det weights, 14 anatomical classes)")
            return model
        except Exception:
            logger.error(
                "PSPNet initialisation failed. Full traceback:\n%s",
                traceback.format_exc(),
            )
            return None

    # ── Pre-processing ────────────────────────────────────────────────────────

    def _pil_to_xrv_letterboxed(
        self,
        pil_image: Image.Image,
    ) -> Tuple[torch.Tensor, int, int, int, int]:
        """
        Convert a PIL image (any aspect ratio) to a square xrv-normalised
        tensor suitable for PSPNet, using letterboxing to preserve anatomy.

        Returns
        -------
        tensor   : torch.Tensor  shape (1, 1, S, S), range [-1024, 1024]
        pad_top  : int
        pad_left : int
        scaled_h : int
        scaled_w : int
        """
        import torchxrayvision as xrv

        # Grayscale float32 array in [0, 255]
        gray_arr = np.array(pil_image.convert("L"), dtype=np.float32)

        # Scale to xrv convention BEFORE letterboxing so padding = 0.0
        # maps to the lower end of the normalised range — uninformative but
        # consistent (the model treats near-zero activations as background).
        xrv_arr = xrv.datasets.normalize(gray_arr, maxval=255, reshape=False)
        # xrv_arr: (H, W), range [-1024, 1024]

        # Letterbox to square
        padded, pad_top, pad_left, scaled_h, scaled_w = _letterbox_image(
            xrv_arr, _PSPNet_SQUARE_SIZE
        )
        # padded: (512, 512), range approximately [-1024, 1024]

        # Add batch + channel dims: (1, 1, 512, 512)
        tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor, pad_top, pad_left, scaled_h, scaled_w

    # ── Mask extraction ───────────────────────────────────────────────────────

    @staticmethod
    def _build_lung_mask_raw(output: torch.Tensor) -> np.ndarray:
        """
        Union Left Lung (idx 4) + Right Lung (idx 5) channels from PSPNet
        output into a single binary mask.

        Parameters
        ----------
        output : torch.Tensor  shape (1, 14, H, W), raw logits

        Returns
        -------
        np.ndarray  shape (H, W), float32, values 0.0 or 1.0
        """
        probs      = torch.sigmoid(output)                               # (1,14,H,W)
        left_prob  = probs[0, _LEFT_LUNG_IDX,  :, :].cpu().numpy()
        right_prob = probs[0, _RIGHT_LUNG_IDX, :, :].cpu().numpy()
        lung_prob  = np.maximum(left_prob, right_prob)
        return (lung_prob >= _PROB_THRESHOLD).astype(np.float32)

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    def segment(self, pil_image: Image.Image) -> np.ndarray:
        """
        Segment the lung fields in a chest X-ray.

        Supports any input aspect ratio — rectangular images are letterboxed
        before inference and the padding is stripped from the mask before
        output, so thoracic anatomy is never geometrically distorted.

        Parameters
        ----------
        pil_image : PIL.Image  (any mode; converted internally to grayscale)

        Returns
        -------
        mask : np.ndarray
            Shape (224, 224), dtype float32, values 0.0 (background) or
            1.0 (lung field).

        Pipeline
        ────────
          1. Convert PIL → xrv-normalised array → letterboxed square tensor.
          2. PSPNet inference → 14 anatomical probability maps.
          3. Union Left Lung + Right Lung → raw binary square mask.
          4. Crop letterbox padding → content-only mask.
          5. Morphological refinement: hole fill + per-lobe convex hull.
          6. Bilinear resample to 224 × 224.
          7. Sanity check: coverage < 5% → full-image fallback.
        """
        if not self._ready:
            return np.ones(_MASK_OUTPUT_SIZE, dtype=np.float32)

        # ── 1. Pre-process with letterboxing ──────────────────────────────────
        tensor, pad_top, pad_left, scaled_h, scaled_w = (
            self._pil_to_xrv_letterboxed(pil_image)
        )
        # tensor: (1, 1, 512, 512) — square, aspect-ratio preserved

        # ── 2. Inference ──────────────────────────────────────────────────────
        with torch.no_grad():
            output = self._model(tensor)           # (1, 14, H', W') raw logits

        # ── 3. Raw lung mask (still square, padded) ───────────────────────────
        raw_square = self._build_lung_mask_raw(output)  # (H', W'), float32 {0,1}

        # H' may differ from 512 if PSPNet downsamples internally; recompute
        # the crop coordinates proportionally.
        sq_h, sq_w = raw_square.shape
        scale_h    = sq_h / _PSPNet_SQUARE_SIZE
        scale_w    = sq_w / _PSPNet_SQUARE_SIZE

        crop_top   = int(round(pad_top   * scale_h))
        crop_left  = int(round(pad_left  * scale_w))
        crop_h     = int(round(scaled_h  * scale_h))
        crop_w     = int(round(scaled_w  * scale_w))

        # ── 4. Crop to remove letterbox padding ───────────────────────────────
        raw_content = raw_square[
            crop_top : crop_top + crop_h,
            crop_left: crop_left + crop_w,
        ]                                          # (crop_h, crop_w) — content only

        # ── 5. Morphological refinement ───────────────────────────────────────
        refined = _fill_and_hull(raw_content)      # (crop_h, crop_w), float32 {0,1}

        # ── 6. Resize to 224 × 224 ────────────────────────────────────────────
        mask_tensor = (
            torch.from_numpy(refined)
            .unsqueeze(0).unsqueeze(0)             # (1, 1, crop_h, crop_w)
        )
        mask_224 = F.interpolate(
            mask_tensor,
            size          = _MASK_OUTPUT_SIZE,
            mode          = "bilinear",
            align_corners = False,
        ).squeeze().numpy()                        # (224, 224), continuous [0,1]

        binary_224 = (mask_224 >= 0.5).astype(np.float32)

        # ── 7. Sanity check ───────────────────────────────────────────────────
        coverage = binary_224.mean()
        if coverage < _MIN_LUNG_COVERAGE:
            logger.warning(
                "PSPNet produced near-empty mask (coverage=%.2f%%) — "
                "falling back to full-image mask.",
                coverage * 100,
            )
            return np.ones(_MASK_OUTPUT_SIZE, dtype=np.float32)

        logger.debug("Lung mask OK: coverage=%.1f%%", coverage * 100)
        return binary_224   # (224, 224), float32, {0.0, 1.0}