"""
model.py
────────
PneumoniaDetector  — ResNet-34 classifier + Grad-CAM.

Changes from the original:
  • preprocess() now accepts an optional lung_mask argument (np.ndarray 224×224).
    When supplied, the lung-field isolation is applied *inside* the tensor
    preparation step (before normalisation) so the backbone never sees
    out-of-distribution background pixels.
  • compute_gradcam() accepts the same optional mask and multiplies the final
    heatmap by it, zeroing activations outside the lung fields.
  • Everything else is unchanged.
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchcam.methods import GradCAM

logger = logging.getLogger(__name__)


class PneumoniaDetector:
    """
    Wraps a trained ResNet-34 for pneumonia detection (2 classes: Normal, Pneumonia)
    and provides Grad-CAM heatmaps.
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)

        self.model = models.resnet34(weights=None)
        num_ftrs        = self.model.fc.in_features
        self.model.fc   = nn.Linear(num_ftrs, 2)

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # ImageNet statistics — must match training
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Plain transform (resize + to-tensor only; normalisation applied manually
        # so we can interleave the masking step at the right moment)
        self._resize_to_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),          # → float32 in [0, 1], shape (3, 224, 224)
        ])

        self._normalise = transforms.Normalize(
            mean=self._mean.tolist(),
            std =self._std.tolist(),
        )

        # Target layer for Grad-CAM
        self.target_layer = self.model.layer4[-1].conv2

    # ──────────────────────────────────────────────────────────────────────
    def preprocess(
        self,
        image     : Image.Image,
        lung_mask : Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Convert a PIL image to a model-ready (1, 3, 224, 224) tensor.

        When lung_mask is provided (shape 224×224, values 0.0/1.0):
          1. The image is resized to 224×224 and converted to a float tensor.
          2. The mask is broadcast over the 3 colour channels and applied
             *before* ImageNet normalisation.  Background pixels are set to
             the channel mean so they become exactly 0 after normalisation —
             the cleanest possible neutral value for a pre-trained backbone.
          3. The tensor is then normalised normally.
        """
        # ── Step 1: resize + to-tensor ────────────────────────────────
        rgb      = image.convert("RGB")
        t        = self._resize_to_tensor(rgb)        # (3, 224, 224) in [0,1]

        # ── Step 2: apply lung mask (optional) ───────────────────────
        if lung_mask is not None:
            # lung_mask: (224, 224), float32, {0, 1}
            mask_t = torch.from_numpy(lung_mask).unsqueeze(0)  # (1, 224, 224)

            # Set background pixels to the channel mean so they become 0 after
            # normalisation: (mean - mean) / std = 0
            mean_t = torch.tensor(self._mean).view(3, 1, 1)    # (3, 1, 1)
            t      = t * mask_t + mean_t * (1.0 - mask_t)

        # ── Step 3: ImageNet normalisation ────────────────────────────
        t = self._normalise(t)

        return t.unsqueeze(0).to(self.device)           # (1, 3, 224, 224)

    # ──────────────────────────────────────────────────────────────────────
    def predict(
        self,
        input_tensor: torch.Tensor,
    ) -> Tuple[int, float, torch.Tensor]:
        """
        Run inference.

        Returns
        -------
        pred_class     : int   — 0 = Normal, 1 = Pneumonia
        prob_pneumonia : float — probability of class 1
        logits         : torch.Tensor
        """
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs  = torch.softmax(logits, dim=1)
            pred_class     = torch.argmax(probs, dim=1).item()
            prob_pneumonia = probs[0, 1].item()
        return pred_class, prob_pneumonia, logits

    # ──────────────────────────────────────────────────────────────────────
    def compute_gradcam(
        self,
        input_tensor: torch.Tensor,
        class_idx   : int,
        lung_mask   : Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM for *class_idx* on *input_tensor*.

        When lung_mask is provided the final 224×224 heatmap is multiplied
        element-wise by the mask, zeroing any activation outside the lung fields.

        Returns
        -------
        heatmap : np.ndarray, shape (224, 224), values in [0, 1]
        """
        self.model.eval()

        with GradCAM(self.model, target_layer=self.target_layer) as cam:
            out            = self.model(input_tensor)
            activation_map = cam(class_idx, out)
            raw            = activation_map[0].cpu().numpy()   # (7, 7) typically

        # Upscale to 224×224
        heatmap = self._resize_heatmap(raw, (224, 224))

        # Min-max normalise
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max - h_min > 1e-8:
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap = np.zeros_like(heatmap)

        # Apply lung mask — zero activations outside lung fields
        if lung_mask is not None:
            heatmap = heatmap * lung_mask

            # Re-normalise so the colour map uses the full [0,1] range
            h_max2 = heatmap.max()
            if h_max2 > 1e-8:
                heatmap = heatmap / h_max2

        return heatmap.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _resize_heatmap(heatmap: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Robust bilinear upscaling with PIL, handles any input shape."""
        heatmap = np.array(heatmap)
        heatmap = np.squeeze(heatmap)

        if heatmap.ndim > 2:
            heatmap = np.mean(heatmap, axis=-1)
        elif heatmap.ndim < 2:
            heatmap = np.zeros((7, 7), dtype=np.float32)

        heatmap = np.maximum(heatmap, 0.0)
        peak    = heatmap.max()
        if peak > 0:
            heatmap = heatmap / peak

        pil_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        try:
            resized = pil_img.resize(size, Image.Resampling.LANCZOS)
        except AttributeError:
            resized = pil_img.resize(size, Image.LANCZOS)

        return np.array(resized, dtype=np.float32) / 255.0
