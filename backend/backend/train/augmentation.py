# augmentation.py

import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


class CXRAugmentation:
    """
    Augmentation pipeline specifically designed for masked chest X-rays.
    
    Key constraint: augmentations are applied BEFORE masking in training
    (augment the raw image, then apply the pre-computed mask) OR the mask
    is augmented with IDENTICAL spatial transforms.
    
    We use Option B here: augment both image and mask with the same
    spatial transform, then apply mask to augmented image.
    This preserves the mask-image registration exactly.
    """
    
    def __init__(self, is_training: bool = True, input_size: int = 224):
        self.is_training = is_training
        self.input_size  = input_size
    
    def __call__(
        self,
        image:     Image.Image,
        mask:      np.ndarray,        # (224, 224) float32 {0, 1}
    ) -> tuple[torch.Tensor, np.ndarray]:
        
        if not self.is_training:
            return self._val_transform(image, mask)
        
        return self._train_transform(image, mask)
    
    def _train_transform(
        self,
        image: Image.Image,
        mask:  np.ndarray,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """
        All spatial transforms are applied jointly to (image, mask).
        Photometric transforms are applied to image AFTER masking.
        """
        # Convert mask to PIL for joint spatial transforms
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        
        # ── 1. Joint resize to slightly larger than target ────────────────────
        # Gives spatial transforms room to operate without border artifacts
        oversize = int(self.input_size * 1.1)  # 246 for input_size=224
        image    = TF.resize(image,    oversize, interpolation=Image.BILINEAR)
        mask_pil = TF.resize(mask_pil, oversize, interpolation=Image.NEAREST)
        # NEAREST for mask — no interpolation artifacts at lung boundary
        
        # ── 2. Joint random rotation ──────────────────────────────────────────
        # ±5° captures patient positioning variance
        # Beyond ±7° is non-physical for a properly positioned PA CXR
        if random.random() > 0.3:
            angle = random.uniform(-5.0, 5.0)
            image    = TF.rotate(image,    angle, fill=0,   interpolation=Image.BILINEAR)
            mask_pil = TF.rotate(mask_pil, angle, fill=0,   interpolation=Image.NEAREST)
        
        # ── 3. Joint random affine (translation + mild shear) ─────────────────
        # Translation: ±3% — simulates off-center positioning
        # Shear: ±3° — simulates oblique projections
        if random.random() > 0.5:
            affine_params = T.RandomAffine.get_params(
                degrees     = (0, 0),
                translate   = (0.03, 0.03),
                scale_ranges= (0.95, 1.05),
                shears      = (-3.0, 3.0),
                img_size    = image.size,
            )
            image    = TF.affine(image,    *affine_params, interpolation=Image.BILINEAR, fill=0)
            mask_pil = TF.affine(mask_pil, *affine_params, interpolation=Image.NEAREST,  fill=0)
        
        # ── 4. Joint random crop to target size ───────────────────────────────
        i, j, h, w = T.RandomCrop.get_params(image, (self.input_size, self.input_size))
        image    = TF.crop(image,    i, j, h, w)
        mask_pil = TF.crop(mask_pil, i, j, h, w)
        
        # ── 5. Horizontal flip (with clinical awareness) ──────────────────────
        # NOTE: PA CXR flipping is debated — left cardiac notch moves to right.
        # For pneumonia detection (not laterality diagnosis), it is acceptable
        # as the model learns parenchymal texture, not side-specific anatomy.
        # Disable if your downstream task requires laterality reasoning.
        if random.random() > 0.5:
            image    = TF.hflip(image)
            mask_pil = TF.hflip(mask_pil)
        
        # ── 6. Convert mask back to numpy after spatial transforms ────────────
        mask_aug = (np.array(mask_pil, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32)
        
        # ── 7. To tensor (BEFORE masking, BEFORE normalization) ───────────────
        t = TF.to_tensor(image)   # (3, 224, 224), float32 in [0,1]
        
        # ── 8. Apply augmented mask ───────────────────────────────────────────
        mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        mask_t = torch.from_numpy(mask_aug).unsqueeze(0)
        t      = t * mask_t + mean_t * (1.0 - mask_t)
        
        # ── 9. Photometric augmentation (post-masking, pre-normalization) ──────
        # Applied only to unmasked (lung) pixels — background already set to mean
        # Simulates scanner variability: kVp, mAs, detector sensitivity, CLAHE preprocessing
        if random.random() > 0.3:
            brightness_factor = random.uniform(0.8, 1.2)
            t = torch.clamp(t * brightness_factor, 0.0, 1.0)
        
        if random.random() > 0.5:
            # Contrast: stretch/compress intensity range around mean
            mean_val = t[mask_t.expand_as(t) > 0.5].mean()
            contrast = random.uniform(0.85, 1.15)
            t = torch.clamp((t - mean_val) * contrast + mean_val, 0.0, 1.0)
        
        # ── 10. Gaussian noise (simulates detector quantum noise) ──────────────
        if random.random() > 0.5:
            noise_std = random.uniform(0.005, 0.02)
            noise     = torch.randn_like(t) * noise_std
            t         = torch.clamp(t + noise * mask_t, 0.0, 1.0)
        
        # ── 11. ImageNet normalization ────────────────────────────────────────
        normalise = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        t = normalise(t)
        
        # ── 12. Random Erasing (simulates medical device artifacts, ECG leads) ─
        # Applied AFTER normalization so erased regions = normalized background
        # Probability 20%: clinical artifacts are relatively rare
        if random.random() > 0.8:
            # Constrain to small regions — large erasures destroy lung anatomy
            erase_h = random.randint(5, 25)
            erase_w = random.randint(5, 25)
            i_e = random.randint(0, self.input_size - erase_h)
            j_e = random.randint(0, self.input_size - erase_w)
            t[:, i_e:i_e+erase_h, j_e:j_e+erase_w] = 0.0  # normalized background
        
        return t, mask_aug
    
    def _val_transform(
        self,
        image: Image.Image,
        mask:  np.ndarray,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Deterministic transform — must exactly match inference pipeline."""
        t     = TF.to_tensor(TF.resize(image, self.input_size, Image.BILINEAR))
        mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        t      = t * mask_t + mean_t * (1.0 - mask_t)
        t      = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(t)
        return t, mask