# dataset_preparation.py
# Run ONCE before training — cache masks to disk.
# This decouples segmentation latency from training throughput.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) # Look in backend/ folder

import json
from pathlib import Path
import numpy as np
from PIL import Image
from segmentation import LungSegmenter

def precompute_masks(image_dir: Path, mask_dir: Path, device: str = "cpu"):
    """
    For every image in image_dir, compute the PSPNet lung mask and
    save as a float16 .npy file alongside the image.
    float16 halves disk I/O versus float32 with no precision loss
    for a binary-derived mask.
    """
    mask_dir.mkdir(parents=True, exist_ok=True)
    segmenter = LungSegmenter(device=device)
    
    failed = []
    for img_path in sorted(image_dir.rglob("*.jpg")) + \
                    sorted(image_dir.rglob("*.png")):
        mask_path = mask_dir / (img_path.stem + ".npy")
        if mask_path.exists():
            continue  # idempotent — safe to re-run
        
        try:
            pil = Image.open(img_path)
            mask = segmenter.segment(pil)           # (224, 224) float32 {0,1}
            np.save(mask_path, mask.astype(np.float16))
        except Exception as e:
            failed.append(str(img_path))
            # Fallback: save all-ones mask (passthrough)
            np.save(mask_path, np.ones((224, 224), dtype=np.float16))
    
    if failed:
        (mask_dir / "failed_masks.json").write_text(json.dumps(failed, indent=2))
    print(f"Done. {len(failed)} failures (saved as passthrough masks).")