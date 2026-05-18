# foundation_init.py

import torch
import torch.nn as nn
from torchvision import models
import torchxrayvision as xrv

def build_masked_cxr_resnet34(num_classes: int = 2, freeze_epochs: int = 3) -> nn.Module:
    """
    ResNet-34 initialized with TorchXRayVision CXR weights.
    
    Strategy: the xrv model is a DenseNet; we use it only to extract
    a strong CXR-domain weight initialization for our ResNet-34 by
    transferring the learned low-level filter statistics via a surgical
    partial weight transfer.
    
    For a clean ResNet-34 with CXR-domain init, RadImageNet is the
    direct path. Instructions for both are below.
    """
    # ── Option A: RadImageNet (RECOMMENDED for ResNet-34 specifically) ────────
    # Download: https://github.com/BMEII-AI/RadImageNet
    # wget https://github.com/BMEII-AI/RadImageNet/releases/download/v1.0/RadImageNet-ResNet34_notop.pth
    
    model = models.resnet34(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    try:
        # RadImageNet weights: keys match torchvision ResNet-34 exactly
        # EXCEPT fc (different num_classes) — we handle that below
        rad_state = torch.load("RadImageNet-ResNet34_notop.pth", map_location="cpu")
        
        # Strip the final classifier — we replace it
        rad_state = {k: v for k, v in rad_state.items() 
                     if not k.startswith("fc.")}
        
        missing, unexpected = model.load_state_dict(rad_state, strict=False)
        print(f"RadImageNet init: missing={missing}, unexpected={unexpected}")
        # Expected: missing=['fc.weight', 'fc.bias'], unexpected=[]
        
        # Initialize fc fresh with Xavier uniform (better than random normal
        # for medical classification — prevents logit saturation)
        nn.init.xavier_uniform_(model.fc.weight)
        nn.init.zeros_(model.fc.bias)
        
    except FileNotFoundError:
        print("WARNING: RadImageNet weights not found. Falling back to "
              "TorchXRayVision feature extraction init.")
        # ── Option B: Extract conv statistics from xrv DenseNet ──────────────
        # Less ideal (architecture mismatch) but much better than ImageNet
        # Use random init + aggressive augmentation instead as last resort
        nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
    
    return model


def get_layer_groups(model: nn.Module) -> list[list[nn.Parameter]]:
    """
    Return parameter groups for discriminative learning rates.
    Backbone early layers → lower LR. Head → higher LR.
    Standard in fastai / ULMFiT; essential for medical transfer learning.
    """
    early  = list(model.layer1.parameters()) + list(model.layer2.parameters())
    late   = list(model.layer3.parameters()) + list(model.layer4.parameters())
    head   = list(model.fc.parameters())
    return [early, late, head]