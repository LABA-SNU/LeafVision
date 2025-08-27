import os
import torch
from pathlib import Path
from torch import nn
from torchvision import models as torchvision_models

import vision_transformer as vits


class LinearClassifier(nn.Module):
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)

def init_pretrained_model(arch, ssl, num_labels=1000, map_location="cpu", device="cpu"):
    """
    Load a pretrained backbone from `models/LeafVision_{ssl}_{arch}.pth` and return it with weights.

    Args:
        arch (str): Model name (e.g., 'resnet50', 'efficientnet_b0', 'vit_base'). Must exist in
            `torchvision.models.__dict__` or `vits.__dict__`.
        ssl (str): Pretraining tag used in the filename (e.g., 'DINO', 'BYOL', 'SimCLR', 'Supervised').
        num_labels (int): Number of labels to classify.
        map_location (str | torch.device, optional): Passed to `torch.load` (e.g., 'cpu', 'cuda:0').
        device (str|torch.device): device to map the model onto (default: 'cpu')

    Returns:
        torch.nn.Module: Backbone with classification head removed.

    Raises:
        FileNotFoundError: Weight file not found.
        ValueError: Unknown architecture name.
        RuntimeError: Invalid checkpoint format.
    """
    ROOT = Path.cwd()
    models_dir = ROOT / "models"
    weight_path = models_dir / f"LeafVision_{ssl}_{arch}.pth"

    if not weight_path.exists():
        raise FileNotFoundError(f"Pretrained weight not found: {weight_path}")

    # Rebuild model
    if "vit" in arch:
        model = vits.__dict__[arch](patch_size=16, num_classes=0)
        embed_dim = model.embed_dim * (1 + int(True))
    elif "resnet" in arch:
        model = torchvision_models.__dict__[arch](weights=None)
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    elif "efficientnet" in arch:
        model = torchvision_models.__dict__[arch](weights=None)
        embed_dim = model.classifier[1].weight.shape[1]
        if hasattr(model, "classifier"):
            model.classifier = nn.Identity()
        else:
            # fallback (older torchvision variants)
            model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Load state dict (be robust to different checkpoint styles)
    state = torch.load(weight_path, map_location=map_location)
    if isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values()):
        state_dict = state
    elif isinstance(state, dict):
        # common keys: 'state_dict', 'model_state_dict'
        if "state_dict" in state:
            state_dict = state["state_dict"]
        elif "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        else:
            # try to find the first tensor-dict
            cand = {k: v for k, v in state.items() if isinstance(v, dict) and v and all(isinstance(x, torch.Tensor) for x in v.values())}
            if cand:
                state_dict = next(iter(cand.values()))
            else:
                raise RuntimeError(f"Could not locate a valid state_dict in checkpoint: {weight_path}")
    else:
        raise RuntimeError(f"Unexpected checkpoint format at: {weight_path}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] Key mismatch for {weight_path.name}: missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) <= 10 and len(unexpected) <= 10:
            if missing:
                print("  missing:", missing)
            if unexpected:
                print("  unexpected:", unexpected)
    else:
        print(f"[ok] Loaded cleanly: {weight_path.name}")

    classifier = LinearClassifier(embed_dim, num_labels=num_labels)

    with torch.no_grad():
        if "resnet" in arch:
            last_block = model.layer4[-1]
            if hasattr(last_block, 'bn3'):  
                final_bn = last_block.bn3
            else:
                final_bn = last_block.bn2
            
            mean_val = final_bn.weight.data.mean()
            std_val = final_bn.weight.data.std()

        elif "vit" in arch:
            final_ln = model.norm
            mean_val = final_ln.weight.data.mean()
            std_val = final_ln.weight.data.std()
        else:
            mean_val = 0.0
            std_val = 0.01

    return model.to(device), classifier.to(device)