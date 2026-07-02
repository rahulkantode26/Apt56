"""Model definition — transfer learning on an ImageNet-pretrained backbone.

We freeze most of the backbone and replace the classifier head. With only a
few hundred images per class this is the setup that actually works; training
from scratch would badly overfit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from . import config


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a backbone with a fresh classification head sized to num_classes."""
    if config.BACKBONE == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif config.BACKBONE == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {config.BACKBONE}")

    return model


def save_checkpoint(model: nn.Module, class_names: list[str], path) -> None:
    """Save weights plus the metadata inference needs to be reproducible."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "backbone": config.BACKBONE,
            "image_size": config.IMAGE_SIZE,
            "norm_mean": config.NORM_MEAN,
            "norm_std": config.NORM_STD,
        },
        path,
    )


def load_checkpoint(path, device) -> tuple[nn.Module, list[str]]:
    """Rebuild a model from a checkpoint for evaluation / inference."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, class_names
