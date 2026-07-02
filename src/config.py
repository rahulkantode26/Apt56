"""Central configuration for the eye-disease training pipeline.

Everything path- and hyperparameter-related lives here so the other modules
stay small. Classes are discovered from the data folders at runtime, so adding
a new disease later means adding image folders, not editing code.
"""

from __future__ import annotations

from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # unsplit source images: raw/<class>/*.jpg
PROCESSED_DIR = DATA_DIR / "processed"  # split output: processed/{train,val}/<class>/*
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Image / model
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224                     # EfficientNet-B0 / ResNet default input
# ImageNet normalization (the pretrained backbones expect these stats).
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

BACKBONE = "efficientnet_b0"         # or "resnet18"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2                      # fraction held out for validation
EARLY_STOP_PATIENCE = 5             # epochs without val-loss improvement
NUM_WORKERS = 2
SEED = 42

# ---------------------------------------------------------------------------
# Device (Apple Silicon GPU -> CUDA -> CPU)
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
