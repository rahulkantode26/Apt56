"""Evaluate the saved model on the validation set.

Prints a per-class precision/recall/F1 report and a confusion matrix — for a
medical screening task, recall on the disease class matters more than raw
accuracy, so look at those numbers, not just the headline accuracy.

Run:
    python -m src.evaluate
"""

from __future__ import annotations

import torch
from sklearn.metrics import classification_report, confusion_matrix

from . import config, model as model_lib
from .dataset import build_dataloaders


def main() -> None:
    device = config.DEVICE
    best_path = config.MODELS_DIR / "best_model.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"No model at {best_path}. Run `python -m src.train` first.")

    model, class_names = model_lib.load_checkpoint(best_path, device)
    _, val_loader, _ = build_dataloaders()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

    print("Confusion matrix (rows = true, cols = predicted):")
    print("labels:", class_names)
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
