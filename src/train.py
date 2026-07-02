"""Training loop with class weighting, early stopping, and best-model saving.

Run:
    python -m src.train
"""

from __future__ import annotations

import collections

import torch
import torch.nn as nn
from tqdm import tqdm

from . import config, model as model_lib
from .dataset import build_dataloaders


def _class_weights(train_loader, num_classes: int, device) -> torch.Tensor:
    """Inverse-frequency weights so a minority class isn't ignored.

    These small cataract datasets are imbalanced (often ~3x more 'normal'),
    so we weight the loss instead of letting the model predict the majority
    class every time.
    """
    counts = collections.Counter(train_loader.dataset.targets)
    total = sum(counts.values())
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float, device=device)


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main() -> None:
    torch.manual_seed(config.SEED)
    device = config.DEVICE
    print(f"Device: {device}")

    train_loader, val_loader, class_names = build_dataloaders()
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    model = model_lib.build_model(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(train_loader, num_classes, device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    config.MODELS_DIR.mkdir(exist_ok=True)
    best_path = config.MODELS_DIR / "best_model.pt"
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{config.NUM_EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model_lib.save_checkpoint(model, class_names, best_path)
            print(f"  ↳ saved new best model (val loss {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"Done. Best val loss {best_val_loss:.4f}. Model at {best_path}")


if __name__ == "__main__":
    main()
