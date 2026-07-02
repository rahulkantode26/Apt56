"""Single-image inference — the bridge the web app will call.

Loads the trained checkpoint once and returns class probabilities for an
uploaded image. The LLM explanation layer (added when we build the app) will
consume this structured output; it never makes the classification itself.

CLI usage:
    python -m src.predict path/to/eye.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

from . import config, model as model_lib
from .dataset import get_eval_transforms


class Predictor:
    def __init__(self, checkpoint_path: Path | None = None):
        self.device = config.DEVICE
        path = checkpoint_path or (config.MODELS_DIR / "best_model.pt")
        if not path.exists():
            raise FileNotFoundError(f"No model at {path}. Train one first.")
        self.model, self.class_names = model_lib.load_checkpoint(path, self.device)
        self.transform = get_eval_transforms()

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> dict:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(tensor), dim=1).squeeze(0).cpu()

        scores = {cls: float(probs[i]) for i, cls in enumerate(self.class_names)}
        top_class = max(scores, key=scores.get)
        return {
            "prediction": top_class,
            "confidence": scores[top_class],
            "probabilities": scores,
        }


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m src.predict <image_path>")
        sys.exit(1)
    result = Predictor().predict(sys.argv[1])
    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']:.1%}")
    print("All scores :")
    for cls, p in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        print(f"  {cls:<15} {p:.1%}")


if __name__ == "__main__":
    main()
