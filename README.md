# Eye Disease Detection — Training Pipeline

Detects eye conditions (starting with **cataract**) from photographs using a
transfer-learned CNN. A future web app will send an uploaded photo through this
model and use an LLM to explain the result in plain language — the LLM never
makes the diagnosis itself.

> ⚠️ **Not a medical device.** This is a learning/portfolio project. Its output
> is not a diagnosis. Anyone with a real concern should see a qualified eye care
> professional.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Training uses Apple Silicon GPU (MPS) automatically, falling back to CUDA/CPU.

## 1. Get training data

The pipeline expects **external eye photos** (close-up of the eye), not fundus
/ retinal images. Good starting datasets:

| Dataset | Link |
|---|---|
| Kaggle — Cataract Image Dataset | https://www.kaggle.com/datasets/nandanp6/cataract-image-dataset/data |
| Mendeley — Eye Diseases Classification (cataract, conjunctivitis, uveitis, eyelid) | https://data.mendeley.com/datasets/n9zp473wfw/1 |
| Roboflow — Cataract v01 | https://universe.roboflow.com/cataract/cataract-v01 |

Arrange the images into one folder per class under `data/raw/`:

```
data/raw/
├── normal/       <- healthy eye photos
└── cataract/     <- cataract eye photos
```

To add more diseases later, just add another folder (e.g. `data/raw/glaucoma/`).
No code changes needed — classes are discovered from the folder names.

## 2. Split into train/val

```bash
python -m src.prepare_data
```

Produces `data/processed/{train,val}/<class>/`. An 80/20 split by default
(`VAL_SPLIT` in `src/config.py`).

## 3. Train

```bash
python -m src.train
```

- EfficientNet-B0 backbone, ImageNet-pretrained, new classification head.
- Inverse-frequency class weighting (the datasets are imbalanced).
- Early stopping on validation loss; best checkpoint saved to
  `models/best_model.pt`.

## 4. Evaluate

```bash
python -m src.evaluate
```

Prints per-class precision/recall/F1 and a confusion matrix. **For a screening
task, watch recall on the disease class** — missing a real cataract (false
negative) is worse than a false alarm.

## 5. Predict on one image

```bash
python -m src.predict path/to/eye.jpg
```

Returns the predicted class, confidence, and full probability distribution.
This is exactly what the web app backend will call.

## Project layout

```
src/
├── config.py        # paths + hyperparameters + device selection
├── prepare_data.py  # raw/ -> processed/ train/val split
├── dataset.py       # transforms + dataloaders (augmentation)
├── model.py         # backbone + head; checkpoint save/load
├── train.py         # training loop, class weights, early stopping
├── evaluate.py      # classification report + confusion matrix
└── predict.py       # single-image inference (app entry point)
```

## Tuning knobs (`src/config.py`)

- `BACKBONE` — `"efficientnet_b0"` or `"resnet18"`
- `IMAGE_SIZE`, `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`
- `VAL_SPLIT`, `EARLY_STOP_PATIENCE`

## Notes on data realism

Public cataract datasets are small (hundreds of images) and often web-scraped,
so labels vary in quality — inspect them before trusting results, and treat
accuracy numbers conservatively. The augmentation in `dataset.py` (rotation,
flip, colour jitter) partly compensates for the small size.
