"""Split raw images into train/val folders.

Input  : data/raw/<class>/*.{jpg,jpeg,png}
Output : data/processed/{train,val}/<class>/*

Run:
    python -m src.prepare_data
"""

from __future__ import annotations

import random
import shutil

from . import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(folder):
    return [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]


def prepare(val_split: float = config.VAL_SPLIT, seed: int = config.SEED) -> None:
    random.seed(seed)

    if not config.RAW_DIR.exists():
        raise FileNotFoundError(
            f"{config.RAW_DIR} does not exist. Create data/raw/<class>/ folders "
            "and drop images in them first."
        )

    class_dirs = sorted(d for d in config.RAW_DIR.iterdir() if d.is_dir())
    if not class_dirs:
        raise FileNotFoundError(
            f"No class subfolders found in {config.RAW_DIR}. Expected e.g. "
            "data/raw/normal/ and data/raw/cataract/."
        )

    # Start clean so re-runs are deterministic.
    if config.PROCESSED_DIR.exists():
        shutil.rmtree(config.PROCESSED_DIR)

    summary = {}
    for class_dir in class_dirs:
        images = _list_images(class_dir)
        if not images:
            print(f"  ! {class_dir.name}: no images, skipping")
            continue
        random.shuffle(images)

        n_val = max(1, int(len(images) * val_split))
        val_images = images[:n_val]
        train_images = images[n_val:]

        for split, files in (("train", train_images), ("val", val_images)):
            dest = config.PROCESSED_DIR / split / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for src in files:
                shutil.copy2(src, dest / src.name)

        summary[class_dir.name] = (len(train_images), len(val_images))

    print(f"Prepared data in {config.PROCESSED_DIR}")
    print(f"{'class':<20}{'train':>8}{'val':>8}")
    for name, (n_train, n_val) in summary.items():
        print(f"{name:<20}{n_train:>8}{n_val:>8}")


if __name__ == "__main__":
    prepare()
