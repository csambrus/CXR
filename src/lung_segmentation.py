# src/lung_segmentation.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Any
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf

from src.config import (
    RAW_DIR,
    SEGMENTATION_RAW_DIR,
    SEGMENTATION_DATA_DIR,
    SEGMENTATION_SPLITS_DIR,
    SEGMENTATION_MODELS_DIR,
    LUNG_MASK_DIR,
    LUNG_MASKED_DIR,
    LUNG_CROP_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
    PLOT_DPI,
    ensure_dir,
)

AUTOTUNE = tf.data.AUTOTUNE


# =========================================================
# Paths
# =========================================================

MONT_DIR = SEGMENTATION_RAW_DIR / "montgomery"
SHEN_DIR = SEGMENTATION_RAW_DIR / "shenzhen"

MERGED_DIR = SEGMENTATION_RAW_DIR / "merged"
MERGED_IMAGES_DIR = MERGED_DIR / "images"
MERGED_MASKS_DIR = MERGED_DIR / "masks"

# =========================================================
# Utils
# =========================================================

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_json(data: dict, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def list_images(
    folder: str | Path,
    exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
) -> list[Path]:
    folder = Path(folder)
    exts = {e.lower() for e in exts}
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def open_gray(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img)


def save_gray(arr: np.ndarray, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    Image.fromarray(arr.astype(np.uint8)).save(path)


# =========================================================
# Dataset detection
# =========================================================

def find_montgomery_structure(root: Path) -> tuple[Path, Path] | None:
    candidates_img = [
        root / "CXR_png",
        root / "images",
    ]

    candidates_mask = [
        root / "ManualMask" / "leftMask",
        root / "Masks" / "left",
    ]

    img_dir = None
    left_dir = None

    for p in candidates_img:
        if p.exists():
            img_dir = p
            break

    for p in candidates_mask:
        if p.exists():
            left_dir = p
            break

    if img_dir is None or left_dir is None:
        return None

    return img_dir, left_dir


def find_shenzhen_structure(root: Path) -> tuple[Path, Path] | None:
    candidates_img = [
        root / "ChinaSet_AllFiles" / "CXR_png",
        root / "images",
    ]

    candidates_mask = [
        root / "ChinaSet_AllFiles" / "ManualMask",
        root / "masks",
    ]

    img_dir = None
    mask_dir = None

    for p in candidates_img:
        if p.exists():
            img_dir = p
            break

    for p in candidates_mask:
        if p.exists():
            mask_dir = p
            break

    if img_dir is None or mask_dir is None:
        return None

    return img_dir, mask_dir


# =========================================================
# Prepare merged segmentation dataset
# =========================================================

def merge_montgomery():
    if not MONT_DIR.exists():
        print("[SKIP] Montgomery folder missing:", MONT_DIR)
        return 0

    found = find_montgomery_structure(MONT_DIR)
    if found is None:
        print("[WARN] Could not detect Montgomery structure.")
        return 0

    img_dir, left_dir = found
    right_dir = left_dir.parent / "rightMask"

    count = 0

    for img_path in list_images(img_dir):
        stem = img_path.stem

        left_mask = right_dir.parent / "leftMask" / f"{stem}.png"
        right_mask = right_dir / f"{stem}.png"

        if not left_mask.exists() or not right_mask.exists():
            continue

        img = open_gray(img_path)
        lm = open_gray(left_mask)
        rm = open_gray(right_mask)

        mask = np.maximum(lm, rm)
        mask = (mask > 0).astype(np.uint8) * 255

        out_img = MERGED_IMAGES_DIR / f"mont_{stem}.png"
        out_mask = MERGED_MASKS_DIR / f"mont_{stem}.png"

        save_gray(img, out_img)
        save_gray(mask, out_mask)
        count += 1

    print(f"[OK] Montgomery merged: {count}")
    return count


def merge_shenzhen():
    if not SHEN_DIR.exists():
        print("[SKIP] Shenzhen folder missing:", SHEN_DIR)
        return 0

    found = find_shenzhen_structure(SHEN_DIR)
    if found is None:
        print("[WARN] Could not detect Shenzhen structure.")
        return 0

    img_dir, mask_dir = found

    count = 0

    for img_path in list_images(img_dir):
        stem = img_path.stem

        possible_masks = [
            mask_dir / f"{stem}.png",
            mask_dir / f"{stem}_mask.png",
        ]

        mask_path = None
        for p in possible_masks:
            if p.exists():
                mask_path = p
                break

        if mask_path is None:
            continue

        img = open_gray(img_path)
        mask = open_gray(mask_path)
        mask = (mask > 0).astype(np.uint8) * 255

        out_img = MERGED_IMAGES_DIR / f"shen_{stem}.png"
        out_mask = MERGED_MASKS_DIR / f"shen_{stem}.png"

        save_gray(img, out_img)
        save_gray(mask, out_mask)
        count += 1

    print(f"[OK] Shenzhen merged: {count}")
    return count


def prepare_segmentation_dataset():
    ensure_dir(MERGED_IMAGES_DIR)
    ensure_dir(MERGED_MASKS_DIR)

    c1 = merge_montgomery()
    c2 = merge_shenzhen()

    total = c1 + c2
    print(f"[OK] Total merged dataset size: {total}")
    return total


# =========================================================
# Split
# =========================================================

def create_splits(
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
):
    set_seed(seed)

    images = list_images(MERGED_IMAGES_DIR)
    stems = [p.stem for p in images]

    random.shuffle(stems)

    n = len(stems)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_ids = stems[:n_test]
    val_ids = stems[n_test:n_test + n_val]
    train_ids = stems[n_test + n_val:]

    ensure_dir(SEGMENTATION_SPLITS_DIR)

    for name, ids in {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }.items():
        pd.DataFrame({"id": ids}).to_csv(SEGMENTATION_SPLITS_DIR / f"{name}.csv", index=False)

    print("[OK] Splits saved:", SEGMENTATION_SPLITS_DIR)

    return {
        "train": len(train_ids),
        "val": len(val_ids),
        "test": len(test_ids),
    }


# =========================================================
# TF loader
# =========================================================

def load_pair(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMAGE_SIZE, method="nearest")
    mask = tf.cast(mask > 0, tf.float32)

    return img, mask


def build_dataset(split_name: str):
    df = pd.read_csv(SEGMENTATION_SPLITS_DIR / f"{split_name}.csv")

    img_paths = [str(MERGED_IMAGES_DIR / f"{x}.png") for x in df["id"]]
    mask_paths = [str(MERGED_MASKS_DIR / f"{x}.png") for x in df["id"]]

    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))

    if split_name == "train":
        ds = ds.shuffle(len(df), seed=SEED)

    ds = ds.map(load_pair, num_parallel_calls=AUTOTUNE)

    if split_name == "train":
        ds = ds.map(
            lambda x, y: (
                tf.image.random_flip_left_right(x),
                tf.image.random_flip_left_right(y),
            ),
            num_parallel_calls=AUTOTUNE,
        )

    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


# =========================================================
# Metrics / loss
# =========================================================

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


# =========================================================
# Model
# =========================================================

def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def encoder_block(x, filters):
    c = conv_block(x, filters)
    p = tf.keras.layers.MaxPooling2D()(c)
    return c, p


def decoder_block(x, skip, filters):
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


def build_unet(input_shape=(224, 224, 1)):
    inputs = tf.keras.Input(shape=input_shape)

    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)

    b1 = conv_block(p4, 512)

    d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(d4)

    model = tf.keras.Model(inputs, outputs, name="lung_unet")
    return model


# =========================================================
# Training
# =========================================================

def train_segmentation(
    epochs: int = 20,
    learning_rate: float = 1e-3,
):
    train_ds = build_dataset("train")
    val_ds = build_dataset("val")

    model = build_unet((IMAGE_SIZE[0], IMAGE_SIZE[1], 1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=bce_dice_loss,
        metrics=[dice_coef],
    )

    out_dir = ensure_dir(SEGMENTATION_MODELS_DIR / "lung_unet")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best_model.keras"),
            save_best_only=True,
            monitor="val_dice_coef",
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coef",
            mode="max",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(str(out_dir / "history.csv")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(out_dir / "last_model.keras")

    return history.history


# =========================================================
# Visualization
# =========================================================

def plot_predictions(n: int = 6):
    model = tf.keras.models.load_model(
        SEGMENTATION_MODELS_DIR / "lung_unet" / "best_model.keras",
        custom_objects={
            "dice_coef": dice_coef,
            "bce_dice_loss": bce_dice_loss,
        },
    )

    ds = build_dataset("test").unbatch().take(n)

    fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n))

    for i, (img, mask) in enumerate(ds):
        pred = model.predict(img[None, ...], verbose=0)[0, :, :, 0]

        axes[i, 0].imshow(img[:, :, 0], cmap="gray")
        axes[i, 0].set_title("Image")

        axes[i, 1].imshow(mask[:, :, 0], cmap="gray")
        axes[i, 1].set_title("True Mask")

        axes[i, 2].imshow(pred > 0.5, cmap="gray")
        axes[i, 2].set_title("Pred Mask")

        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()


# =========================================================
# Inference on classifier dataset
# =========================================================

def predict_mask(model, img_arr: np.ndarray) -> np.ndarray:
    x = img_arr.astype(np.float32) / 255.0
    x = tf.image.resize(x[..., None], IMAGE_SIZE).numpy()[None, ...]

    pred = model.predict(x, verbose=0)[0, :, :, 0]
    pred = (pred > 0.5).astype(np.uint8)

    pred = Image.fromarray(pred * 255).resize(
        (img_arr.shape[1], img_arr.shape[0]),
        resample=Image.NEAREST,
    )

    return np.array(pred)


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (img * (mask > 0)).astype(np.uint8)


def crop_to_mask(img: np.ndarray, mask: np.ndarray, margin: int = 10) -> np.ndarray:
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return img

    x1 = max(0, xs.min() - margin)
    x2 = min(img.shape[1], xs.max() + margin)
    y1 = max(0, ys.min() - margin)
    y2 = min(img.shape[0], ys.max() + margin)

    crop = img[y1:y2, x1:x2]
    return crop


def generate_classifier_variants():
    model = tf.keras.models.load_model(
        SEGMENTATION_MODELS_DIR / "lung_unet" / "best_model.keras",
        custom_objects={
            "dice_coef": dice_coef,
            "bce_dice_loss": bce_dice_loss,
        },
    )

    ensure_dir(LUNG_MASK_DIR)
    ensure_dir(LUNG_MASKED_DIR)
    ensure_dir(LUNG_CROP_DIR)

    image_files = list_images(RAW_DIR)

    for path in image_files:
        rel = path.relative_to(RAW_DIR)

        img = open_gray(path)
        mask = predict_mask(model, img)

        masked = apply_mask(img, mask)
        crop = crop_to_mask(img, mask)

        save_gray(mask, LUNG_MASK_DIR / rel)
        save_gray(masked, LUNG_MASKED_DIR / rel)
        save_gray(crop, LUNG_CROP_DIR / rel)

    print("[OK] Generated:")
    print(" -", LUNG_MASK_DIR)
    print(" -", LUNG_MASKED_DIR)
    print(" -", LUNG_CROP_DIR)


# =========================================================
# Full pipeline
# =========================================================

def run_full_segmentation_pipeline(
    epochs: int = 20,
):
    prepare_segmentation_dataset()
    create_splits()
    train_segmentation(epochs=epochs)
    plot_predictions()
    generate_classifier_variants()
