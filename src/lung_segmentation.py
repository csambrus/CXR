from __future__ import annotations

from pathlib import Path
from typing import Sequence, Any
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf

from src.config import (
    DATA_DIR,
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
    save_json
)

AUTOTUNE = tf.data.AUTOTUNE

# =========================================================
# Paths
# =========================================================

CRD_DIR = SEGMENTATION_RAW_DIR / "crd_lung_masks"

MERGED_IMAGES_DIR = SEGMENTATION_DATA_DIR / "merged_images"
MERGED_MASKS_DIR = SEGMENTATION_DATA_DIR / "merged_masks"

SEG_MODEL_DIR = SEGMENTATION_MODELS_DIR / "lung_unet"

# =========================================================
# General utils
# =========================================================

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


def save_gray(arr: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    Image.fromarray(arr.astype(np.uint8)).save(path)


# =========================================================
# Dataset preparation
# =========================================================
def prepare_segmentation_dataset() -> int:
    """
    Prepares the segmentation dataset from the flattened raw structure:

        SEGMENTATION_RAW_DIR / images
        SEGMENTATION_RAW_DIR / masks

    and writes normalized PNG image-mask pairs into:

        MERGED_DIR / images
        MERGED_DIR / masks
    """
    raw_images_dir = SEGMENTATION_RAW_DIR / "images"
    raw_masks_dir = SEGMENTATION_RAW_DIR / "masks"

    merged_images_dir = SEGMENTATION_DATA_DIR / "images"
    merged_masks_dir = SEGMENTATION_DATA_DIR / "masks"

    if not raw_images_dir.exists():
        raise RuntimeError(
            f"[ERROR] Segmentation images folder not found: {raw_images_dir}\n"
            f"Download it first into {SEGMENTATION_RAW_DIR}."
        )

    if not raw_masks_dir.exists():
        raise RuntimeError(
            f"[ERROR] Segmentation masks folder not found: {raw_masks_dir}\n"
            f"Download it first into {SEGMENTATION_RAW_DIR}."
        )

    image_files = list_images(raw_images_dir)

    if len(image_files) == 0:
        raise RuntimeError(f"[ERROR] No images found in: {raw_images_dir}")

    count = 0
    missing_masks = 0

    for img_path in image_files:
        mask_path = raw_masks_dir / img_path.name
        if mask_path is None:
            missing_masks += 1
            continue

        img = open_gray(img_path)
        mask = open_gray(mask_path)

        mask = (mask > 0).astype(np.uint8) * 255

        out_img = merged_images_dir / f"{img_path.stem}.png"
        out_mask = merged_masks_dir / f"{img_path.stem}.png"

        save_gray(img, out_img)
        save_gray(mask, out_mask)
        count += 1

    summary = {
        "source_root": str(SEGMENTATION_RAW_DIR),
        "image_dir": str(raw_images_dir),
        "mask_dir": str(raw_masks_dir),
        "num_images_found": len(image_files),
        "num_pairs_saved": count,
        "num_missing_masks": missing_masks,
    }
    save_json(summary, MERGED_DIR / "prepare_summary.json")

    print(f"[OK] Prepared segmentation dataset: {count} pairs")
    if missing_masks:
        print(f"[WARN] Missing masks for {missing_masks} images")

    return count
# =========================================================
# Splits
# =========================================================

def create_splits(
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
) -> dict[str, int]:
    set_seed(seed)

    image_files = list_images(MERGED_IMAGES_DIR)
    ids = [p.stem for p in image_files]

    if len(ids) == 0:
        raise RuntimeError(
            "[ERROR] No prepared segmentation images found. "
            "Run prepare_segmentation_dataset() first."
        )

    random.shuffle(ids)

    n = len(ids)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test + n_val]
    train_ids = ids[n_test + n_val:]

    ensure_dir(SEGMENTATION_SPLITS_DIR)

    pd.DataFrame({"id": train_ids}).to_csv(SEGMENTATION_SPLITS_DIR / "train.csv", index=False)
    pd.DataFrame({"id": val_ids}).to_csv(SEGMENTATION_SPLITS_DIR / "val.csv", index=False)
    pd.DataFrame({"id": test_ids}).to_csv(SEGMENTATION_SPLITS_DIR / "test.csv", index=False)

    summary = {
        "train": len(train_ids),
        "val": len(val_ids),
        "test": len(test_ids),
        "total": n,
    }
    save_json(summary, SEGMENTATION_SPLITS_DIR / "split_summary.json")

    print("[OK] Segmentation splits created:", summary)
    return summary


# =========================================================
# TF data pipeline
# =========================================================

def load_pair(img_path: tf.Tensor, mask_path: tf.Tensor):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMAGE_SIZE, method="nearest")
    mask = tf.cast(mask > 0, tf.float32)

    return img, mask


def augment_pair(image: tf.Tensor, mask: tf.Tensor):
    # shared flip
    flip = tf.random.uniform(()) > 0.5
    image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)
    mask = tf.cond(flip, lambda: tf.image.flip_left_right(mask), lambda: mask)

    # small brightness/contrast only on image
    image = tf.image.random_brightness(image, max_delta=0.05)
    image = tf.image.random_contrast(image, lower=0.95, upper=1.05)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask


def build_dataset(split_name: str, batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    split_csv = SEGMENTATION_SPLITS_DIR / f"{split_name}.csv"
    if not split_csv.exists():
        raise RuntimeError(f"[ERROR] Missing split file: {split_csv}")

    df = pd.read_csv(split_csv)
    if len(df) == 0:
        raise RuntimeError(f"[ERROR] Empty split file: {split_csv}")

    img_paths = [str(MERGED_IMAGES_DIR / f"{x}.png") for x in df["id"]]
    mask_paths = [str(MERGED_MASKS_DIR / f"{x}.png") for x in df["id"]]

    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))

    if split_name == "train":
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(load_pair, num_parallel_calls=AUTOTUNE)

    if split_name == "train":
        ds = ds.map(augment_pair, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# =========================================================
# Metrics / losses
# =========================================================

def dice_coef(y_true, y_pred, smooth: float = 1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    return (2.0 * intersection + smooth) / (denom + smooth)


def iou_coef(y_true, y_pred, smooth: float = 1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    return (intersection + smooth) / (union + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


# =========================================================
# Model
# =========================================================

def conv_block(x, filters: int):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def encoder_block(x, filters: int):
    c = conv_block(x, filters)
    p = tf.keras.layers.MaxPooling2D()(c)
    return c, p


def decoder_block(x, skip, filters: int):
    x = tf.keras.layers.UpSampling2D(interpolation="bilinear")(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


def build_unet(input_shape: tuple[int, int, int] = (224, 224, 1)) -> tf.keras.Model:
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

    return tf.keras.Model(inputs, outputs, name="lung_unet")


# =========================================================
# Training
# =========================================================

def train_segmentation(
    epochs: int = 20,
    learning_rate: float = 1e-3,
    batch_size: int = BATCH_SIZE,
) -> dict[str, list[float]]:
    #set_seed(SEED)

    train_ds = build_dataset("train", batch_size=batch_size)
    val_ds = build_dataset("val", batch_size=batch_size)

    model = build_unet((IMAGE_SIZE[0], IMAGE_SIZE[1], 1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=bce_dice_loss,
        metrics=[dice_coef, iou_coef],
    )

    out_dir = ensure_dir(SEG_MODEL_DIR)

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
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
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

    history_dict = history.history
    save_json(history_dict, out_dir / "history.json")

    return history_dict


def evaluate_segmentation(batch_size: int = BATCH_SIZE) -> dict[str, float]:
    model_path = SEG_MODEL_DIR / "best_model.keras"
    if not model_path.exists():
        raise RuntimeError(f"[ERROR] Missing trained model: {model_path}")

    test_ds = build_dataset("test", batch_size=batch_size)

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "dice_coef": dice_coef,
            "iou_coef": iou_coef,
            "bce_dice_loss": bce_dice_loss,
        },
    )

    result = model.evaluate(test_ds, verbose=1)

    if isinstance(result, list):
        names = model.metrics_names
        metrics = {k: float(v) for k, v in zip(names, result)}
    else:
        metrics = {"loss": float(result)}

    save_json(metrics, SEG_MODEL_DIR / "test_metrics.json")

    print("[OK] Segmentation test metrics:", metrics)
    return metrics


# =========================================================
# Visualization
# =========================================================

def plot_training_history():
    history_csv = SEG_MODEL_DIR / "history.csv"
    if not history_csv.exists():
        raise RuntimeError(f"[ERROR] Missing history file: {history_csv}")

    df = pd.read_csv(history_csv)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # loss
    axes[0].plot(df["loss"], label="train")
    if "val_loss" in df.columns:
        axes[0].plot(df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # dice
    if "dice_coef" in df.columns:
        axes[1].plot(df["dice_coef"], label="train")
    if "val_dice_coef" in df.columns:
        axes[1].plot(df["val_dice_coef"], label="val")
    axes[1].set_title("Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # iou
    if "iou_coef" in df.columns:
        axes[2].plot(df["iou_coef"], label="train")
    if "val_iou_coef" in df.columns:
        axes[2].plot(df["val_iou_coef"], label="val")
    axes[2].set_title("IoU")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle("Lung segmentation training history")
    fig.tight_layout()
    fig.savefig(SEG_MODEL_DIR / "training_history.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.show()


def plot_predictions(n: int = 6):
    model_path = SEG_MODEL_DIR / "best_model.keras"
    if not model_path.exists():
        raise RuntimeError(f"[ERROR] Missing trained model: {model_path}")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "dice_coef": dice_coef,
            "iou_coef": iou_coef,
            "bce_dice_loss": bce_dice_loss,
        },
    )

    ds = build_dataset("test", batch_size=1).unbatch().take(n)

    fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n))
    if n == 1:
        axes = np.array([axes])

    for i, (img, mask) in enumerate(ds):
        pred = model.predict(img[None, ...], verbose=0)[0, :, :, 0]
        pred_bin = (pred > 0.5).astype(np.uint8)

        axes[i, 0].imshow(img[:, :, 0], cmap="gray")
        axes[i, 0].set_title("Image")

        axes[i, 1].imshow(mask[:, :, 0], cmap="gray")
        axes[i, 1].set_title("True Mask")

        axes[i, 2].imshow(pred_bin, cmap="gray")
        axes[i, 2].set_title("Pred Mask")

        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()


# =========================================================
# Inference on classifier dataset
# =========================================================

def load_segmentation_model() -> tf.keras.Model:
    model_path = SEG_MODEL_DIR / "best_model.keras"
    if not model_path.exists():
        raise RuntimeError(f"[ERROR] Missing trained model: {model_path}")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "dice_coef": dice_coef,
            "iou_coef": iou_coef,
            "bce_dice_loss": bce_dice_loss,
        },
    )
    return model


def predict_mask(model: tf.keras.Model, img_arr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    x = img_arr.astype(np.float32) / 255.0
    x = tf.image.resize(x[..., None], IMAGE_SIZE).numpy()[None, ...]

    pred = model.predict(x, verbose=0)[0, :, :, 0]
    pred = (pred > threshold).astype(np.uint8)

    pred = Image.fromarray(pred * 255).resize(
        (img_arr.shape[1], img_arr.shape[0]),
        resample=Image.Resampling.NEAREST,
    )

    return np.array(pred)


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (img * (mask > 0)).astype(np.uint8)


def crop_to_mask(
    img: np.ndarray,
    mask: np.ndarray,
    margin: int = 10,
    min_size: int = 32,
) -> np.ndarray:
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return img

    x1 = max(0, int(xs.min()) - margin)
    x2 = min(img.shape[1], int(xs.max()) + margin + 1)
    y1 = max(0, int(ys.min()) - margin)
    y2 = min(img.shape[0], int(ys.max()) + margin + 1)

    crop = img[y1:y2, x1:x2]

    if crop.shape[0] < min_size or crop.shape[1] < min_size:
        return img

    return crop


def generate_classifier_variants(
    source_root: str | Path = RAW_DIR,
    threshold: float = 0.5,
    crop_margin: int = 10,
) -> dict[str, Any]:
    """
    Generates lung_masks / lung_masked / lung_crop from the classifier dataset.
    """
    source_root = Path(source_root)
    if not source_root.exists():
        raise RuntimeError(f"[ERROR] Source classifier dataset not found: {source_root}")

    model = load_segmentation_model()

    ensure_dir(LUNG_MASK_DIR)
    ensure_dir(LUNG_MASKED_DIR)
    ensure_dir(LUNG_CROP_DIR)

    image_files = list_images(source_root)
    if len(image_files) == 0:
        raise RuntimeError(f"[ERROR] No images found in: {source_root}")

    count = 0

    for path in image_files:
        rel = path.relative_to(source_root)

        img = open_gray(path)
        mask = predict_mask(model, img, threshold=threshold)

        masked = apply_mask(img, mask)
        crop = crop_to_mask(img, mask, margin=crop_margin)

        save_gray(mask, LUNG_MASK_DIR / rel)
        save_gray(masked, LUNG_MASKED_DIR / rel)
        save_gray(crop, LUNG_CROP_DIR / rel)

        count += 1

    summary = {
        "source_root": str(source_root),
        "num_images_processed": count,
        "lung_mask_dir": str(LUNG_MASK_DIR),
        "lung_masked_dir": str(LUNG_MASKED_DIR),
        "lung_crop_dir": str(LUNG_CROP_DIR),
        "threshold": float(threshold),
        "crop_margin": int(crop_margin),
    }
    save_json(summary, SEGMENTATION_DATA_DIR / "classifier_variants_summary.json")

    print("[OK] Generated classifier variants")
    print(" -", LUNG_MASK_DIR)
    print(" -", LUNG_MASKED_DIR)
    print(" -", LUNG_CROP_DIR)

    return summary


# backward compatible alias
def generate_dataset_variants(
    source_root: str | Path = RAW_DIR,
    threshold: float = 0.5,
    crop_margin: int = 10,
):
    return generate_classifier_variants(
        source_root=source_root,
        threshold=threshold,
        crop_margin=crop_margin,
    )


# =========================================================
# Full pipeline
# =========================================================

def run_full_segmentation_pipeline(
    epochs: int = 20,
    learning_rate: float = 1e-3,
    batch_size: int = BATCH_SIZE,
    threshold: float = 0.5,
    crop_margin: int = 10,
) -> dict[str, Any]:
    prepare_segmentation_dataset()
    split_summary = create_splits()
    history = train_segmentation(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    test_metrics = evaluate_segmentation(batch_size=batch_size)

    try:
        plot_training_history()
    except Exception as e:
        print("[WARN] plot_training_history failed:", e)

    try:
        plot_predictions()
    except Exception as e:
        print("[WARN] plot_predictions failed:", e)

    variant_summary = generate_classifier_variants(
        source_root=RAW_DIR,
        threshold=threshold,
        crop_margin=crop_margin,
    )

    result = {
        "split_summary": split_summary,
        "test_metrics": test_metrics,
        "variant_summary": variant_summary,
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
    }
    save_json(result, SEG_MODEL_DIR / "pipeline_summary.json")
    return result
