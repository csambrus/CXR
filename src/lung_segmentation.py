from __future__ import annotations

from pathlib import Path
from typing import Sequence, Any
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
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
    save_json,
)

from src.runtime import set_seed

AUTOTUNE = tf.data.AUTOTUNE


# =========================================================
# Paths
# =========================================================

CRD_DIR = SEGMENTATION_RAW_DIR / "crd_lung_masks"

MERGED_IMAGES_DIR = SEGMENTATION_DATA_DIR / "images"
MERGED_MASKS_DIR = SEGMENTATION_DATA_DIR / "masks"

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
    with Image.open(path) as img:
        return np.array(img.convert("L"))


def save_gray(arr: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


# =========================================================
# Dataset preparation
# =========================================================

def prepare_segmentation_dataset(
    overwrite: bool = False,
    show_every: int = 250,
) -> int:
    """
    Prepares the segmentation dataset from:

        SEGMENTATION_RAW_DIR / images
        SEGMENTATION_RAW_DIR / masks

    and writes normalized PNG image-mask pairs into:

        SEGMENTATION_DATA_DIR / images
        SEGMENTATION_DATA_DIR / masks
    """
    raw_images_dir = SEGMENTATION_RAW_DIR / "images"
    raw_masks_dir = SEGMENTATION_RAW_DIR / "masks"

    ensure_dir(MERGED_IMAGES_DIR)
    ensure_dir(MERGED_MASKS_DIR)

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

    print("=" * 72)
    print("PREPARE SEGMENTATION DATASET")
    print("=" * 72)
    print("raw_images_dir   :", raw_images_dir)
    print("raw_masks_dir    :", raw_masks_dir)
    print("merged_images_dir:", MERGED_IMAGES_DIR)
    print("merged_masks_dir :", MERGED_MASKS_DIR)
    print("num_images_found :", len(image_files))
    print("overwrite        :", overwrite)
    print("show_every       :", show_every)
    print("=" * 72)

    count = 0
    missing_masks = 0
    skipped_validation = 0
    skipped_existing = 0
    read_errors = 0
    skipped_examples: list[str] = []

    start_time = time.time()

    for idx, img_path in enumerate(
        tqdm(image_files, desc="Preparing pairs", unit="img"),
        start=1,
    ):
        mask_path = raw_masks_dir / img_path.name
        if not mask_path.exists():
            missing_masks += 1
            if len(skipped_examples) < 20:
                skipped_examples.append(f"missing_mask: {img_path.name}")
            continue

        out_img = MERGED_IMAGES_DIR / f"{img_path.stem}.png"
        out_mask = MERGED_MASKS_DIR / f"{img_path.stem}.png"

        if not overwrite and out_img.exists() and out_mask.exists():
            skipped_existing += 1
            continue

        try:
            img = open_gray(img_path)
            mask = open_gray(mask_path)
        except Exception as e:
            read_errors += 1
            if len(skipped_examples) < 20:
                skipped_examples.append(f"read_error: {img_path.name} ({e})")
            continue

        if img.shape[:2] != mask.shape[:2]:
            skipped_validation += 1
            if len(skipped_examples) < 20:
                skipped_examples.append(
                    f"shape_mismatch: {img_path.name} img={img.shape} mask={mask.shape}"
                )
            continue

        mask = (mask > 0).astype(np.uint8) * 255

        save_gray(img, out_img)
        save_gray(mask, out_mask)
        count += 1

        if show_every > 0 and idx % show_every == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0.0
            print(
                f"[INFO] processed={idx}/{len(image_files)} | "
                f"saved={count} | missing_masks={missing_masks} | "
                f"skipped_validation={skipped_validation} | "
                f"skipped_existing={skipped_existing} | "
                f"read_errors={read_errors} | "
                f"{rate:.1f} img/s"
            )

    elapsed = time.time() - start_time

    summary: dict[str, Any] = {
        "source_root": str(SEGMENTATION_RAW_DIR),
        "image_dir": str(raw_images_dir),
        "mask_dir": str(raw_masks_dir),
        "merged_images_dir": str(MERGED_IMAGES_DIR),
        "merged_masks_dir": str(MERGED_MASKS_DIR),
        "num_images_found": len(image_files),
        "num_pairs_saved": count,
        "num_missing_masks": missing_masks,
        "num_skipped_validation": skipped_validation,
        "num_skipped_existing": skipped_existing,
        "num_read_errors": read_errors,
        "elapsed_seconds": elapsed,
        "images_per_second": (len(image_files) / elapsed) if elapsed > 0 else None,
        "skipped_examples": skipped_examples,
        "overwrite": overwrite,
    }
    save_json(summary, SEGMENTATION_DATA_DIR / "prepare_summary.json")

    print("\n" + "=" * 72)
    print("[OK] Prepared segmentation dataset")
    print("=" * 72)
    print("images found       :", len(image_files))
    print("pairs saved        :", count)
    print("missing masks      :", missing_masks)
    print("skipped validation :", skipped_validation)
    print("skipped existing   :", skipped_existing)
    print("read errors        :", read_errors)
    print(f"elapsed            : {elapsed:.2f} sec")
    if elapsed > 0:
        print(f"speed              : {len(image_files) / elapsed:.2f} img/s")
    print("=" * 72)

    return count


# =========================================================
# Splits
# =========================================================

def create_splits(
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
    overwrite: bool = False,
    require_masks: bool = True,
    show_examples: int = 10,
) -> dict[str, Any]:
    """
    Creates train/val/test splits for the prepared segmentation dataset.

    Uses:
        MERGED_IMAGES_DIR
        MERGED_MASKS_DIR

    Saves:
        SEGMENTATION_SPLITS_DIR / train.csv
        SEGMENTATION_SPLITS_DIR / val.csv
        SEGMENTATION_SPLITS_DIR / test.csv
    """
    set_seed(seed)

    train_csv = SEGMENTATION_SPLITS_DIR / "train.csv"
    val_csv = SEGMENTATION_SPLITS_DIR / "val.csv"
    test_csv = SEGMENTATION_SPLITS_DIR / "test.csv"
    summary_json = SEGMENTATION_SPLITS_DIR / "split_summary.json"

    if not overwrite and train_csv.exists() and val_csv.exists() and test_csv.exists():
        print("[SKIP] Segmentation splits already exist.")
        if summary_json.exists():
            with open(summary_json, "r", encoding="utf-8") as f:
                summary = json.load(f)
            print("[INFO] Existing split summary:", summary)
            return summary

        train_n = len(pd.read_csv(train_csv))
        val_n = len(pd.read_csv(val_csv))
        test_n = len(pd.read_csv(test_csv))
        return {
            "train": train_n,
            "val": val_n,
            "test": test_n,
            "total_valid_pairs": train_n + val_n + test_n,
            "seed": seed,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "overwrite": overwrite,
            "require_masks": require_masks,
        }

    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("val_ratio and test_ratio must be >= 0")

    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    if not MERGED_IMAGES_DIR.exists():
        raise RuntimeError(
            f"[ERROR] Prepared segmentation images folder not found: {MERGED_IMAGES_DIR}\n"
            "Run prepare_segmentation_dataset() first."
        )

    if require_masks and not MERGED_MASKS_DIR.exists():
        raise RuntimeError(
            f"[ERROR] Prepared segmentation masks folder not found: {MERGED_MASKS_DIR}\n"
            "Run prepare_segmentation_dataset() first."
        )

    image_files = list_images(MERGED_IMAGES_DIR)

    if len(image_files) == 0:
        raise RuntimeError(
            "[ERROR] No prepared segmentation images found. "
            "Run prepare_segmentation_dataset() first."
        )

    print("=" * 72)
    print("CREATE SEGMENTATION SPLITS")
    print("=" * 72)
    print("images_dir     :", MERGED_IMAGES_DIR)
    print("masks_dir      :", MERGED_MASKS_DIR)
    print("num_images     :", len(image_files))
    print("val_ratio      :", val_ratio)
    print("test_ratio     :", test_ratio)
    print("seed           :", seed)
    print("overwrite      :", overwrite)
    print("require_masks  :", require_masks)
    print("=" * 72)

    valid_ids: list[str] = []
    missing_masks = 0
    invalid_examples: list[str] = []

    for img_path in tqdm(image_files, desc="Validating split candidates", unit="img"):
        img_id = img_path.stem

        if require_masks:
            mask_path = MERGED_MASKS_DIR / f"{img_id}.png"
            if not mask_path.exists():
                missing_masks += 1
                if len(invalid_examples) < show_examples:
                    invalid_examples.append(f"missing_mask: {img_id}")
                continue

        valid_ids.append(img_id)

    n_total_found = len(image_files)
    n_valid = len(valid_ids)

    if n_valid == 0:
        raise RuntimeError("[ERROR] No valid image-mask pairs found for splitting.")

    random.shuffle(valid_ids)

    n_test = int(round(n_valid * test_ratio))
    n_val = int(round(n_valid * val_ratio))

    if n_test + n_val >= n_valid:
        if n_valid >= 3:
            n_test = max(1, min(n_test, n_valid - 2))
            n_val = max(1, min(n_val, n_valid - n_test - 1))
        else:
            raise RuntimeError(
                f"[ERROR] Too few valid samples for requested split ratios: {n_valid}"
            )

    test_ids = valid_ids[:n_test]
    val_ids = valid_ids[n_test:n_test + n_val]
    train_ids = valid_ids[n_test + n_val:]

    if len(train_ids) == 0:
        raise RuntimeError("[ERROR] Train split would be empty.")

    ensure_dir(SEGMENTATION_SPLITS_DIR)

    pd.DataFrame({"id": train_ids}).to_csv(train_csv, index=False)
    pd.DataFrame({"id": val_ids}).to_csv(val_csv, index=False)
    pd.DataFrame({"id": test_ids}).to_csv(test_csv, index=False)

    summary: dict[str, Any] = {
        "train": len(train_ids),
        "val": len(val_ids),
        "test": len(test_ids),
        "total_valid_pairs": n_valid,
        "total_images_found": n_total_found,
        "missing_masks": missing_masks,
        "seed": seed,
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "overwrite": overwrite,
        "require_masks": require_masks,
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "invalid_examples": invalid_examples,
    }
    save_json(summary, summary_json)

    print("\n" + "=" * 72)
    print("CREATE SEGMENTATION SPLITS - SUMMARY")
    print("=" * 72)
    print("images found       :", n_total_found)
    print("valid pairs        :", n_valid)
    print("missing masks      :", missing_masks)
    print("train              :", len(train_ids))
    print("val                :", len(val_ids))
    print("test               :", len(test_ids))
    print("summary json       :", summary_json)
    if invalid_examples:
        print("example invalids   :")
        for item in invalid_examples[:show_examples]:
            print(" -", item)
    print("=" * 72)

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
    flip = tf.random.uniform(()) > 0.5
    image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)
    mask = tf.cond(flip, lambda: tf.image.flip_left_right(mask), lambda: mask)

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

    axes[0].plot(df["loss"], label="train")
    if "val_loss" in df.columns:
        axes[0].plot(df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if "dice_coef" in df.columns:
        axes[1].plot(df["dice_coef"], label="train")
    if "val_dice_coef" in df.columns:
        axes[1].plot(df["val_dice_coef"], label="val")
    axes[1].set_title("Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

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
    overwrite: bool = False,
    show_every: int = 250,
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

    print("=" * 72)
    print("GENERATE CLASSIFIER VARIANTS")
    print("=" * 72)
    print("source_root   :", source_root)
    print("num_images    :", len(image_files))
    print("threshold     :", threshold)
    print("crop_margin   :", crop_margin)
    print("overwrite     :", overwrite)
    print("show_every    :", show_every)
    print("=" * 72)

    count = 0
    skipped_existing = 0
    errors = 0
    error_examples: list[str] = []

    start_time = time.time()

    for idx, path in enumerate(
        tqdm(image_files, desc="Generating classifier variants", unit="img"),
        start=1,
    ):
        rel = path.relative_to(source_root)

        out_mask = LUNG_MASK_DIR / rel
        out_masked = LUNG_MASKED_DIR / rel
        out_crop = LUNG_CROP_DIR / rel

        if not overwrite and out_mask.exists() and out_masked.exists() and out_crop.exists():
            skipped_existing += 1
            continue

        try:
            img = open_gray(path)
            mask = predict_mask(model, img, threshold=threshold)

            masked = apply_mask(img, mask)
            crop = crop_to_mask(img, mask, margin=crop_margin)

            save_gray(mask, out_mask)
            save_gray(masked, out_masked)
            save_gray(crop, out_crop)

            count += 1
        except Exception as e:
            errors += 1
            if len(error_examples) < 20:
                error_examples.append(f"{path.name}: {e}")

        if show_every > 0 and idx % show_every == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0.0
            print(
                f"[INFO] processed={idx}/{len(image_files)} | "
                f"saved={count} | skipped_existing={skipped_existing} | "
                f"errors={errors} | {rate:.1f} img/s"
            )

    elapsed = time.time() - start_time

    summary = {
        "source_root": str(source_root),
        "num_images_found": len(image_files),
        "num_images_processed": count,
        "num_skipped_existing": skipped_existing,
        "num_errors": errors,
        "lung_mask_dir": str(LUNG_MASK_DIR),
        "lung_masked_dir": str(LUNG_MASKED_DIR),
        "lung_crop_dir": str(LUNG_CROP_DIR),
        "threshold": float(threshold),
        "crop_margin": int(crop_margin),
        "overwrite": overwrite,
        "elapsed_seconds": elapsed,
        "images_per_second": (len(image_files) / elapsed) if elapsed > 0 else None,
        "error_examples": error_examples,
    }
    save_json(summary, SEGMENTATION_DATA_DIR / "classifier_variants_summary.json")

    print("\n" + "=" * 72)
    print("[OK] Generated classifier variants")
    print("=" * 72)
    print("images found       :", len(image_files))
    print("newly saved        :", count)
    print("skipped existing   :", skipped_existing)
    print("errors             :", errors)
    print("lung_mask_dir      :", LUNG_MASK_DIR)
    print("lung_masked_dir    :", LUNG_MASKED_DIR)
    print("lung_crop_dir      :", LUNG_CROP_DIR)
    print(f"elapsed            : {elapsed:.2f} sec")
    if elapsed > 0:
        print(f"speed              : {len(image_files) / elapsed:.2f} img/s")
    if error_examples:
        print("example errors     :")
        for item in error_examples[:10]:
            print(" -", item)
    print("=" * 72)

    return summary


def generate_dataset_variants(
    source_root: str | Path = RAW_DIR,
    threshold: float = 0.5,
    crop_margin: int = 10,
    overwrite: bool = False,
    show_every: int = 250,
):
    return generate_classifier_variants(
        source_root=source_root,
        threshold=threshold,
        crop_margin=crop_margin,
        overwrite=overwrite,
        show_every=show_every,
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
    split_summary = create_splits(overwrite=True)

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
        "history_keys": list(history.keys()),
    }
    save_json(result, SEG_MODEL_DIR / "pipeline_summary.json")
    return result


def verify_png_files(paths, label="Files"):
    if isinstance(paths, (str, Path)):
        paths = Path(paths)

        if paths.is_dir():
            paths = sorted(paths.glob("*.png"))
        else:
            paths = [paths]

    bad = 0
    for p in paths:
        p = Path(p)

        if not p.exists():
            print(f"[MISSING] {p}")
            bad += 1
        elif p.stat().st_size == 0:
            print(f"[EMPTY] {p}")
            bad += 1

    print(f"{label}: checked {len(paths)} files, bad={bad}")

