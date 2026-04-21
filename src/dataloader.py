from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import tensorflow as tf

from src.config import (
    BATCH_SIZE,
    CACHE_DATASET,
    IMAGE_SIZE,
    RAW_DIR,
    SEED,
)

AUTOTUNE = tf.data.AUTOTUNE


# =========================================================
# Path helpers
# =========================================================

def resolve_image_path(root_dir: str | Path, relative_path: str) -> str:
    return str(Path(root_dir) / relative_path)


def read_split_csv(split_csv_path: str | Path) -> pd.DataFrame:
    split_csv_path = Path(split_csv_path)
    df = pd.read_csv(split_csv_path)

    required_cols = {"relative_path", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {split_csv_path}: {sorted(missing)}"
        )

    return df


# =========================================================
# Image loading
# =========================================================

def load_image_from_path(
    path: tf.Tensor,
    label: tf.Tensor,
    image_size: tuple[int, int] = IMAGE_SIZE,
    channels: int = 1,
):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=channels, expand_animations=False)
    image.set_shape([None, None, channels])

    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0

    return image, tf.cast(label, tf.int32)


# =========================================================
# Augmentation
# =========================================================

def build_default_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(height_factor=0.05, width_factor=0.05),
            tf.keras.layers.RandomTranslation(height_factor=0.03, width_factor=0.03),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="cxr_augmentation",
    )


# =========================================================
# Dataset builders
# =========================================================

def build_dataset_from_dataframe(
    df: pd.DataFrame,
    root_dir: str | Path,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    augment_fn: Callable | None = None,
    cache: bool = CACHE_DATASET,
    image_size: tuple[int, int] = IMAGE_SIZE,
    channels: int = 1,
):
    if len(df) == 0:
        raise ValueError("Received empty dataframe.")

    paths = [resolve_image_path(root_dir, p) for p in df["relative_path"].tolist()]
    labels = df["label"].astype(int).tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(df),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        lambda p, y: load_image_from_path(
            p,
            y,
            image_size=image_size,
            channels=channels,
        ),
        num_parallel_calls=AUTOTUNE,
    )

    if cache:
        ds = ds.cache()

    if augment_fn is not None:
        ds = ds.map(
            lambda x, y: (augment_fn(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_datasets_from_split_csvs(
    split_dir: str | Path,
    data_root: str | Path | None = None,
    batch_size: int = BATCH_SIZE,
    augment_fn: Callable | None = None,
    cache: bool = CACHE_DATASET,
    image_size: tuple[int, int] = IMAGE_SIZE,
    channels: int = 1,
):
    split_dir = Path(split_dir)
    data_root = Path(data_root) if data_root is not None else RAW_DIR

    train_df = read_split_csv(split_dir / "train.csv")
    val_df = read_split_csv(split_dir / "val.csv")
    test_df = read_split_csv(split_dir / "test.csv")

    train_ds = build_dataset_from_dataframe(
        train_df,
        root_dir=data_root,
        batch_size=batch_size,
        shuffle=True,
        augment_fn=augment_fn,
        cache=cache,
        image_size=image_size,
        channels=channels,
    )
    val_ds = build_dataset_from_dataframe(
        val_df,
        root_dir=data_root,
        batch_size=batch_size,
        shuffle=False,
        augment_fn=None,
        cache=cache,
        image_size=image_size,
        channels=channels,
    )
    test_ds = build_dataset_from_dataframe(
        test_df,
        root_dir=data_root,
        batch_size=batch_size,
        shuffle=False,
        augment_fn=None,
        cache=cache,
        image_size=image_size,
        channels=channels,
    )

    return train_ds, val_ds, test_ds


# =========================================================
# Diagnostics
# =========================================================

def summarize_split(split_dir: str | Path) -> dict[str, pd.DataFrame]:
    split_dir = Path(split_dir)

    result = {}
    for split_name in ["train", "val", "test"]:
        df = read_split_csv(split_dir / f"{split_name}.csv")
        summary = (
            df.groupby(["label", "class_name"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("label")
            .reset_index(drop=True)
        )
        result[split_name] = summary

    return result


def print_split_summary(split_dir: str | Path) -> None:
    summaries = summarize_split(split_dir)

    for split_name, summary_df in summaries.items():
        print(f"\n{split_name.title()} class distribution")
        print("-" * 72)
        total = int(summary_df["count"].sum())

        for _, row in summary_df.iterrows():
            pct = 100.0 * row["count"] / total if total > 0 else 0.0
            print(
                f"{int(row['label']):>2} | "
                f"{str(row['class_name']):<22} | "
                f"{int(row['count']):>6} | "
                f"{pct:6.2f}%"
            )
        print("-" * 72)
        print(f"Total: {total}")
