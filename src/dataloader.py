# src/dataloader.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import config


# =========================================================
# Alap utility-k
# =========================================================
def get_class_to_index() -> dict:
    """Osztálynév -> index mapping."""
    return {class_name: idx for idx, class_name in enumerate(config.CLASS_NAMES)}


def get_index_to_class() -> dict:
    """Index -> osztálynév mapping."""
    return {idx: class_name for idx, class_name in enumerate(config.CLASS_NAMES)}


def is_supported_image_file(path: Path) -> bool:
    """Eldönti, hogy támogatott képfájl-e."""
    return path.suffix.lower() in config.SUPPORTED_EXTENSIONS


# =========================================================
# Fájlok összegyűjtése raw mappából
# =========================================================
def collect_image_paths(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Bejárja a raw datasetet, és összegyűjti a képeket.

    Elvárt struktúra:
        data/raw/
            normal/
            pneumonia_viral/
            pneumonia_bacterial/
            covid19/

    Returns
    -------
    pd.DataFrame
        Oszlopok:
        - filepath
        - class_name
        - label
        - filename
    """
    raw_dir = raw_dir or config.RAW_DIR
    class_to_index = get_class_to_index()

    records = []

    for class_name in config.CLASS_NAMES:
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            print(f"[WARN] Hiányzó osztálykönyvtár: {class_dir}")
            continue

        for file_path in sorted(class_dir.rglob("*")):
            if file_path.is_file() and is_supported_image_file(file_path):
                records.append(
                    {
                        "filepath": str(file_path.resolve()),
                        "class_name": class_name,
                        "label": class_to_index[class_name],
                        "filename": file_path.name,
                    }
                )

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            f"Nem találtam képeket a raw datasetben: {raw_dir}\n"
            f"Ellenőrizd a könyvtárstruktúrát és a fájlokat."
        )

    return df


# =========================================================
# Split készítés
# =========================================================
def make_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / val / test split.

    Fontos: train_ratio + val_ratio + test_ratio = 1.0
    """
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"A split arányok összege nem 1.0: {train_ratio} + {val_ratio} + {test_ratio} = {total}"
        )

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        stratify=df["label"],
        random_state=random_state,
        shuffle=True,
    )

    # a maradékot bontjuk val/test-re
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df["label"],
        random_state=random_state,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def save_splits_to_csv(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Elmenti a split DataFrame-eket CSV-be."""
    config.ensure_project_dirs()

    train_df.to_csv(config.TRAIN_CSV, index=False)
    val_df.to_csv(config.VAL_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)

    print(f"[INFO] Train split mentve: {config.TRAIN_CSV}")
    print(f"[INFO] Val split mentve:   {config.VAL_CSV}")
    print(f"[INFO] Test split mentve:  {config.TEST_CSV}")


def load_split_csv(csv_path: Path) -> pd.DataFrame:
    """Split CSV visszatöltése."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Nem található a split fájl: {csv_path}")
    return pd.read_csv(csv_path)


# =========================================================
# Képfeldolgozás tf.data-hoz
# =========================================================
def decode_and_resize_image(
    filepath: tf.Tensor,
    label: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Kép beolvasása, dekódolása, RGB-re alakítása, resize.
    """
    image_bytes = tf.io.read_file(filepath)
    image = tf.image.decode_image(image_bytes, channels=config.IMAGE_CHANNELS, expand_animations=False)
    image = tf.image.resize(image, config.IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    if config.NORMALIZE_TO_0_1:
        image = image / 255.0

    return image, label


def get_basic_augmentation() -> tf.keras.Sequential:
    """
    Egyszerű, visszafogott augmentáció.
    Orvosi képeknél direkt nem agresszív.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(0.05),
            tf.keras.layers.RandomTranslation(height_factor=0.02, width_factor=0.02),
        ],
        name="basic_augmentation",
    )


def apply_augmentation(
    image: tf.Tensor,
    label: tf.Tensor,
    augmenter: tf.keras.Sequential,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Augmentáció alkalmazása."""
    image = augmenter(image, training=True)
    return image, label


# =========================================================
# tf.data Dataset építés
# =========================================================
def dataframe_to_dataset(
    df: pd.DataFrame,
    training: bool = False,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
    augment: bool = False,
) -> tf.data.Dataset:
    """
    DataFrame -> tf.data.Dataset
    """
    filepaths = df["filepath"].astype(str).values
    labels = df["label"].astype(np.int32).values

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if training and shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    ds = ds.map(decode_and_resize_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training and augment:
        augmenter = get_basic_augmentation()
        ds = ds.map(
            lambda x, y: apply_augmentation(x, y, augmenter),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(batch_size)

    if config.PREFETCH_DATASET:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def build_datasets_from_raw(
    save_csv: bool = True,
    batch_size: int = config.BATCH_SIZE,
    augment_train: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Teljes pipeline:
    - raw bejárása
    - DataFrame építés
    - stratified split
    - opcionális CSV mentés
    - tf.data datasetek létrehozása

    Returns
    -------
    train_ds, val_ds, test_ds, train_df, val_df, test_df
    """
    df = collect_image_paths(config.RAW_DIR)
    train_df, val_df, test_df = make_train_val_test_split(df)

    if save_csv:
        save_splits_to_csv(train_df, val_df, test_df)

    train_ds = dataframe_to_dataset(
        train_df,
        training=True,
        batch_size=batch_size,
        shuffle=True,
        augment=augment_train,
    )

    val_ds = dataframe_to_dataset(
        val_df,
        training=False,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
    )

    test_ds = dataframe_to_dataset(
        test_df,
        training=False,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
    )

    return train_ds, val_ds, test_ds, train_df, val_df, test_df


def build_datasets_from_csv(
    batch_size: int = config.BATCH_SIZE,
    augment_train: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Már elmentett split CSV-kből épít tf.data dataseteket.
    """
    train_df = load_split_csv(config.TRAIN_CSV)
    val_df = load_split_csv(config.VAL_CSV)
    test_df = load_split_csv(config.TEST_CSV)

    train_ds = dataframe_to_dataset(
        train_df,
        training=True,
        batch_size=batch_size,
        shuffle=True,
        augment=augment_train,
    )

    val_ds = dataframe_to_dataset(
        val_df,
        training=False,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
    )

    test_ds = dataframe_to_dataset(
        test_df,
        training=False,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
    )

    return train_ds, val_ds, test_ds, train_df, val_df, test_df


# =========================================================
# Gyors statisztika
# =========================================================
def summarize_dataframe(df: pd.DataFrame, name: str = "dataset") -> None:
    """Kiír egy rövid osztályeloszlás összefoglalót."""
    print(f"\n===== {name.upper()} =====")
    print(f"Elemszám: {len(df)}")
    print(df["class_name"].value_counts().sort_index())
    print()


if __name__ == "__main__":
    config.ensure_project_dirs()

    train_ds, val_ds, test_ds, train_df, val_df, test_df = build_datasets_from_raw(
        save_csv=True,
        batch_size=config.BATCH_SIZE,
        augment_train=True,
    )

    summarize_dataframe(train_df, "train")
    summarize_dataframe(val_df, "validation")
    summarize_dataframe(test_df, "test")

    for images, labels in train_ds.take(1):
        print(f"Train batch image shape: {images.shape}")
        print(f"Train batch label shape: {labels.shape}")
