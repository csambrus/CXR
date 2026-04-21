# dataloader.py

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

from src.config import (
    BATCH_SIZE,
    CLASS_BY_IDX,
    CLASS_INFOS,
    IMAGE_SIZE,
    RAW_DIR,
    SEED,
    CACHE_DATASET,
    ClassInfo,
    ensure_dir
)
from src.preprocessing import (
    build_classification_dataset,
    apply_batch_augmentation,
    get_all_image_files,
    get_class_dir,
)

# =========================================================
# Metadata tábla építés
# =========================================================

def build_metadata_dataframe(
    root_dir: str | Path = RAW_DIR,
    class_infos: Sequence[ClassInfo] = CLASS_INFOS,
) -> pd.DataFrame:
    """
    A root_dir alatt a class mappákból metadata DataFrame-et épít.

    Kimeneti oszlopok:
        filepath
        filename
        class_key
        class_name
        label
        raw_dir
    """
    root_dir = Path(root_dir)

    rows: list[dict] = []

    for class_info in class_infos:
        class_dir = get_class_dir(root_dir, class_info)

        if not class_dir.exists():
            print(f"[WARN] Missing class dir: {class_dir}")
            continue

        files = get_all_image_files(class_dir)
        print(f"[INFO] {class_info.display_name}: {len(files)} image(s)")

        for fp in files:
            rows.append(
                {
                    "filepath": str(fp),
                    "filename": fp.name,
                    "class_key": class_info.key,
                    "class_name": class_info.display_name,
                    "label": class_info.idx,
                    "raw_dir": class_info.raw_dir,
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        print(f"[WARN] No image files found under: {root_dir}")
        return df

    df = df.sort_values(["label", "filepath"]).reset_index(drop=True)
    return df


# =========================================================
# QC / class distribution
# =========================================================

def get_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Osztályeloszlás DataFrame-et ad vissza.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["label", "class_key", "class_name", "count", "ratio"]
        )

    dist = (
        df.groupby(["label", "class_key", "class_name"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("label")
        .reset_index(drop=True)
    )
    dist["ratio"] = dist["count"] / dist["count"].sum()
    return dist


def print_class_distribution(df: pd.DataFrame, title: str = "Class distribution") -> None:
    print(f"\n{title}")
    print("-" * 72)

    if df.empty:
        print("[WARN] Empty dataframe.")
        print("-" * 72)
        return

    dist = get_class_distribution(df)

    for _, row in dist.iterrows():
        print(
            f"{int(row['label']):>2} | "
            f"{row['class_key']:<22} | "
            f"{row['class_name']:<22} | "
            f"{int(row['count']):>6} | "
            f"{row['ratio'] * 100:>6.2f}%"
        )

    print("-" * 72)
    print(f"Total: {len(df)}")


# =========================================================
# Split
# =========================================================

def split_metadata_dataframe(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = SEED,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Kétlépcsős train / val / test split.

    Megjegyzés:
    - test_size és val_size a teljes dataset arányai
    - a val split a maradék trainből kerül leválasztásra úgy,
      hogy globálisan kb. a megadott arány jöjjön ki

    Példa:
        test_size=0.15, val_size=0.15
        => kb. 70 / 15 / 15
    """
    if df.empty:
        raise ValueError("A metadata dataframe üres, nem lehet splitelni.")

    if test_size <= 0 or test_size >= 1:
        raise ValueError("A test_size értéke 0 és 1 közé kell essen.")
    if val_size <= 0 or val_size >= 1:
        raise ValueError("A val_size értéke 0 és 1 közé kell essen.")
    if test_size + val_size >= 1:
        raise ValueError("A test_size + val_size összege legyen 1-nél kisebb.")

    stratify_labels = df["label"] if stratify else None

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_labels,
    )

    remaining = 1.0 - test_size
    val_relative = val_size / remaining

    stratify_labels_2 = train_val_df["label"] if stratify else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative,
        random_state=seed,
        stratify=stratify_labels_2,
    )

    train_df = train_df.sort_values(["label", "filepath"]).reset_index(drop=True)
    val_df = val_df.sort_values(["label", "filepath"]).reset_index(drop=True)
    test_df = test_df.sort_values(["label", "filepath"]).reset_index(drop=True)

    return train_df, val_df, test_df


# =========================================================
# Split QC
# =========================================================

def print_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    total = len(train_df) + len(val_df) + len(test_df)

    print("\nSplit summary")
    print("-" * 72)
    print(f"Train: {len(train_df):>6} ({100 * len(train_df) / max(total, 1):>6.2f}%)")
    print(f"Val:   {len(val_df):>6} ({100 * len(val_df) / max(total, 1):>6.2f}%)")
    print(f"Test:  {len(test_df):>6} ({100 * len(test_df) / max(total, 1):>6.2f}%)")
    print(f"Total: {total:>6}")
    print("-" * 72)

    print_class_distribution(train_df, title="Train class distribution")
    print_class_distribution(val_df, title="Validation class distribution")
    print_class_distribution(test_df, title="Test class distribution")


# =========================================================
# Export
# =========================================================

def export_split_csvs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: str | Path,
    prefix: str = "split",
) -> None:
    out_dir = ensure_dir(out_dir)

    train_path = out_dir / f"{prefix}_train.csv"
    val_path = out_dir / f"{prefix}_val.csv"
    test_path = out_dir / f"{prefix}_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[INFO] Saved: {train_path}")
    print(f"[INFO] Saved: {val_path}")
    print(f"[INFO] Saved: {test_path}")


def export_distribution_csvs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: str | Path,
    prefix: str = "distribution",
) -> None:
    out_dir = ensure_dir(out_dir)

    get_class_distribution(train_df).to_csv(
        out_dir / f"{prefix}_train.csv", index=False
    )
    get_class_distribution(val_df).to_csv(
        out_dir / f"{prefix}_val.csv", index=False
    )
    get_class_distribution(test_df).to_csv(
        out_dir / f"{prefix}_test.csv", index=False
    )

    print(f"[INFO] Saved distribution CSVs to: {out_dir}")


# =========================================================
# DataFrame -> tf.data
# =========================================================

def dataframe_to_dataset(
    df: pd.DataFrame,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    training: bool = False,
    shuffle: bool = True,
    seed: int = SEED,
):
    if df.empty:
        raise ValueError("Az input DataFrame üres, nem lehet datasetet építeni.")

    ds = build_classification_dataset(
        filepaths=df["filepath"].tolist(),
        labels=df["label"].tolist(),
        image_size=image_size,
        training=training,
        shuffle=shuffle,
        seed=seed,
    )

    if CACHE_DATASET:
        ds = ds.cache()
    
    ds = ds.batch(batch_size)

    if training:
        ds = apply_batch_augmentation(ds)
    
    ds = ds.prefetch(tf.data.AUTOTUNE)   
    return ds


# =========================================================
# High-level helper
# =========================================================

def build_datasets_from_root(
    root_dir: str | Path = RAW_DIR,
    test_size: float = 0.15,
    val_size: float = 0.15,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    stratify: bool = True,
    export_dir: str | Path | None = None,
):
    """
    Komplett pipeline:
    - metadata df építés
    - split
    - tf.data datasetek létrehozása
    - opcionális CSV export

    Visszatérés:
        train_ds, val_ds, test_ds, train_df, val_df, test_df
    """
    df = build_metadata_dataframe(root_dir=root_dir)

    if df.empty:
        raise ValueError(f"Nem találtam használható képeket itt: {root_dir}")

    print_class_distribution(df, title="Full dataset class distribution")

    train_df, val_df, test_df = split_metadata_dataframe(
        df,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
        stratify=stratify,
    )

    print_split_summary(train_df, val_df, test_df)

    if export_dir is not None:
        export_split_csvs(train_df, val_df, test_df, export_dir, prefix="split")
        export_distribution_csvs(
            train_df, val_df, test_df, export_dir, prefix="distribution"
        )

    train_ds = dataframe_to_dataset(
        train_df,
        image_size=image_size,
        batch_size=batch_size,
        training=True,
        shuffle=True,
        seed=seed,
    )

    val_ds = dataframe_to_dataset(
        val_df,
        image_size=image_size,
        batch_size=batch_size,
        training=False,
        shuffle=False,
        seed=seed,
    )

    test_ds = dataframe_to_dataset(
        test_df,
        image_size=image_size,
        batch_size=batch_size,
        training=False,
        shuffle=False,
        seed=seed,
    )

    return train_ds, val_ds, test_ds, train_df, val_df, test_df


# =========================================================
# Beolvasás meglévő split CSV-kből
# =========================================================

def load_split_csvs(
    split_dir: str | Path,
    prefix: str = "split",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_dir = Path(split_dir)

    train_path = split_dir / f"{prefix}_train.csv"
    val_path = split_dir / f"{prefix}_val.csv"
    test_path = split_dir / f"{prefix}_test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


def build_datasets_from_split_csvs(
    split_dir: str | Path,
    prefix: str = "split",
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
):
    train_df, val_df, test_df = load_split_csvs(split_dir, prefix=prefix)

    print_split_summary(train_df, val_df, test_df)

    train_ds = dataframe_to_dataset(
        train_df,
        image_size=image_size,
        batch_size=batch_size,
        training=True,
        shuffle=True,
        seed=seed,
    )

    val_ds = dataframe_to_dataset(
        val_df,
        image_size=image_size,
        batch_size=batch_size,
        training=False,
        shuffle=False,
        seed=seed,
    )

    test_ds = dataframe_to_dataset(
        test_df,
        image_size=image_size,
        batch_size=batch_size,
        training=False,
        shuffle=False,
        seed=seed,
    )

    return train_ds, val_ds, test_ds, train_df, val_df, test_df


# =========================================================
# Egyszerű sanity check
# =========================================================

def inspect_batch(ds, n_classes: int | None = None, sample_size: int = 16) -> None:
    """
    Egy véletlenszerű batch-et vizsgál meg, és abból random label mintát ír ki.
    """
    shuffle_buffer = 5000
    ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    
    for images, labels in ds.take(1):
        batch_size = images.shape[0]

        print(f"images shape: {images.shape}")
        print(f"labels shape: {labels.shape}")

        # Hány mintát írjunk ki max.
        k = min(sample_size, batch_size)

        # Véletlen indexek ismétlés nélkül
        idx = np.random.choice(batch_size, size=k, replace=False)

        sampled_labels = tf.gather(labels, idx).numpy().tolist()

        print(f"random labels sample: {sampled_labels}")

        # opcionálisan indexeket is kiírhatjuk
        print(f"sample indices: {idx.tolist()}")

        if n_classes is not None:
            print(f"n_classes: {n_classes}")

        break

# =========================================================
# Opcionális main
# =========================================================

if __name__ == "__main__":
    EXPORT_DIR = Path("outputs") / "splits"

    train_ds, val_ds, test_ds, train_df, val_df, test_df = build_datasets_from_root(
        root_dir=RAW_DIR,
        test_size=0.15,
        val_size=0.15,
        export_dir=EXPORT_DIR,
    )

    inspect_batch(train_ds, n_classes=len(CLASS_BY_IDX))
