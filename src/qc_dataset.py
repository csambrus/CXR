# src/qc_dataset.py

# qc_dataset.py

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from src.config import BATCH_SIZE, CLASS_INFOS, IMAGE_SIZE, RAW_DIR, SEED, SPLITS_DIR, OUTPUT_DIR, ensure_dir
from src.dataloader import (
    build_datasets_from_split_csvs,
    build_metadata_dataframe,
    create_splits,
    read_split_csv,
)
from src.preprocessing import plot_random_pre_post_samples_per_class

def distribution_df_to_records(dist_df):
    """
    Pandas DataFrame -> JSON-kompatibilis lista.
    """
    if dist_df.empty:
        return []

    records = dist_df[["label", "class_key", "class_name", "count", "ratio"]].to_dict(orient="records")
    for rec in records:
        rec["label"] = int(rec["label"])
        rec["class_key"] = str(rec["class_key"])
        rec["class_name"] = str(rec["class_name"])
        rec["count"] = int(rec["count"])
        rec["ratio"] = float(rec["ratio"])
    return records


def get_class_distribution(df):
    dist_df = (
        df.groupby(["label", "class_key", "class_name"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("label")
        .reset_index(drop=True)
    )
    total = int(dist_df["count"].sum())
    if total > 0:
        dist_df["ratio"] = dist_df["count"] / total
    else:
        dist_df["ratio"] = 0.0
    return dist_df


def print_class_distribution(df, title="Class distribution"):
    dist_df = get_class_distribution(df)
    print("\n" + title)
    print("-" * 80)
    if dist_df.empty:
        print("[WARN] Empty dataframe.")
    else:
        print(dist_df.to_string(index=False))
    print("-" * 80)


def inspect_batch(ds):
    x_batch, y_batch = next(iter(ds.take(1)))
    print(f"images shape: {tuple(x_batch.shape)}")
    print(f"labels shape: {tuple(y_batch.shape)}")
    labels = y_batch.numpy().astype(int)
    uniq, counts = np.unique(labels, return_counts=True)
    print("label counts in batch:", {int(k): int(v) for k, v in zip(uniq, counts)})


# =========================================================
# Fő QC pipeline
# =========================================================

def run_dataset_qc(
    root_dir: str | Path = RAW_DIR,
    out_dir: str | Path | None = None,
    splits_dir: str | Path = SPLITS_DIR,
    test_size: float = 0.15,
    val_size: float = 0.15,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    n_preview_per_class: int = 4,
):
    root_dir = Path(root_dir)

    out_dir = out_dir or OUTPUT_DIR / "dataset_qc"
    out_dir = ensure_dir(out_dir)

    preview_dir = ensure_dir(out_dir / "previews")
    split_dir = ensure_dir(splits_dir)
    
    print("=" * 80)
    print("DATASET QC START")
    print("=" * 80)
    print(f"Root dir: {root_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Split dir: {split_dir}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Seed: {seed}")
    print(f"Classes: {[c.display_name for c in CLASS_INFOS]}")
    print("=" * 80)

    # -----------------------------------------------------
    # 1. Metadata build
    # -----------------------------------------------------
    print("\n[1/6] Building metadata dataframe...")
    full_df = build_metadata_dataframe(root_dir=root_dir)

    if full_df.empty:
        raise ValueError(f"Nem találtam képeket itt: {root_dir}")

    print_class_distribution(full_df, title="Full dataset class distribution")

    full_df.to_csv(out_dir / "full_metadata.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'full_metadata.csv'}")

    full_dist_df = get_class_distribution(full_df)
    full_dist_df.to_csv(out_dir / "full_distribution.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'full_distribution.csv'}")

    # -----------------------------------------------------
    # 2. Split + tf.data datasets
    # -----------------------------------------------------
    print("\n[2/6] Building train/val/test splits and tf.data datasets...")
    create_splits(
        source_root=root_dir,
        split_dir=split_dir,
        train_size=1.0 - (test_size + val_size),
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        overwrite=True,
    )

    # A dataset építés ugyanazt a split CSV-t használja, ezért a későbbi
    # minőségellenőrzési statisztikák (train/val/test) reprodukálhatóak.
    train_ds, val_ds, test_ds = build_datasets_from_split_csvs(
        split_dir=split_dir,
        data_root=root_dir,
        batch_size=batch_size,
        image_size=image_size,
        channels=1,
    )
    train_df = read_split_csv(split_dir / "train.csv")
    val_df = read_split_csv(split_dir / "val.csv")
    test_df = read_split_csv(split_dir / "test.csv")

    # -----------------------------------------------------
    # 3. Preview képek mentése
    # -----------------------------------------------------
    print("\n[3/6] Saving raw vs processed previews...")
    raw_proc_path = preview_dir / "pre_post_examples.png"
    plot_random_pre_post_samples_per_class(
        root_dir=root_dir,
        image_size=image_size,
        n_per_class=n_preview_per_class,
        seed=seed,
        augment_preview=False,
        save_path=raw_proc_path,
    )
    print(f"[INFO] Saved: {raw_proc_path}")

    print("\n[4/6] Saving raw vs processed+augmented previews...")
    raw_proc_aug_path = preview_dir / "pre_post_aug_examples.png"
    plot_random_pre_post_samples_per_class(
        root_dir=root_dir,
        image_size=image_size,
        n_per_class=n_preview_per_class,
        seed=seed,
        augment_preview=True,
        save_path=raw_proc_aug_path,
    )
    print(f"[INFO] Saved: {raw_proc_aug_path}")

    # -----------------------------------------------------
    # 4. Batch sanity check
    # -----------------------------------------------------
    print("\n[5/6] Batch sanity check...")
    print("\nTrain batch:")
    inspect_batch(train_ds)

    print("\nValidation batch:")
    inspect_batch(val_ds)

    print("\nTest batch:")
    inspect_batch(test_ds)

    # -----------------------------------------------------
    # 5. Summary JSON
    # -----------------------------------------------------
    print("\n[6/6] Saving dataset summary JSON...")
    train_dist_df = get_class_distribution(train_df)
    val_dist_df = get_class_distribution(val_df)
    test_dist_df = get_class_distribution(test_df)

    # Ez a JSON a teljes QC futás "forrás-igazsága": paraméterek, eloszlások
    # és artefakt útvonalak egy helyen, hogy később auditálható legyen.
    summary = {
        "root_dir": str(root_dir),
        "out_dir": str(out_dir),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "batch_size": int(batch_size),
        "seed": int(seed),
        "n_classes": int(len(CLASS_INFOS)),
        "classes": [
            {
                "key": c.key,
                "raw_dir": c.raw_dir,
                "display_name": c.display_name,
                "idx": int(c.idx),
            }
            for c in CLASS_INFOS
        ],
        "split": {
            "test_size": float(test_size),
            "val_size": float(val_size),
            "full_count": int(len(full_df)),
            "train_count": int(len(train_df)),
            "val_count": int(len(val_df)),
            "test_count": int(len(test_df)),
        },
        "distributions": {
            "full": distribution_df_to_records(full_dist_df),
            "train": distribution_df_to_records(train_dist_df),
            "val": distribution_df_to_records(val_dist_df),
            "test": distribution_df_to_records(test_dist_df),
        },
        "artifacts": {
            "full_metadata_csv": str(out_dir / "full_metadata.csv"),
            "full_distribution_csv": str(out_dir / "full_distribution.csv"),
            "split_train_csv": str(split_dir / "train.csv"),
            "split_val_csv": str(split_dir / "val.csv"),
            "split_test_csv": str(split_dir / "test.csv"),
            "distribution_train_csv": str(out_dir / "distribution_train.csv"),
            "distribution_val_csv": str(out_dir / "distribution_val.csv"),
            "distribution_test_csv": str(out_dir / "distribution_test.csv"),
            "preview_raw_processed": str(raw_proc_path),
            "preview_raw_processed_augmented": str(raw_proc_aug_path),
        },
    }

    summary_path = out_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved: {summary_path}")

    print("\n" + "=" * 80)
    print("DATASET QC DONE")
    print("=" * 80)

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "summary_path": summary_path,
        "preview_raw_processed": raw_proc_path,
        "preview_raw_processed_augmented": raw_proc_aug_path,
    }


# =========================================================
# Main
# =========================================================

def main():
    run_dataset_qc(
        root_dir=RAW_DIR,
        out_dir=OUTPUT_DIR / "dataset_qc",
        test_size=0.15,
        val_size=0.15,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        n_preview_per_class=4,
    )


if __name__ == "__main__":
    main()