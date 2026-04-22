# src/download_dataset.py
from __future__ import annotations

import shutil
from pathlib import Path
import kagglehub

from src.config import RAW_DIR, DATA_DIR, ensure_dir


# =========================================================
# DATASET SLUGS
# =========================================================

# Classifier dataset (4 classes)
COVID_SLUG = "unaissait/curated-chest-xray-image-dataset-for-covid19"

# Segmentation dataset (lung masks)
CRD_SLUG = "mrunalnshah/crd-chest-x-ray-images-with-lung-segmented-masks"


# =========================================================
# PATHS
# =========================================================

# Main classifier dataset
COVID_TMP_DIR = RAW_DIR / "_tmp_download"
COVID_READY_MARKER = RAW_DIR / ".dataset_ready"

COVID_EXPECTED_FOLDERS = [
    "COVID-19",
    "Normal",
    "Pneumonia-Bacterial",
    "Pneumonia-Viral",
]

# Segmentation dataset
SEG_RAW_DIR = DATA_DIR / "segmentation_raw"
CRD_DIR = SEG_RAW_DIR / "crd_lung_masks"
CRD_READY_MARKER = CRD_DIR / ".dataset_ready"


# =========================================================
# HELPERS
# =========================================================

def safe_download(slug: str, out_dir: Path) -> Path:
    """
    Downloads dataset with kagglehub.
    Requires Kaggle authentication in Colab / local environment.
    """
    ensure_dir(out_dir)

    print("=" * 72)
    print(f"[INFO] Downloading dataset: {slug}")
    print(f"[INFO] Output dir        : {out_dir}")
    print("=" * 72)

    try:
        path = kagglehub.dataset_download(
            slug,
            output_dir=out_dir,
        )
    except Exception as e:
        raise RuntimeError(
            f"\n[ERROR] Failed to download dataset: {slug}\n"
            "Possible reasons:\n"
            "1. Missing Kaggle authentication (kaggle.json)\n"
            "2. Dataset requires Kaggle consent\n"
            "3. Wrong dataset slug\n"
            f"\nOriginal error:\n{e}"
        ) from e

    path = Path(path)

    print("[OK] Download finished:", path)
    return path


def touch(path: Path):
    ensure_dir(path.parent)
    path.touch(exist_ok=True)


def folder_has_any_files(folder: Path) -> bool:
    if not folder.exists():
        return False

    for p in folder.rglob("*"):
        if p.is_file():
            return True
    return False


# =========================================================
# CLASSIFIER DATASET CHECK
# =========================================================

def covid_exists() -> bool:
    """
    Checks if the main 4-class classifier dataset already exists.
    """
    if COVID_READY_MARKER.exists():
        return True

    return all((RAW_DIR / name).exists() for name in COVID_EXPECTED_FOLDERS)


# =========================================================
# SEGMENTATION DATASET CHECK
# =========================================================

def crd_exists() -> bool:
    """
    Checks if CRD segmentation dataset already exists.
    """
    if CRD_READY_MARKER.exists():
        return True

    return folder_has_any_files(CRD_DIR)


# =========================================================
# DOWNLOAD CLASSIFIER DATASET
# =========================================================

def move_classifier_dataset(tmp_root: Path):
    """
    Kaggle dataset typically extracts into:
    Curated X-Ray Dataset/
    """
    possible_roots = [
        tmp_root / "Curated X-Ray Dataset",
        tmp_root,
    ]

    src_root = None
    for p in possible_roots:
        if p.exists():
            src_root = p
            break

    if src_root is None:
        raise RuntimeError(
            f"[ERROR] Could not locate extracted classifier dataset in: {tmp_root}"
        )

    print("[INFO] Moving classifier dataset into RAW_DIR...")

    for item in src_root.iterdir():
        if not item.is_dir():
            continue

        target = RAW_DIR / item.name

        if target.exists():
            print(f"[SKIP] Already exists: {target}")
            continue

        shutil.move(str(item), str(target))
        print(f"[OK] Moved: {item.name}")

    touch(COVID_READY_MARKER)


def download_classifier_dataset(force: bool = False):
    """
    Downloads the COVID / pneumonia classifier dataset.
    """
    if covid_exists() and not force:
        print("[SKIP] Classifier dataset already exists.")
        return

    if force and RAW_DIR.exists():
        print("[INFO] Force mode enabled.")

    ensure_dir(RAW_DIR)
    ensure_dir(COVID_TMP_DIR)

    safe_download(COVID_SLUG, COVID_TMP_DIR)
    move_classifier_dataset(COVID_TMP_DIR)

    shutil.rmtree(COVID_TMP_DIR, ignore_errors=True)

    print("[OK] Classifier dataset ready:", RAW_DIR)


# =========================================================
# DOWNLOAD SEGMENTATION DATASET
# =========================================================

def download_segmentation_dataset(force: bool = False):
    """
    Downloads CRD lung segmentation dataset.
    """
    if crd_exists() and not force:
        print("[SKIP] Segmentation dataset already exists.")
        return

    ensure_dir(CRD_DIR)

    safe_download(CRD_SLUG, CRD_DIR)

    touch(CRD_READY_MARKER)

    print("[OK] Segmentation dataset ready:", CRD_DIR)


# =========================================================
# SUMMARY
# =========================================================

def print_dataset_summary():
    print("\n" + "=" * 72)
    print("DATASET SUMMARY")
    print("=" * 72)

    print("Classifier dataset:")
    print(" Ready:", covid_exists())
    print(" Path :", RAW_DIR)

    print("\nSegmentation dataset:")
    print(" Ready:", crd_exists())
    print(" Path :", CRD_DIR)

    print("=" * 72)


# =========================================================
# MAIN ENTRY
# =========================================================

def download_dataset(
    include_classifier: bool = True,
    include_segmentation: bool = True,
    force: bool = False,
):
    """
    Downloads requested datasets.

    Example:
        download_dataset()

    Only segmentation:
        download_dataset(
            include_classifier=False,
            include_segmentation=True,
        )
    """
    if include_classifier:
        download_classifier_dataset(force=force)

    if include_segmentation:
        download_segmentation_dataset(force=force)

    print_dataset_summary()


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    download_dataset(
        include_classifier=True,
        include_segmentation=True,
        force=False,
    )
