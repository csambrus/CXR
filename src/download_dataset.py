# src/download_dataset.py
from __future__ import annotations

import shutil
from pathlib import Path
import kagglehub

from src.config import RAW_DIR 

DATASET_SLUG = "unaissait/curated-chest-xray-image-dataset-for-covid19"

TMP_DIR = RAW_DIR / "tmp"
KAGGLE_ZIP_DIR = TMP_DIR / "Curated X-Ray Dataset"
READY_MARKER = RAW_DIR / ".dataset_ready"

def dataset_exists() -> bool:
    if READY_MARKER.exists():
        return True

    # fallback check (ha marker hiányzik)
    # tipikusan class mappák vannak benne
    expected = ["COVID-19", "Normal", "Pneumonia-Bacterial", "Pneumonia-Viral"]
    return all((RAW_DIR / cls).exists() for cls in expected)

def download_from_kaggle() -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Downloading dataset...")
    path = kagglehub.dataset_download(DATASET_SLUG, output_dir=TMP_DIR)

    print("[INFO] Temp path:", path)
    return Path(path)

def move_dataset():
    # Kaggle zip miatt ez jön létre
    src_dir = KAGGLE_ZIP_DIR

    if not src_dir.exists():
        raise RuntimeError(f"[ERROR] Nem található: {src_dir}")

    print("[INFO] Moving dataset to RAW_DIR...")

    for item in src_dir.iterdir():
        target = RAW_DIR / item.name

        if target.exists():
            print(f"[SKIP] már létezik: {target}")
            continue

        shutil.move(str(item), str(target))

    # cleanup
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    # marker
    READY_MARKER.touch()

    print("[OK] Dataset ready at:", RAW_DIR)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def download_dataset() -> None:
    if dataset_exists():
        print("[SKIP] Dataset already exists, no download needed.")
        return

    download_from_kaggle()
    move_dataset()


if __name__ == "__main__":
    download_dataset()