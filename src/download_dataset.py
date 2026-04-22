from __future__ import annotations

import shutil
from pathlib import Path
import kagglehub

from src.config import RAW_DIR, DATA_DIR, SEG_DIR, ensure_dir


# =========================================================
# MAIN CLASSIFIER DATASET
# =========================================================

COVID_SLUG = "unaissait/curated-chest-xray-image-dataset-for-covid19"

TMP_DIR = RAW_DIR / "tmp"
COVID_READY_MARKER = RAW_DIR / ".dataset_ready"

COVID_EXPECTED = [
    "COVID-19",
    "Normal",
    "Pneumonia-Bacterial",
    "Pneumonia-Viral",
]


# =========================================================
# SEGMENTATION DATASETS
# =========================================================

MONT_DIR = SEG_DIR / "montgomery"
SHEN_DIR = SEG_DIR / "shenzhen"

MONT_SLUG = "nih-chest-xray/montgomery-cxr"
SHEN_SLUG = "nih-chest-xray/shenzhen-cxr"

MONT_MARKER = MONT_DIR / ".ready"
SHEN_MARKER = SHEN_DIR / ".ready"


# =========================================================
# HELPERS
# =========================================================

def is_ready(folder: Path, marker: Path) -> bool:
    return marker.exists() and folder.exists()


def safe_download(slug: str, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    print(f"[INFO] Downloading {slug}")
    path = kagglehub.dataset_download(slug, output_dir=out_dir)
    print("[INFO] Downloaded to:", path)
    return Path(path)


# =========================================================
# COVID DATASET
# =========================================================

def covid_exists() -> bool:
    if COVID_READY_MARKER.exists():
        return True
    return all((RAW_DIR / cls).exists() for cls in COVID_EXPECTED)


def download_covid():
    if covid_exists():
        print("[SKIP] COVID dataset already exists.")
        return

    ensure_dir(TMP_DIR)

    safe_download(COVID_SLUG, TMP_DIR)

    src_dir = TMP_DIR / "Curated X-Ray Dataset"

    if not src_dir.exists():
        raise RuntimeError(f"Missing: {src_dir}")

    for item in src_dir.iterdir():
        target = RAW_DIR / item.name

        if target.exists():
            continue

        shutil.move(str(item), str(target))

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    COVID_READY_MARKER.touch()

    print("[OK] COVID dataset ready:", RAW_DIR)


# =========================================================
# MONTGOMERY
# =========================================================

def download_montgomery():
    if is_ready(MONT_DIR, MONT_MARKER):
        print("[SKIP] Montgomery already exists.")
        return

    ensure_dir(MONT_DIR)
    safe_download(MONT_SLUG, MONT_DIR)

    MONT_MARKER.touch()
    print("[OK] Montgomery ready:", MONT_DIR)


# =========================================================
# SHENZHEN
# =========================================================

def download_shenzhen():
    if is_ready(SHEN_DIR, SHEN_MARKER):
        print("[SKIP] Shenzhen already exists.")
        return

    ensure_dir(SHEN_DIR)
    safe_download(SHEN_SLUG, SHEN_DIR)

    SHEN_MARKER.touch()
    print("[OK] Shenzhen ready:", SHEN_DIR)


# =========================================================
# ALL
# =========================================================

def download_dataset(
    include_segmentation: bool = True,
):
    download_covid()

    if include_segmentation:
        download_montgomery()
        download_shenzhen()


if __name__ == "__main__":
    download_dataset()
