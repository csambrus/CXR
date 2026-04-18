# src/download_dataset.py
from __future__ import annotations

import shutil
import zipfile
import subprocess

from pathlib import Path
import kagglehub
import config


DATASET_SLUG = "unaissait/curated-chest-xray-image-dataset-for-covid19"
ZIP_NAME = "curated-chest-xray-image-dataset-for-covid19.zip"

def download_dataset() -> Path:
    # Download latest version
    path = kagglehub.dataset_download(DATASET_SLUG, output_dir = RAW_DIR)
    print("Path to dataset files:", path)
    return(path)

def run_command(cmd: list[str]) -> None:
    """Parancs futtatása hibakezeléssel."""
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_dataset_zip() -> Path:
    """
    Letölti a Kaggle dataset zip fájlt a data könyvtárba.

    Returns
    -------
    Path
        A letöltött zip fájl útvonala.
    """
    config.ensure_project_dirs()
    download_dir = config.DATA_DIR / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    run_command([
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_SLUG,
        "-p",
        str(download_dir),
    ])

    zip_path = download_dir / ZIP_NAME
    if not zip_path.exists():
        raise FileNotFoundError(f"Nem található a letöltött zip: {zip_path}")

    print(f"[INFO] Letöltve: {zip_path}")
    return zip_path


def extract_dataset(zip_path: Path) -> Path:
    """
    Kicsomagolja a datasetet.

    Returns
    -------
    Path
        A kicsomagolt célmappa.
    """
    extract_dir = config.DATA_DIR / "raw_download"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    print(f"[INFO] Kicsomagolva ide: {extract_dir}")
    return extract_dir


def main() -> None:
    zip_path = download_dataset()
    extract_dataset(zip_path)


if __name__ == "__main__":
    main()
