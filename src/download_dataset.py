from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import kagglehub
from tqdm import tqdm

from src.config import GDRIVE_DATA, IS_COLAB, RAW_DIR, SEGMENTATION_RAW_DIR, ensure_dir


# =========================================================
# Kaggle dataset azonosítók
# =========================================================

COVID_CRD_SLUG = "unaissait/curated-chest-xray-image-dataset-for-covid19"
CRD_SEG_SLUG = "mrunalnshah/crd-chest-x-ray-images-with-lung-segmented-masks"


# =========================================================
# Marker fájlok
# =========================================================

COVID_READY_MARKER = RAW_DIR / ".dataset_ready"
SEG_READY_MARKER = SEGMENTATION_RAW_DIR / ".dataset_ready"

DRIVE_CACHE_ROOT = GDRIVE_DATA / "download_cache"
CLASSIFIER_CACHE_DIR = DRIVE_CACHE_ROOT / "classifier_dataset"
SEGMENTATION_CACHE_DIR = DRIVE_CACHE_ROOT / "segmentation_dataset"

# Colab: egy zip a Drive-on (kisméretű meta + kevés nagy I/O), nem több tízezer kis fájl másolása.
CLASSIFIER_ARCHIVE_ZIP = DRIVE_CACHE_ROOT / "classifier_unaissait_curated_cxr.zip"
SEGMENTATION_ARCHIVE_ZIP = DRIVE_CACHE_ROOT / "segmentation_mrunalnshah_crd.zip"


# =========================================================
# Általános utilok
# =========================================================

def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def remove_if_exists(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)


def _copy_large_file(src: Path, dst: Path, *, desc: str) -> None:
    """Egy nagy fájl másolása chunkokban tqdm mérettel (Drive ↔ helyi)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    size = max(src.stat().st_size, 1)
    chunk = 1024 * 1024 * 16
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        with tqdm(
            total=size,
            desc=desc,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            file=sys.stdout,
            mininterval=0.25,
        ) as bar:
            while True:
                buf = fsrc.read(chunk)
                if not buf:
                    break
                fdst.write(buf)
                bar.update(len(buf))


def _zip_directory_tree(src: Path, out_zip: Path, *, desc: str) -> None:
    """
    A teljes forrásfa egy zipbe. ZIP_STORED: PNG/JPEG úgysem tömörít sokat,
    a DEFLATED CPU+idő drága lenne.
    """
    files: list[tuple[Path, str]] = []
    for p in sorted(src.rglob("*")):
        if p.is_file():
            files.append((p, str(p.relative_to(src))))
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for path, arcname in tqdm(
            files,
            desc=desc,
            unit="file",
            file=sys.stdout,
            mininterval=0.2,
        ):
            zf.write(path, arcname=arcname)


def _unzip_to_dir(zip_path: Path, dest_dir: Path, *, desc: str) -> None:
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for info in tqdm(
            members,
            desc=desc,
            unit="file",
            file=sys.stdout,
            mininterval=0.15,
        ):
            if info.is_dir():
                continue
            target = (dest_dir / info.filename).resolve()
            try:
                target.relative_to(dest_dir)
            except ValueError as e:
                raise RuntimeError(f"[ERROR] Illegal zip path: {info.filename!r}") from e
            zf.extract(info, path=dest_dir)


def _iter_merge_copy_jobs(src: Path, dst: Path) -> Iterator[tuple[Path, Path]]:
    """
    Soronként adja a másolandó (forrás, cél) párokat — nem épít memóriában
    teljes listát (nagy fáknál ez percekig „néma” volt a régi megoldásnál).
    """
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    try:
        items = sorted(src.iterdir())
    except OSError:
        return
    for item in items:
        target = dst / item.name
        if item.is_dir():
            yield from _iter_merge_copy_jobs(item, target)
        elif not target.exists():
            yield (item, target)


def copytree_merge(
    src: Path,
    dst: Path,
    *,
    show_progress: bool = True,
    desc: str | None = None,
) -> None:
    """
    Python 3.8+ kompatibilis merge-szerű másolás.
    Ha a cél nem létezik, simán másol.
    Ha létezik, a hiányzó fájlokat/mappákat belemozgatja.

    Haladás: generátor + tqdm (stdout, nem notebook widget) — az első
    másolásnál azonnal látszik a sáv; nincs előzetes teljes fasor bejárás.
    """
    jobs = _iter_merge_copy_jobs(src, dst)
    label = desc or f"Copy {src.name}"
    if show_progress:
        jobs = tqdm(
            jobs,
            desc=label,
            unit="file",
            file=sys.stdout,
            mininterval=0.25,
            dynamic_ncols=True,
        )
    for s_path, t_path in jobs:
        t_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s_path, t_path)


def _has_any_files(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(path.iterdir())


# =========================================================
# Classifier dataset mozgatása
# =========================================================

def move_classifier_dataset(tmp_root: Path) -> None:
    """
    Kaggle dataset tipikusan így bontódik ki:

    tmp_root/
        Curated X-Ray Dataset/
            Normal/
            COVID-19/
            Pneumonia-Bacterial/
            Pneumonia-Viral/

    vagy esetenként közvetlenül a tmp_root alá.
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

    moved_any = False

    for item in src_root.iterdir():
        if not item.is_dir():
            continue

        target = RAW_DIR / item.name

        if target.exists():
            print(f"[SKIP] Already exists: {target}")
            continue

        shutil.move(str(item), str(target))
        print(f"[OK] Moved: {item.name}")
        moved_any = True

    if not moved_any:
        print("[WARN] No new classifier folders were moved.")

    touch(COVID_READY_MARKER)


# =========================================================
# Segmentation dataset mozgatása
# =========================================================

def move_segmentation_dataset(
    tmp_root: Path,
) -> None:
    """
    Várt forrásstruktúra például:

    tmp_root/
        crd_lung_masks/
            CXR_Combined/
                images/
                masks/
            CXR_RadioLucent/
            CXR_RadioLucent_masks/
            CXR_RadioOpaque/
            CXR_RadioOpaque_masks/

    A cél:
        SEGMENTATION_RAW_DIR/
            images/
            masks/
    """
    possible_roots = [
        tmp_root / "crd_lung_masks",
        tmp_root,
    ]

    src_root = None
    for p in possible_roots:
        if p.exists():
            src_root = p
            break

    if src_root is None:
        raise RuntimeError(
            f"[ERROR] Could not locate extracted segmentation dataset in: {tmp_root}"
        )

    print(f"[INFO] Segmentation source root: {src_root}")
    print("[INFO] Moving segmentation dataset into SEGMENTATION_RAW_DIR...")

    combined_dir = src_root / "CXR_Combined"
    src_images = combined_dir / "images"
    src_masks = combined_dir / "masks"

    if not combined_dir.exists():
        raise RuntimeError(f"[ERROR] Missing folder: {combined_dir}")
    if not src_images.exists():
        raise RuntimeError(f"[ERROR] Missing folder: {src_images}")
    if not src_masks.exists():
        raise RuntimeError(f"[ERROR] Missing folder: {src_masks}")

    dst_images = SEGMENTATION_RAW_DIR / "images"
    dst_masks = SEGMENTATION_RAW_DIR / "masks"

    if dst_images.exists():
        print(f"[SKIP] Already exists: {dst_images}")
    else:
        shutil.move(str(src_images), str(dst_images))
        print(f"[OK] Moved: {src_images} -> {dst_images}")

    if dst_masks.exists():
        print(f"[SKIP] Already exists: {dst_masks}")
    else:
        shutil.move(str(src_masks), str(dst_masks))
        print(f"[OK] Moved: {src_masks} -> {dst_masks}")

    if not dst_images.exists():
        raise RuntimeError(f"[ERROR] Target images dir missing: {dst_images}")
    if not dst_masks.exists():
        raise RuntimeError(f"[ERROR] Target masks dir missing: {dst_masks}")

    touch(SEG_READY_MARKER)
    print("[OK] Segmentation dataset structure verified.")
# =========================================================
# Letöltés
# =========================================================

def _download_to_temp(
    slug: str,
    cache_dir: Path | None = None,
    drive_archive_zip: Path | None = None,
) -> Path:
    """
    Ideiglenes könyvtár, ahonnan a ``move_*`` függvények a RAW struktúrába mozgatnak.

    Colab + ``drive_archive_zip``: egy zip a Drive-on → egy nagy fájl másolása +
    helyi kicsomagolás (sokkal kevesebb kis fájl I/O a Drive-on, mint a régi
    mappa-cache). Első futás: KaggleHub letöltés → helyi zip → Drive-ra egy
    ``copy`` → kicsomagolás a tempbe.

    Ha még a régi, kibontott ``cache_dir`` létezik zip nélkül, arról továbbra is
    működik a fájlmásolás (lassabb útvonal).
    """
    tmp_work = Path(tempfile.mkdtemp(prefix="cxr_download_"))
    print(f"[INFO] Temporary working dir: {tmp_work}", flush=True)

    if cache_dir is not None and drive_archive_zip is not None:
        ensure_dir(DRIVE_CACHE_ROOT)
        ensure_dir(cache_dir)

        if drive_archive_zip.exists():
            print(
                f"[INFO] Using Drive zip archive (single-file copy): {drive_archive_zip}",
                flush=True,
            )
            staging = tmp_work / "drive_dataset.zip"
            _copy_large_file(
                drive_archive_zip,
                staging,
                desc="Copy zip ← Drive",
            )
            print("[INFO] Extracting archive under temp dir...", flush=True)
            _unzip_to_dir(staging, tmp_work, desc="Unzip dataset")
            staging.unlink(missing_ok=True)
            return tmp_work

        if _has_any_files(cache_dir):
            print(
                f"[INFO] Legacy folder cache on Drive (many small files): {cache_dir}",
                flush=True,
            )
            print(
                "[INFO] Copying to temp (slow on Drive). "
                "Optional: remove this folder so the next run uses a .zip cache only.",
                flush=True,
            )
            copytree_merge(
                cache_dir,
                tmp_work,
                desc="Copy dataset folder → temp",
            )
            return tmp_work

        downloaded_path = Path(kagglehub.dataset_download(slug))
        print(f"[INFO] KaggleHub path: {downloaded_path}", flush=True)
        if not downloaded_path.is_dir():
            raise RuntimeError(
                f"[ERROR] Downloaded path is not a directory: {downloaded_path}"
            )

        staging_zip = tmp_work / ".staging.zip"
        print(
            "[INFO] Building local zip (STORE); then one upload to Drive…",
            flush=True,
        )
        _zip_directory_tree(
            downloaded_path,
            staging_zip,
            desc="Zip dataset (local, STORED)",
        )
        print(f"[INFO] Saving archive to Drive: {drive_archive_zip}", flush=True)
        ensure_dir(drive_archive_zip.parent)
        _copy_large_file(
            staging_zip,
            drive_archive_zip,
            desc="Copy zip → Drive",
        )
        print("[INFO] Extracting for install/move step…", flush=True)
        _unzip_to_dir(staging_zip, tmp_work, desc="Unzip dataset")
        staging_zip.unlink(missing_ok=True)
        return tmp_work

    downloaded_path = Path(kagglehub.dataset_download(slug))
    print(f"[INFO] KaggleHub downloaded/cached at: {downloaded_path}", flush=True)

    if not downloaded_path.is_dir():
        raise RuntimeError(
            f"[ERROR] Downloaded path is not a directory: {downloaded_path}"
        )

    print(
        "[INFO] Copying into temp dir (streaming; may take a while on slow disks)…",
        flush=True,
    )
    copytree_merge(
        downloaded_path,
        tmp_work,
        desc="Copy dataset → temp dir",
    )
    return tmp_work


def download_classifier_dataset(force: bool = False) -> None:
    ensure_dir(RAW_DIR)

    if COVID_READY_MARKER.exists() and not force:
        print("[SKIP] Classifier dataset already exists, no download needed.")
        return

    if force:
        print("[INFO] Force download requested for classifier dataset.")
        remove_if_exists(COVID_READY_MARKER)
        if IS_COLAB:
            remove_if_exists(CLASSIFIER_CACHE_DIR)
            remove_if_exists(CLASSIFIER_ARCHIVE_ZIP)

    tmp_dir = None
    try:
        print(f"[INFO] Downloading classifier dataset from Kaggle: {COVID_CRD_SLUG}")
        cache_dir = CLASSIFIER_CACHE_DIR if IS_COLAB else None
        zip_path = CLASSIFIER_ARCHIVE_ZIP if IS_COLAB else None
        tmp_dir = _download_to_temp(
            COVID_CRD_SLUG,
            cache_dir=cache_dir,
            drive_archive_zip=zip_path,
        )
        move_classifier_dataset(tmp_dir)
        print("[OK] Classifier dataset ready.")
    finally:
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def download_segmentation_dataset(force: bool = False) -> None:
    ensure_dir(SEGMENTATION_RAW_DIR)

    if SEG_READY_MARKER.exists() and not force:
        print("[SKIP] Segmentation dataset already exists, no download needed.")
        return

    if force:
        print("[INFO] Force download requested for segmentation dataset.")
        remove_if_exists(SEG_READY_MARKER)
        if IS_COLAB:
            remove_if_exists(SEGMENTATION_CACHE_DIR)
            remove_if_exists(SEGMENTATION_ARCHIVE_ZIP)

    tmp_dir = None
    try:
        print(f"[INFO] Downloading segmentation dataset from Kaggle: {CRD_SEG_SLUG}")
        cache_dir = SEGMENTATION_CACHE_DIR if IS_COLAB else None
        zip_path = SEGMENTATION_ARCHIVE_ZIP if IS_COLAB else None
        tmp_dir = _download_to_temp(
            CRD_SEG_SLUG,
            cache_dir=cache_dir,
            drive_archive_zip=zip_path,
        )
        move_segmentation_dataset(tmp_root=tmp_dir)
        print("[OK] Segmentation dataset ready.")
    finally:
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def download_all_datasets(force: bool = False) -> None:
    download_classifier_dataset(force=force)
    download_segmentation_dataset(force=force)


# =========================================================
# CLI futtatás
# =========================================================

if __name__ == "__main__":
    download_all_datasets()
