# src/config.py
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import json

# =========================================================
# Projekt gyökér
# =========================================================
PROJECT_DIR = Path(__file__).resolve().parents[1]

# =========================================================
# Könyvtárak
# =========================================================
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
SPLITS_DIR = INTERIM_DIR / "splits"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODELS_DIR = INTERIM_DIR / "models"

FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOGS_DIR = OUTPUT_DIR / "logs"

# =========================================================
# Dataset / osztályok
# =========================================================

@dataclass(frozen=True)
class ClassInfo:
    key: str
    raw_dir: str
    display_name: str
    idx: int

CLASS_INFOS = [
    ClassInfo("normal", "Normal", "Normal", 0),
    ClassInfo("pneumonia_viral", "Pneumonia-Viral", "Pneumonia-Viral", 1),
    ClassInfo("pneumonia_bacterial", "Pneumonia-Bacterial", "Pneumonia-Bacterial", 2),
    ClassInfo("covid19", "COVID-19", "COVID-19", 3),
]

NUM_CLASSES = len(CLASS_INFOS)
CLASS_BY_KEY = {c.key: c for c in CLASS_INFOS}
CLASS_BY_IDX = {c.idx: c for c in CLASS_INFOS}

# =========================================================
# Split CSV-k
# =========================================================
TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "val.csv"
TEST_CSV = SPLITS_DIR / "test.csv"

# =========================================================
# Input image paraméterek
# =========================================================
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# =========================================================
# Split arányok
# =========================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
SHUFFLE_DATA = True

# =========================================================
# DataLoader / tf.data
# =========================================================
BATCH_SIZE = 256
AUTOTUNE = -1  # tf.data.AUTOTUNE majd runtime-ban lesz használva
CACHE_DATASET = False
PREFETCH_DATASET = True

# =========================================================
# Preprocessing / augmentáció (első verzió)
# =========================================================
USE_MODEL_PREPROCESSING = True
NORMALIZE_TO_0_1 = True

# Később bővíthető:
USE_CLAHE = False
USE_GAUSSIAN_BLUR = False
USE_HIST_EQ = False

# =========================================================
# Training baseline
# =========================================================
EPOCHS = 15
LEARNING_RATE = 1e-4

# =========================================================
# Modellek
# =========================================================
BACKBONE_NAME = "DenseNet121"
USE_IMAGENET_WEIGHTS = True
FREEZE_BACKBONE = True


# =========================================================
# Utilities
# =========================================================
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

# Main project dirs:
for d in [DATA_DIR, RAW_DIR, INTERIM_DIR, OUTPUT_DIR, SPLITS_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    ensure_dir(d)

def get_class_names() -> list[str]:
    return [c.display_name for c in CLASS_INFOS]

def get_class_name(idx: int) -> str:
    class_names = get_class_names()
    if 0 <= idx < len(class_names):
        return class_names[idx]
    return f"class_{idx}"

def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
