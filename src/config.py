# src/config.py
from pathlib import Path

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
SPLITS_DIR = DATA_DIR / "splits"

OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOGS_DIR = OUTPUT_DIR / "logs"

SRC_DIR = PROJECT_DIR / "src"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"

# =========================================================
# Dataset / osztályok
# =========================================================
CLASS_NAMES = [
    "normal",
    "pneumonia_viral",
    "pneumonia_bacterial",
    "covid19",
]

NUM_CLASSES = len(CLASS_NAMES)

# Ha a raw datasetben más néven vannak a mappák, később itt lehet
# átnevezési térképet kezelni.
CLASS_NAME_MAP = {
    "normal": "normal",
    "pneumonia_viral": "pneumonia_viral",
    "pneumonia_bacterial": "pneumonia_bacterial",
    "covid19": "covid19",
}

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

RANDOM_STATE = 42
SHUFFLE_DATA = True

# =========================================================
# DataLoader / tf.data
# =========================================================
BATCH_SIZE = 32
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
# Fájlnevek / split CSV-k
# =========================================================
TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "val.csv"
TEST_CSV = SPLITS_DIR / "test.csv"

# =========================================================
# Utility
# =========================================================
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def ensure_project_dirs() -> None:
    """Létrehozza a projekt fontos könyvtárait, ha még nem léteznek."""
    dirs = [
        DATA_DIR,
        RAW_DIR,
        INTERIM_DIR,
        SPLITS_DIR,
        OUTPUT_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
