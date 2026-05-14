from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Belső data_variant értékek (dataloader / train), megjelenítés: raw, crop, masked
VARIANT_ORDER: tuple[str, ...] = ("raw", "lung_crop", "lung_masked")

VARIANT_DISPLAY: dict[str, str] = {
    "raw": "raw",
    "lung_crop": "crop",
    "lung_masked": "masked",
}

# Belső model nevek (train.py), megjelenítés a felhasználó által kért formában
MODEL_ORDER: tuple[str, ...] = ("baseline_cnn", "efficientnetb0", "resnet50", "vgg16")

MODEL_DISPLAY: dict[str, str] = {
    "baseline_cnn": "baseline_cnn",
    "efficientnetb0": "efficientnet_b0",
    "resnet50": "resnet_50",
    "vgg16": "vgg_16",
}

# Összehasonlító metrikák oszlopnevei és tengelyfeliratok
COMPARISON_METRIC_COLUMNS: tuple[str, ...] = (
    "loss",
    "accuracy",
    "recall_macro",
    "roc_auc_macro_ovr",
)

METRIC_LABEL: dict[str, str] = {
    "loss": "loss",
    "accuracy": "Accuracy",
    "recall_macro": "Macro recall",
    "roc_auc_macro_ovr": "ROC-AUC",
}

# Tanítási history epoch-összehasonlításhoz (val_*)
VAL_METRIC_HISTORY_KEYS: tuple[tuple[str, ...], ...] = (
    ("val_loss", "loss"),
    ("val_accuracy", "val_sparse_categorical_accuracy", "accuracy"),
    ("val_recall_macro", "recall_macro"),
    ("val_roc_auc_macro_ovr", "roc_auc_macro_ovr"),
)


def sort_variants(variants: Iterable[str]) -> list[str]:
    uniq = list(dict.fromkeys(str(v) for v in variants))
    rank = {v: i for i, v in enumerate(VARIANT_ORDER)}

    def key(x: str) -> tuple[int, str]:
        return (rank.get(x, 999), x)

    return sorted(uniq, key=key)


def sort_models(models: Iterable[str]) -> list[str]:
    uniq = list(dict.fromkeys(str(m) for m in models))
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}

    def key(x: str) -> tuple[int, str]:
        return (rank.get(x, 999), x)

    return sorted(uniq, key=key)


def variant_display(v: str) -> str:
    return VARIANT_DISPLAY.get(str(v), str(v))


def model_display(m: str) -> str:
    return MODEL_DISPLAY.get(str(m), str(m))


def model_colors() -> dict[str, np.ndarray | tuple]:
    cmap = plt.get_cmap("tab10")
    models = list(MODEL_ORDER)
    return {m: cmap(i % 10) for i, m in enumerate(models)}


def variant_colors() -> dict[str, np.ndarray | tuple]:
    cmap = plt.get_cmap("Set2")
    vars_ = list(VARIANT_ORDER)
    return {v: cmap(i % 8) for i, v in enumerate(vars_)}


def reorder_comparison_df(df: pd.DataFrame) -> pd.DataFrame:
    """Modell és variáns szerinti kanonikus sorrend (elemzés / táblák / ábrák előtt)."""
    if len(df) == 0:
        return df
    out = df.copy()
    vm = sort_variants(out["data_variant"].dropna().unique())
    mm = sort_models(out["model"].dropna().unique())
    out["data_variant"] = pd.Categorical(out["data_variant"].astype(str), categories=vm, ordered=True)
    out["model"] = pd.Categorical(out["model"].astype(str), categories=mm, ordered=True)
    return out.sort_values(["data_variant", "model"]).reset_index(drop=True)


def leaderboard_display_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["model_variant", "model", "data_variant", *COMPARISON_METRIC_COLUMNS, "f1_macro"]
    return [c for c in preferred if c in df.columns]


def pick_val_history_column(hist: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in hist.columns:
            return name
    return None
