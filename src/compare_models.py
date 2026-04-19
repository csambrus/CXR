# compare_models.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.config import BATCH_SIZE, CLASS_INFOS, IMAGE_SIZE, NUM_CLASSES, SEED, ensure_dir, get_class_names, save_json
from src.evaluate import collect_predictions, compute_metrics
from src.dataloader import build_datasets_from_split_csvs

# =========================================================
# plot helpers
# =========================================================

def plot_metric_bars(
    comparison_df: pd.DataFrame,
    metric: str,
    save_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(comparison_df["model_name"], comparison_df[metric])
    ax.set_title(metric.replace("_", " ").title())
    ax.set_ylabel(metric)

    if metric == "loss":
        upper = max(1.0, float(comparison_df[metric].max()) * 1.15)
        ax.set_ylim(0, upper)
    else:
        ax.set_ylim(0, 1.0)

    for i, val in enumerate(comparison_df[metric]):
        ax.text(i, float(val), f"{float(val):.4f}", ha="center", va="bottom", fontsize=10)

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_metrics_row_comparison(
    comparison_df: pd.DataFrame,
    save_path: str | Path | None = None,
):
    metrics = ["loss", "accuracy", "recall_macro", "f1_macro"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    for ax, metric in zip(axes, metrics):
        ax.bar(comparison_df["model_name"], comparison_df[metric])
        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=20)

        if metric == "loss":
            upper = max(1.0, float(comparison_df[metric].max()) * 1.15)
            ax.set_ylim(0, upper)
        else:
            ax.set_ylim(0, 1.0)

        for i, val in enumerate(comparison_df[metric]):
            ax.text(i, float(val), f"{float(val):.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_model_roc_comparison(
    roc_payloads: list[dict[str, Any]],
    class_names: list[str],
    save_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    for payload in roc_payloads:
        model_name = payload["model_name"]
        y_true = payload["y_true"]
        y_prob = payload["y_prob"]

        y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))

        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())

        auc_micro = np.nan
        try:
            auc_micro = roc_auc_score(
                y_true_bin,
                y_prob,
                multi_class="ovr",
                average="micro",
            )
        except Exception:
            try:
                auc_micro = roc_auc_score(y_true_bin.ravel(), y_prob.ravel())
            except Exception:
                auc_micro = np.nan

        label = f"{model_name} (AUC={auc_micro:.3f})" if not np.isnan(auc_micro) else model_name
        ax.plot(fpr_micro, tpr_micro, label=label)

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("Model ROC Comparison (micro-average)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


# =========================================================
# model evaluation
# =========================================================

def evaluate_saved_model(
    model_name: str,
    model_path: str | Path,
    test_ds,
    num_classes: int,
):
    model = tf.keras.models.load_model(model_path, safe_mode = False)

    keras_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)
    loss = float(keras_metrics["loss"])

    y_true, y_pred, y_prob = collect_predictions(model, test_ds)
    metrics = compute_metrics(
        loss=loss,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        num_classes=num_classes,
    )

    result = {
        "model_name": model_name,
        **metrics,
    }

    roc_payload = {
        "model_name": model_name,
        "y_true": y_true,
        "y_prob": y_prob,
    }

    return result, roc_payload


# =========================================================
# main compare
# =========================================================

def run_model_comparison(
    model_names: list[str] | tuple[str, ...],
    models_dir: str | Path,
    split_dir: str | Path,
    out_dir: str | Path = "outputs/model_comparison",
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    model_filename: str = "best_model.keras",
):
    models_dir = Path(models_dir)
    out_dir = ensure_dir(out_dir)
    class_names = get_class_names()

    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"Models dir: {models_dir}")
    print(f"Split dir:  {split_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Models:     {list(model_names)}")
    print("=" * 80)

    # közös test dataset
    _, _, test_ds, _, _, test_df = build_datasets_from_split_csvs(
        split_dir=split_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )

    rows: list[dict[str, Any]] = []
    roc_payloads: list[dict[str, Any]] = []

    for model_name in model_names:
        model_path = models_dir / model_name / model_filename

        if not model_path.exists():
            print(f"[WARN] Missing model file, skipped: {model_path}")
            continue

        print(f"[INFO] Evaluating: {model_name}")
        result, roc_payload = evaluate_saved_model(
            model_name=model_name,
            model_path=model_path,
            test_ds=test_ds,
            num_classes=NUM_CLASSES,
        )

        rows.append(result)
        roc_payloads.append(roc_payload)

    if not rows:
        raise ValueError("Nem találtam egyetlen kiértékelhető modellt sem.")

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values("f1_macro", ascending=False).reset_index(drop=True)

    comparison_csv = out_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"[INFO] Saved: {comparison_csv}")

    summary = {
        "n_models": int(len(comparison_df)),
        "n_test_samples": int(len(test_df)),
        "class_names": class_names,
        "models": comparison_df.to_dict(orient="records"),
    }
    save_json(summary, out_dir / "model_comparison.json")

    # fő összehasonlító ábrák
    plot_metrics_row_comparison(
        comparison_df=comparison_df,
        save_path=out_dir / "metrics_row_comparison.png",
    )

    plot_model_roc_comparison(
        roc_payloads=roc_payloads,
        class_names=class_names,
        save_path=out_dir / "roc_model_comparison.png",
    )

    # külön bar chartok
    for metric in ["loss", "accuracy", "recall_macro", "f1_macro", "roc_auc_macro_ovr"]:
        plot_metric_bars(
            comparison_df=comparison_df,
            metric=metric,
            save_path=out_dir / f"{metric}_comparison.png",
        )

    print("\nComparison results:")
    print(comparison_df)

    return {
        "comparison_df": comparison_df,
        "out_dir": out_dir,
        "comparison_csv": comparison_csv,
        "metrics_row_png": out_dir / "metrics_row_comparison.png",
        "roc_comparison_png": out_dir / "roc_model_comparison.png",
    }


# =========================================================
# optional helper after training
# =========================================================

def compare_after_training(
    split_dir: str | Path,
    models_dir: str | Path,
    out_dir: str | Path,
    model_names: list[str] | tuple[str, ...] = ("resnet50", "vgg16", "efficientnetb0", "baseline_cnn"),
):
    return run_model_comparison(
        model_names=model_names,
        models_dir=models_dir,
        split_dir=split_dir,
        out_dir=out_dir,
    )


# =========================================================
# main
# =========================================================

if __name__ == "__main__":
    run_model_comparison(
        model_names=["resnet50", "vgg16", "efficientnetb0", "baseline_cnn"],
        models_dir=MODELS_DIR,
        split_dir=SPLITS_DIR,
        out_dir="outputs/model_comparison",
    )
    