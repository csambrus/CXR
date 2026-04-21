# src/evaluate.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    NUM_CLASSES,
    SEED,
    ensure_dir,
    get_class_names,
    save_json,
)
from src.dataloader import build_datasets_from_split_csvs


# =========================================================
# prediction helpers
# =========================================================

def collect_predictions(
    model: tf.keras.Model,
    dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Összegyűjti a ground truth címkéket, a prediktált osztályokat
    és a predikciós valószínűségeket.
    """
    y_prob = model.predict(dataset, verbose=0)

    if y_prob.ndim == 1:
        y_prob = np.expand_dims(y_prob, axis=-1)

    y_pred = np.argmax(y_prob, axis=1)

    y_true_batches: list[np.ndarray] = []
    for _, labels in dataset:
        labels_np = labels.numpy()
        if labels_np.ndim > 1:
            labels_np = np.argmax(labels_np, axis=1)
        y_true_batches.append(labels_np)

    y_true = np.concatenate(y_true_batches, axis=0)

    return y_true.astype(int), y_pred.astype(int), y_prob.astype(float)


def compute_metrics(
    loss: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    """
    Fő metrikák számítása.
    """
    accuracy = accuracy_score(y_true, y_pred)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    roc_auc_macro_ovr = np.nan
    try:
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        roc_auc_macro_ovr = roc_auc_score(
            y_true_bin,
            y_prob,
            multi_class="ovr",
            average="macro",
        )
    except Exception:
        pass

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "roc_auc_macro_ovr": float(roc_auc_macro_ovr) if not np.isnan(roc_auc_macro_ovr) else np.nan,
    }


# =========================================================
# plots
# =========================================================

def plot_training_history(
    history: dict[str, list[float]] | Any,
    model_name: str,
    save_path: str | Path | None = None,
):
    """
    Training history ábra: loss, accuracy, recall, f1 ha elérhető.
    """
    if hasattr(history, "history"):
        history = history.history

    if not isinstance(history, dict) or not history:
        print("[WARN] Nincs használható training history.")
        return

    metric_pairs: list[tuple[str, str]] = []

    if "loss" in history:
        metric_pairs.append(("loss", "val_loss"))

    if "accuracy" in history:
        metric_pairs.append(("accuracy", "val_accuracy"))

    if "recall" in history:
        metric_pairs.append(("recall", "val_recall"))
    elif "recall_macro" in history:
        metric_pairs.append(("recall_macro", "val_recall_macro"))

    if "f1_macro" in history:
        metric_pairs.append(("f1_macro", "val_f1_macro"))

    if not metric_pairs:
        print("[WARN] Nem találtam megjeleníthető metrikát a history-ban.")
        return

    fig, axes = plt.subplots(1, len(metric_pairs), figsize=(5 * len(metric_pairs), 4))
    if len(metric_pairs) == 1:
        axes = [axes]

    for ax, (train_key, val_key) in zip(axes, metric_pairs):
        ax.plot(history.get(train_key, []), label=train_key)
        if val_key in history:
            ax.plot(history.get(val_key, []), label=val_key)
        ax.set_title(train_key.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.legend()

    fig.suptitle(f"TRAINING HISTORY - {model_name}", fontsize=16)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_evaluation_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    model_name: str,
    save_path: str | Path | None = None,
):
    """
    Egy sorban:
    1) confusion matrix
    2) normalized confusion matrix
    3) ROC curve
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # -----------------------------------------------------
    # 1) Confusion matrix
    # -----------------------------------------------------
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # -----------------------------------------------------
    # 2) Normalized confusion matrix
    # -----------------------------------------------------
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title("Normalized Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    # -----------------------------------------------------
    # 3) ROC
    # -----------------------------------------------------
    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))

    for i, class_name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            auc_i = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
            axes[2].plot(fpr, tpr, label=f"{class_name} (AUC={auc_i:.3f})")
        except Exception:
            continue

    try:
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

        if np.isnan(auc_micro):
            axes[2].plot(
                fpr_micro,
                tpr_micro,
                linestyle="--",
                linewidth=2,
                label="micro-average",
            )
        else:
            axes[2].plot(
                fpr_micro,
                tpr_micro,
                linestyle="--",
                linewidth=2,
                label=f"micro-average (AUC={auc_micro:.3f})",
            )
    except Exception:
        pass

    axes[2].plot([0, 1], [0, 1], linestyle="--")
    axes[2].set_title("ROC Curve")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].legend(loc="lower right", fontsize=9)

    fig.suptitle(f"EVALUATION - {model_name}", fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


# =========================================================
# reports
# =========================================================

def build_classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


def print_classification_report_table(
    report_dict: dict[str, Any],
):
    df = pd.DataFrame(report_dict).transpose()
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df)


# =========================================================
# main evaluation
# =========================================================

def run_evaluation(
    model_path: str | Path,
    split_dir: str | Path,
    out_dir: str | Path,
    model_name: str,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
):
    model_path = Path(model_path)
    out_dir = ensure_dir(out_dir)
    class_names = get_class_names()

    print("=" * 80)
    print(f"EVALUATION - {model_name}")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Split dir:  {split_dir}")
    print(f"Output dir: {out_dir}")
    print("=" * 80)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file nem található: {model_path}")

    # Datasets
    _, _, test_ds, _, _, test_df = build_datasets_from_split_csvs(
        split_dir=split_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )

    # Load model
    model = tf.keras.models.load_model(model_path, safe_mode=False)

    # Keras evaluate
    keras_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)
    loss = float(keras_metrics["loss"])

    # Predictions
    y_true, y_pred, y_prob = collect_predictions(model, test_ds)

    # Scalar metrics
    metrics = compute_metrics(
        loss=loss,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        num_classes=NUM_CLASSES,
    )

    # Classification report
    report_dict = build_classification_report_dict(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )

    # Save metrics JSON
    summary = {
        "model_name": model_name,
        "model_path": str(model_path),
        "n_test_samples": int(len(test_df)),
        "class_names": class_names,
        "metrics": metrics,
        "classification_report": report_dict,
    }
    save_json(summary, out_dir / "evaluation_summary.json")

    # Save per-sample predictions CSV
    pred_df = test_df.copy().reset_index(drop=True)
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred

    for i, class_name in enumerate(class_names):
        pred_df[f"prob_{class_name}"] = y_prob[:, i]

    pred_df["correct"] = (pred_df["y_true"] == pred_df["y_pred"]).astype(int)
    pred_csv = out_dir / "test_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    # Print summary
    print("\nMetrics:")
    for k, v in metrics.items():
        if isinstance(v, float) and not np.isnan(v):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\nClassification report:")
    print_classification_report_table(report_dict)

    # Main evaluation figure: CM + normalized CM + ROC
    plot_evaluation_row(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names,
        model_name=model_name,
        save_path=out_dir / f"{model_name}_evaluation_row.png",
    )

    return {
        "model_name": model_name,
        "metrics": metrics,
        "report_dict": report_dict,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "predictions_csv": pred_csv,
        "summary_json": out_dir / "evaluation_summary.json",
        "evaluation_png": out_dir / f"{model_name}_evaluation_row.png",
        "out_dir": out_dir,
    }


# =========================================================
# script entry
# =========================================================

if __name__ == "__main__":
    from src.config import MODELS_DIR, SPLITS_DIR

    model_name = "resnet50"
    model_path = MODELS_DIR / model_name / "best_model.keras"

    run_evaluation(
        model_path=model_path,
        split_dir=SPLITS_DIR,
        out_dir=Path("outputs") / "evaluation" / model_name,
        model_name=model_name,
    )
