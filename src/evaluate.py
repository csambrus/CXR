# evaluate.py

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
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.config import BATCH_SIZE, CLASS_INFOS, IMAGE_SIZE, NUM_CLASSES, SEED, SPLITS_DIR, MODELS_DIR, OUTPUT_DIR, ensure_dir, get_class_names, save_json
from src.dataloader import build_datasets_from_split_csvs


# =========================================================
# util
# =========================================================

def load_gray_image(path: str | Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    img = tf.io.read_file(str(path))
    img = tf.io.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)

    if target_size is not None:
        img = tf.image.resize(img, target_size, method="bilinear", antialias=True)

    return img.numpy()


# =========================================================
# prediction
# =========================================================

def collect_predictions(model, dataset):
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []

    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


# =========================================================
# metrics
# =========================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    loss: float | None = None,
    y_prob: np.ndarray | None = None,
    num_classes: int | None = None,
) -> dict[str, float]:
    metrics = {}

    # --- alap metrikák ---
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["recall_macro"] = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["f1_macro"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    # --- opcionális loss ---
    if loss is not None:
        metrics["loss"] = float(loss)

    # --- opcionális ROC AUC ---
    if y_prob is not None and num_classes is not None:
        try:
            y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
            auc = roc_auc_score(
                y_true_bin,
                y_prob,
                multi_class="ovr",
                average="macro",
            )
            metrics["roc_auc_macro_ovr"] = float(auc)
        except Exception:
            metrics["roc_auc_macro_ovr"] = np.nan

    return metrics

# =========================================================
# plots
# =========================================================

def plot_metrics_row(
    loss: float,
    accuracy: float,
    recall: float,
    f1: float,
    save_path: str | Path | None = None,
):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    titles = ["Loss", "Accuracy", "Recall (macro)", "F1 (macro)"]
    values = [loss, accuracy, recall, f1]

    for ax, title, value in zip(axes, titles, values):
        ax.bar([0], [value])
        ax.set_title(title)
        ax.set_xticks([])

        if title == "Loss":
            upper = max(1.0, value * 1.2)
            ax.set_ylim(0, upper)
        else:
            ax.set_ylim(0, 1.0)

        ax.text(
            0,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: list[str],
    normalize: bool = False,
    save_path: str | Path | None = None,
):
    cm_plot = cm.astype(np.float64)

    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm_plot, row_sums, out=np.zeros_like(cm_plot), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_plot)

    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            text_val = f"{cm_plot[i, j]:.2f}" if normalize else f"{int(cm_plot[i, j])}"
            ax.text(j, i, text_val, ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    save_path: str | Path | None = None,
):
    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, class_name in enumerate(class_names):
        if len(np.unique(y_true_bin[:, i])) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc_i = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={auc_i:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("Multi-class ROC (one-vs-rest)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


# =========================================================
# reports
# =========================================================

def save_classification_report_files(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_dir: str | Path,
):
    out_dir = Path(out_dir)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    save_json(report_dict, out_dir / "classification_report.json")

    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(out_dir / "classification_report.csv", index=True)

    return report_dict, report_text, report_df


# =========================================================
# misclassified
# =========================================================

def save_misclassified_examples(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_dir: str | Path,
    max_examples: int = 12,
    image_size: tuple[int, int] = IMAGE_SIZE,
):
    out_dir = Path(out_dir)
    mis_dir = ensure_dir(out_dir / "misclassified")

    wrong_idx = np.where(y_true != y_pred)[0]

    if len(wrong_idx) == 0:
        print("[INFO] No misclassified samples found.")
        return pd.DataFrame(columns=["filepath", "true_label", "pred_label", "true_name", "pred_name"])

    subset_idx = wrong_idx[:max_examples]

    rows = []
    n = len(subset_idx)
    cols = min(4, n)
    rows_n = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 4 * rows_n), squeeze=False)

    for plot_i, idx in enumerate(subset_idx):
        r = plot_i // cols
        c = plot_i % cols

        row = test_df.iloc[idx]
        filepath = row["filepath"]

        img = load_gray_image(filepath, target_size=image_size)

        true_label = int(y_true[idx])
        pred_label = int(y_pred[idx])

        axes[r, c].imshow(img[..., 0], cmap="gray")
        axes[r, c].set_title(
            f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}",
            fontsize=10,
        )
        axes[r, c].axis("off")

        rows.append(
            {
                "filepath": filepath,
                "true_label": true_label,
                "pred_label": pred_label,
                "true_name": class_names[true_label],
                "pred_name": class_names[pred_label],
            }
        )

    total_axes = rows_n * cols
    for extra_i in range(n, total_axes):
        r = extra_i // cols
        c = extra_i % cols
        axes[r, c].axis("off")

    plt.tight_layout()
    fig_path = mis_dir / "misclassified_examples.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()

    mis_df = pd.DataFrame(rows)
    mis_df.to_csv(mis_dir / "misclassified_examples.csv", index=False)

    print(f"[INFO] Saved misclassified figure: {fig_path}")
    print(f"[INFO] Saved misclassified CSV: {mis_dir / 'misclassified_examples.csv'}")

    return mis_df


# =========================================================
# main
# =========================================================

def run_evaluation(
    model_path: str | Path = MODELS_DIR,
    split_dir: str | Path = SPLITS_DIR,
    out_dir: str | Path = OUTPUT_DIR / "evaluation",
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    max_misclassified_examples: int = 12,
):
    model_path = Path(model_path)
    out_dir = ensure_dir(out_dir)
    class_names = get_class_names()

    print("=" * 80)
    print("EVALUATION")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Split dir: {split_dir}")
    print(f"Output dir: {out_dir}")
    print("=" * 80)

    # -----------------------------------------------------
    # load model
    # -----------------------------------------------------
    model = tf.keras.models.load_model(model_path, safe_mode = False)

    # -----------------------------------------------------
    # dataset
    # -----------------------------------------------------
    _, _, test_ds, _, _, test_df = build_datasets_from_split_csvs(
        split_dir=split_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )

    # -----------------------------------------------------
    # keras evaluate
    # -----------------------------------------------------
    keras_metrics = model.evaluate(test_ds, verbose=1, return_dict=True)

    loss = float(keras_metrics["loss"])
    accuracy = float(keras_metrics.get("accuracy", 0.0))

    # -----------------------------------------------------
    # predictions
    # -----------------------------------------------------
    y_true, y_pred, y_prob = collect_predictions(model, test_ds)

    extra_metrics = compute_metrics(y_true, y_pred)
    recall = extra_metrics["recall_macro"]
    f1 = extra_metrics["f1_macro"]

    # -----------------------------------------------------
    # save summary
    # -----------------------------------------------------
    results = {
        "loss": loss,
        "accuracy": accuracy,
        "recall_macro": recall,
        "f1_macro": f1,
        "n_samples": int(len(test_df)),
        "num_classes": int(NUM_CLASSES),
        "class_names": class_names,
    }

    print("\nResults:")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    save_json(results, out_dir / "evaluation_metrics.json")

    # -----------------------------------------------------
    # plot metrics row
    # -----------------------------------------------------
    plot_metrics_row(
        loss=loss,
        accuracy=accuracy,
        recall=recall,
        f1=f1,
        save_path=out_dir / "metrics_row.png",
    )

    # -----------------------------------------------------
    # confusion matrix
    # -----------------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    np.save(out_dir / "confusion_matrix.npy", cm)

    plot_confusion_matrix_heatmap(
        cm=cm,
        class_names=class_names,
        normalize=False,
        save_path=out_dir / "confusion_matrix.png",
    )

    plot_confusion_matrix_heatmap(
        cm=cm,
        class_names=class_names,
        normalize=True,
        save_path=out_dir / "confusion_matrix_normalized.png",
    )

    # -----------------------------------------------------
    # classification report
    # -----------------------------------------------------
    report_dict, report_text, report_df = save_classification_report_files(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        out_dir=out_dir,
    )

    print("\nClassification report:")
    print(report_text)

    # -----------------------------------------------------
    # ROC
    # -----------------------------------------------------
    try:
        auc_macro_ovr = roc_auc_score(
            label_binarize(y_true, classes=np.arange(len(class_names))),
            y_prob,
            multi_class="ovr",
            average="macro",
        )
        save_json({"roc_auc_macro_ovr": float(auc_macro_ovr)}, out_dir / "roc_auc.json")
        print(f"[INFO] ROC AUC macro OVR: {auc_macro_ovr:.4f}")
    except Exception as e:
        print(f"[WARN] ROC AUC számítás nem sikerült: {e}")

    plot_multiclass_roc(
        y_true=y_true,
        y_prob=y_prob,
        class_names=class_names,
        save_path=out_dir / "roc_multiclass.png",
    )

    # -----------------------------------------------------
    # Misclassified examples
    # -----------------------------------------------------
    mis_df = save_misclassified_examples(
        test_df=test_df,
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        out_dir=out_dir,
        max_examples=max_misclassified_examples,
        image_size=image_size,
    )

    return {
        "results": results,
        "report_df": report_df,
        "misclassified_df": mis_df,
        "out_dir": out_dir,
        "metrics_json": out_dir / "evaluation_metrics.json",
        "metrics_row_png": out_dir / "metrics_row.png",
        "confusion_matrix_png": out_dir / "confusion_matrix.png",
        "confusion_matrix_norm_png": out_dir / "confusion_matrix_normalized.png",
        "roc_png": out_dir / "roc_multiclass.png",
        "classification_report_csv": out_dir / "classification_report.csv",
        "misclassified_csv": out_dir / "misclassified" / "misclassified_examples.csv",
    }


# =========================================================
# main
# =========================================================

if __name__ == "__main__":
    run_evaluation(
        model_path=MODELS_DIR / "resnet50/best_model.keras",
        split_dir=SPLIT_DIR,
        out_dir=OUTPUT_DIR / "evaluation/resnet50",
    )
