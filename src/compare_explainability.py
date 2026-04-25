# src/compare_explainability.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import (
    IMAGE_SIZE,
    MODELS_DIR,
    OUTPUT_DIR,
    SPLITS_DIR,
    PLOT_DPI,
    get_class_name,
    ensure_dir,
)
from src.dataloader import build_datasets_from_split_csvs
from src.explainability import (
    load_raw_image,
    load_processed_input,
    make_gradcam_heatmap,
    resize_heatmap_to_image,
    overlay_heatmap_on_image,
    find_last_conv_layer_name,
)


# =========================================================
# Helpers
# =========================================================

def _normalize_model_names(model_names: str | Iterable[str]) -> list[str]:
    if isinstance(model_names, str):
        return [model_names]
    return list(model_names)


def _normalize_variants(data_variants: str | Iterable[str]) -> list[str]:
    if isinstance(data_variants, str):
        return [data_variants]
    return list(data_variants)


def _safe_get_class_name(label: int) -> str:
    try:
        return get_class_name(int(label))
    except Exception:
        return str(label)


def _find_model_path(model_name: str, data_variant: str | None = None) -> Path:
    candidates: list[Path] = []

    if data_variant is not None:
        candidates.extend(
            [
                Path(MODELS_DIR) / data_variant / model_name / "best_model.keras",
                Path(MODELS_DIR) / data_variant / model_name / "final_model.keras",
                Path(MODELS_DIR) / model_name / data_variant / "best_model.keras",
                Path(MODELS_DIR) / model_name / data_variant / "final_model.keras",
                Path(MODELS_DIR) / f"{model_name}_{data_variant}" / "best_model.keras",
                Path(MODELS_DIR) / f"{model_name}_{data_variant}" / "final_model.keras",
            ]
        )

    candidates.extend(
        [
            Path(MODELS_DIR) / model_name / "best_model.keras",
            Path(MODELS_DIR) / model_name / "final_model.keras",
        ]
    )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Nem található modell. Próbált útvonalak:\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_model_by_name(model_name: str, data_variant: str | None = None):
    model_path = _find_model_path(model_name, data_variant=data_variant)

    model = tf.keras.models.load_model(model_path, safe_mode=False)
    last_conv = find_last_conv_layer_name(model)

    return model, last_conv, model_path


def _build_test_dataset(split_dir: str | Path, data_variant: str):
    try:
        return build_datasets_from_split_csvs(
            split_dir=split_dir,
            image_size=IMAGE_SIZE,
            batch_size=16,
            data_variant=data_variant,
        )
    except TypeError:
        return build_datasets_from_split_csvs(
            split_dir=split_dir,
            image_size=IMAGE_SIZE,
            batch_size=16,
        )


def _load_processed(filepath: str | Path, data_variant: str | None = None):
    try:
        return load_processed_input(filepath, data_variant=data_variant)
    except TypeError:
        return load_processed_input(filepath)


def _ensure_channel_last(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)

    if img.ndim == 2:
        img = img[..., np.newaxis]

    return img


def make_saliency_map(
    model: tf.keras.Model,
    input_tensor: tf.Tensor,
    pred_index: int | None = None,
) -> np.ndarray:
    input_tensor = tf.cast(input_tensor, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        preds = model(input_tensor, training=False)

        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))

        class_score = preds[:, pred_index]

    grads = tape.gradient(class_score, input_tensor)

    if grads is None:
        saliency = np.zeros(input_tensor.shape[1:3], dtype=np.float32)
    else:
        saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()

    saliency = saliency.astype(np.float32)
    saliency -= saliency.min()

    denom = saliency.max()
    if denom > 0:
        saliency /= denom

    return saliency


def _select_examples(
    test_ds,
    ref_model: tf.keras.Model,
    n_examples: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        probs = ref_model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    correct_idx = np.where(y_true == y_pred)[0]
    incorrect_idx = np.where(y_true != y_pred)[0]

    selected: list[int] = []

    half = max(1, n_examples // 2)

    if len(correct_idx) > 0:
        selected.extend(correct_idx[:half].tolist())

    if len(incorrect_idx) > 0:
        selected.extend(incorrect_idx[:half].tolist())

    if len(selected) < n_examples:
        all_idx = list(range(len(y_true)))
        for idx in all_idx:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= n_examples:
                break

    selected = selected[:n_examples]

    return y_true, y_pred, selected


def _imshow_gray(ax, img, title: str):
    img = _ensure_channel_last(img)
    ax.imshow(img[..., 0], cmap="gray")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# =========================================================
# Main: model comparison Grad-CAM + saliency
# =========================================================

def run_compare_explainability(
    model_names: str | Iterable[str] = ("resnet50", "vgg16", "efficientnetb0"),
    data_variants: str | Iterable[str] = ("raw",),
    split_dir=SPLITS_DIR,
    out_dir=None,
    n_examples: int = 6,
    include_saliency: bool = True,
    show: bool = True,
):
    model_names = _normalize_model_names(model_names)
    data_variants = _normalize_variants(data_variants)

    if out_dir is None:
        out_dir = Path(OUTPUT_DIR) / "figures" / "compare_explainability"

    out_dir = ensure_dir(out_dir)

    print("=" * 80)
    print("COMPARE EXPLAINABILITY")
    print("=" * 80)
    print("model_names  :", model_names)
    print("data_variants:", data_variants)
    print("out_dir      :", out_dir)
    print("=" * 80)

    for data_variant in data_variants:
        print("\n" + "=" * 80)
        print(f"DATA VARIANT: {data_variant}")
        print("=" * 80)

        variant_out_dir = ensure_dir(Path(out_dir) / data_variant)

        # -------------------------------------------------
        # dataset
        # -------------------------------------------------
        _, _, test_ds, _, _, test_df = _build_test_dataset(
            split_dir=split_dir,
            data_variant=data_variant,
        )

        # -------------------------------------------------
        # models
        # -------------------------------------------------
        models = {}
        for name in model_names:
            model, last_conv, model_path = load_model_by_name(
                name,
                data_variant=data_variant,
            )
            models[name] = {
                "model": model,
                "last_conv": last_conv,
                "model_path": model_path,
            }
            print(f"[INFO] Loaded {name}: {model_path}")
            print(f"[INFO] Last conv {name}: {last_conv}")

        ref_name = model_names[0]
        ref_model = models[ref_name]["model"]

        y_true, y_pred, selected = _select_examples(
            test_ds=test_ds,
            ref_model=ref_model,
            n_examples=n_examples,
        )

        print(f"[INFO] Selected examples: {len(selected)}")

        # -------------------------------------------------
        # visualization
        # -------------------------------------------------
        for i, idx in enumerate(selected):
            row = test_df.iloc[idx]
            filepath = row["filepath"]

            raw = load_raw_image(filepath)
            proc = _load_processed(filepath, data_variant=data_variant)

            raw = _ensure_channel_last(raw)
            proc = _ensure_channel_last(proc)

            input_tensor = tf.convert_to_tensor(proc[np.newaxis, ...], dtype=tf.float32)

            true_label = int(y_true[idx])

            # layout:
            # raw | processed | model1 Grad-CAM | model1 saliency | ...
            model_cols = 2 if include_saliency else 1
            ncols = 2 + len(models) * model_cols

            fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
            if ncols == 1:
                axes = [axes]

            _imshow_gray(axes[0], raw, "Raw")
            _imshow_gray(axes[1], proc, "Processed")

            col = 2

            for name, info in models.items():
                model = info["model"]
                last_conv = info["last_conv"]

                probs = model.predict(input_tensor, verbose=0)
                pred = int(np.argmax(probs))
                conf = float(np.max(probs))

                heatmap_small = make_gradcam_heatmap(
                    model,
                    input_tensor,
                    last_conv_layer_name=last_conv,
                    pred_index=pred,
                )

                heatmap = resize_heatmap_to_image(heatmap_small, IMAGE_SIZE)
                overlay = overlay_heatmap_on_image(proc, heatmap)

                axes[col].imshow(overlay)
                axes[col].set_title(
                    f"{name} Grad-CAM\nP: {_safe_get_class_name(pred)} ({conf:.2f})",
                    fontsize=9,
                )
                axes[col].axis("off")
                col += 1

                if include_saliency:
                    saliency = make_saliency_map(
                        model=model,
                        input_tensor=input_tensor,
                        pred_index=pred,
                    )

                    axes[col].imshow(saliency, cmap="gray")
                    axes[col].set_title(
                        f"{name} saliency\nP: {_safe_get_class_name(pred)} ({conf:.2f})",
                        fontsize=9,
                    )
                    axes[col].axis("off")
                    col += 1

            fig.suptitle(
                f"Variant: {data_variant} | True: {_safe_get_class_name(true_label)} | file: {Path(filepath).name}",
                fontsize=11,
            )

            plt.tight_layout()

            save_path = variant_out_dir / f"{i:02d}_{Path(filepath).stem}_gradcam_saliency.png"
            plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")

            if show:
                plt.show()
            else:
                plt.close(fig)

            print(f"[INFO] Saved: {save_path}")

    print("=" * 80)
    print("DONE")
    print("=" * 80)


# =========================================================
# Best model per variant helper
# =========================================================

def run_compare_explainability_from_comparison_df(
    comparison_df,
    split_dir=SPLITS_DIR,
    out_dir=None,
    n_examples: int = 6,
    include_saliency: bool = True,
    show: bool = True,
):
    if out_dir is None:
        out_dir = Path(OUTPUT_DIR) / "figures" / "compare_explainability_best"

    if "model" not in comparison_df.columns and "model_name" in comparison_df.columns:
        comparison_df = comparison_df.rename(columns={"model_name": "model"})

    if "data_variant" not in comparison_df.columns:
        raise ValueError("comparison_df must contain data_variant column.")

    if "f1_macro" in comparison_df.columns:
        sort_metric = "f1_macro"
    elif "accuracy" in comparison_df.columns:
        sort_metric = "accuracy"
    else:
        raise ValueError("comparison_df must contain f1_macro or accuracy column.")

    best_rows = []

    for variant, sub in comparison_df.groupby("data_variant"):
        best = sub.sort_values(sort_metric, ascending=False).iloc[0]
        best_rows.append(best)

    best_models = sorted(set(str(row["model"]) for row in best_rows))
    variants = sorted(set(str(row["data_variant"]) for row in best_rows))

    print("[INFO] Best models selected from comparison_df:")
    for row in best_rows:
        print(
            f"  variant={row['data_variant']} | model={row['model']} | "
            f"{sort_metric}={row[sort_metric]:.4f}"
        )

    run_compare_explainability(
        model_names=best_models,
        data_variants=variants,
        split_dir=split_dir,
        out_dir=out_dir,
        n_examples=n_examples,
        include_saliency=include_saliency,
        show=show,
    )