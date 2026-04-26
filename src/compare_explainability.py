# src/compare_explainability.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from src.config import (
    IMAGE_SIZE,
    MODELS_DIR,
    OUTPUT_DIR,
    SPLITS_DIR,
    RAW_DIR,
    PLOT_DPI,
    get_class_name,
    ensure_dir,
)

try:
    from src.config import LUNG_MASKED_DIR, LUNG_CROP_DIR
except Exception:
    LUNG_MASKED_DIR = Path(RAW_DIR).parent / "lung_masked"
    LUNG_CROP_DIR = Path(RAW_DIR).parent / "lung_crop"

from src.dataloader import build_datasets_from_split_csvs, read_split_csv
from src.explainability import resize_heatmap_to_image, overlay_heatmap_on_image


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


def _resolve_data_root(data_variant: str) -> Path:
    variant = str(data_variant).lower()

    if variant == "raw":
        return Path(RAW_DIR)
    if variant == "lung_masked":
        return Path(LUNG_MASKED_DIR)
    if variant == "lung_crop":
        return Path(LUNG_CROP_DIR)

    raise ValueError(
        f"Unknown data_variant: {data_variant!r}. "
        "Supported values: raw, lung_masked, lung_crop."
    )


def _find_model_path(model_name: str, data_variant: str | None = None) -> Path:
    candidates: list[Path] = []

    if data_variant is not None:
        candidates.extend(
            [
                Path(MODELS_DIR) / f"{model_name}_{data_variant}" / "best_model.keras",
                Path(MODELS_DIR) / f"{model_name}_{data_variant}" / "last_model.keras",
                Path(MODELS_DIR) / f"{model_name}_{data_variant}" / "final_model.keras",
                Path(MODELS_DIR) / data_variant / model_name / "best_model.keras",
                Path(MODELS_DIR) / data_variant / model_name / "last_model.keras",
                Path(MODELS_DIR) / model_name / data_variant / "best_model.keras",
                Path(MODELS_DIR) / model_name / data_variant / "last_model.keras",
            ]
        )

    candidates.extend(
        [
            Path(MODELS_DIR) / model_name / "best_model.keras",
            Path(MODELS_DIR) / model_name / "last_model.keras",
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


def _build_test_dataset(split_dir: str | Path, data_variant: str):
    """
    Csak az új dataloader API-t kezeli:
        build_datasets_from_split_csvs(...) -> train_ds, val_ds, test_ds
    A test_df-et külön olvassuk a test.csv-ből.
    """
    split_dir = Path(split_dir)
    data_root = _resolve_data_root(data_variant)

    _, _, test_ds = build_datasets_from_split_csvs(
        split_dir=split_dir,
        data_root=data_root,
        image_size=IMAGE_SIZE,
        batch_size=16,
        channels=1,
        cache=False,
    )

    test_df = read_split_csv(split_dir / "test.csv")
    return test_ds, test_df


def _ensure_channel_last(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    return img


def _load_gray_image(
    path: str | Path,
    image_size: tuple[int, int] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing image file: {path}")

    with Image.open(path) as img:
        img = img.convert("L")
        if image_size is not None:
            img = img.resize((int(image_size[1]), int(image_size[0])))
        arr = np.asarray(img)

    arr = arr.astype(np.float32)
    if normalize:
        arr = arr / 255.0
    return arr[..., np.newaxis]


def _imshow_gray(ax, img, title: str):
    img = _ensure_channel_last(img)
    ax.imshow(img[..., 0], cmap="gray")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _display_saved_image(path: str | Path) -> None:
    try:
        from IPython.display import Image as IPyImage, display
        display(IPyImage(filename=str(path)))
    except Exception:
        pass


def _path_from_split_row(row: pd.Series, root_dir: str | Path) -> Path:
    if "relative_path" not in row:
        raise ValueError("A test.csv-ben nincs relative_path oszlop.")
    return Path(root_dir) / str(row["relative_path"])


def _output_stem_from_row(row: pd.Series, idx: int) -> str:
    if "relative_path" in row:
        rel = str(row["relative_path"])
        safe = rel.replace("/", "__").replace("\\", "__")
        return Path(safe).stem
    if "filename" in row:
        return Path(str(row["filename"])).stem
    return f"example_{idx:04d}"


# =========================================================
# Robust Grad-CAM for nested transfer models
# =========================================================

def _layer_output_rank(layer: tf.keras.layers.Layer) -> int | None:
    try:
        shape = layer.output.shape
        return len(shape) if shape is not None else None
    except Exception:
        return None


def _find_last_4d_layer_name(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if _layer_output_rank(layer) == 4:
            return layer.name
    raise ValueError(f"Nem találtam 4D Grad-CAM célréteget ebben a modellben: {model.name}")


def find_gradcam_target(model: tf.keras.Model) -> dict[str, str]:
    """
    Transfer learning modelleknél a backbone gyakran beágyazott Functional modell
    az outer modelben. Ilyenkor a cél nem maga a 'resnet50'/'vgg16' layer, hanem
    annak belső utolsó 4D feature map rétege.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            try:
                if _layer_output_rank(layer) == 4:
                    return {
                        "mode": "nested",
                        "base_layer_name": layer.name,
                        "inner_layer_name": _find_last_4d_layer_name(layer),
                    }
            except Exception:
                pass

        if _layer_output_rank(layer) == 4:
            return {"mode": "outer", "layer_name": layer.name}

    raise ValueError("Nem találtam Grad-CAM-kompatibilis 4D réteget a modellben.")


def _format_gradcam_target(target: dict[str, str]) -> str:
    if target.get("mode") == "nested":
        return f"{target['base_layer_name']} / {target['inner_layer_name']}"
    return str(target.get("layer_name"))


def _call_layer_safely(layer: tf.keras.layers.Layer, x: tf.Tensor) -> tf.Tensor:
    if isinstance(layer, tf.keras.layers.InputLayer):
        return x
    if isinstance(layer, tf.keras.layers.Concatenate):
        return layer([x, x, x])
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def make_gradcam_heatmap_robust(
    model: tf.keras.Model,
    image_tensor: tf.Tensor,
    gradcam_target: dict[str, str],
    pred_index: int | None = None,
) -> np.ndarray:
    image_tensor = tf.cast(image_tensor, tf.float32)

    if gradcam_target.get("mode") == "outer":
        target_layer = model.get_layer(gradcam_target["layer_name"])
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(image_tensor, training=False)
            if pred_index is None:
                pred_index = int(tf.argmax(preds[0]))
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

    elif gradcam_target.get("mode") == "nested":
        base_layer_name = gradcam_target["base_layer_name"]
        inner_layer_name = gradcam_target["inner_layer_name"]
        base_model = model.get_layer(base_layer_name)
        inner_layer = base_model.get_layer(inner_layer_name)

        base_grad_model = tf.keras.models.Model(
            inputs=base_model.inputs,
            outputs=[inner_layer.output, base_model.output],
        )

        with tf.GradientTape() as tape:
            x = image_tensor
            conv_outputs = None
            passed_base = False

            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue

                if layer.name == base_layer_name:
                    conv_outputs, x = base_grad_model(x, training=False)
                    passed_base = True
                else:
                    x = _call_layer_safely(layer, x)

            if not passed_base or conv_outputs is None:
                raise RuntimeError(f"Nem sikerült elérni a beágyazott backbone-t: {base_layer_name}")

            preds = x
            if pred_index is None:
                pred_index = int(tf.argmax(preds[0]))
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

    else:
        raise ValueError(f"Ismeretlen Grad-CAM target mód: {gradcam_target}")

    if grads is None:
        return np.zeros(tuple(image_tensor.shape[1:3]), dtype=np.float32)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if float(max_val) > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy().astype(np.float32)


def load_model_by_name(model_name: str, data_variant: str | None = None):
    model_path = _find_model_path(model_name, data_variant=data_variant)
    model = tf.keras.models.load_model(model_path, safe_mode=False)
    gradcam_target = find_gradcam_target(model)
    return model, gradcam_target, model_path


# =========================================================
# Saliency
# =========================================================

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


# =========================================================
# Example selection
# =========================================================

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
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())

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
        for idx in range(len(y_true)):
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= n_examples:
                break

    return selected[:n_examples]


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
    skip_existing: bool = False,
) -> dict[str, Any]:
    model_names = _normalize_model_names(model_names)
    data_variants = _normalize_variants(data_variants)
    split_dir = Path(split_dir)

    if out_dir is None:
        out_dir = Path(OUTPUT_DIR) / "figures" / "compare_explainability"
    out_dir = ensure_dir(out_dir)

    print("=" * 80)
    print("COMPARE EXPLAINABILITY")
    print("=" * 80)
    print("model_names  :", model_names)
    print("data_variants:", data_variants)
    print("split_dir    :", split_dir)
    print("out_dir      :", out_dir)
    print("show         :", show)
    print("skip_existing:", skip_existing)
    print("=" * 80)

    summary: dict[str, Any] = {
        "out_dir": str(out_dir),
        "split_dir": str(split_dir),
        "model_names": model_names,
        "data_variants": data_variants,
        "items": [],
    }

    for data_variant in data_variants:
        print("\n" + "=" * 80)
        print(f"DATA VARIANT: {data_variant}")
        print("=" * 80)

        variant_out_dir = ensure_dir(Path(out_dir) / data_variant)
        data_root = _resolve_data_root(data_variant)

        test_ds, test_df = _build_test_dataset(
            split_dir=split_dir,
            data_variant=data_variant,
        )

        models: dict[str, dict[str, Any]] = {}
        for name in model_names:
            model, gradcam_target, model_path = load_model_by_name(name, data_variant=data_variant)
            models[name] = {
                "model": model,
                "gradcam_target": gradcam_target,
                "model_path": model_path,
            }
            print(f"[INFO] Loaded {name}: {model_path}")
            print(f"[INFO] Grad-CAM target {name}: {_format_gradcam_target(gradcam_target)}")

        ref_name = model_names[0]
        y_true, y_pred, selected = _select_examples(
            test_ds=test_ds,
            ref_model=models[ref_name]["model"],
            n_examples=n_examples,
        )
        print(f"[INFO] Selected examples: {len(selected)}")

        for i, idx in enumerate(selected):
            row = test_df.iloc[idx]
            raw_path = _path_from_split_row(row, RAW_DIR)
            processed_path = _path_from_split_row(row, data_root)

            stem = _output_stem_from_row(row, idx)
            suffix = "gradcam_saliency" if include_saliency else "gradcam"
            save_path = variant_out_dir / f"{i:02d}_{stem}_{suffix}.png"

            if skip_existing and save_path.exists():
                print(f"[SKIP] Existing: {save_path}")
                if show:
                    _display_saved_image(save_path)
                summary["items"].append({"data_variant": data_variant, "index": int(idx), "path": str(save_path), "skipped": True})
                continue

            raw = _load_gray_image(raw_path, image_size=IMAGE_SIZE, normalize=True)
            proc = _load_gray_image(processed_path, image_size=IMAGE_SIZE, normalize=True)
            input_tensor = tf.convert_to_tensor(proc[np.newaxis, ...], dtype=tf.float32)
            true_label = int(y_true[idx])

            model_cols = 2 if include_saliency else 1
            ncols = 2 + len(models) * model_cols
            fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
            axes = np.ravel(axes)

            _imshow_gray(axes[0], raw, "Raw")
            _imshow_gray(axes[1], proc, f"Processed\n{data_variant}")

            col = 2
            prediction_rows: list[dict[str, Any]] = []

            for name, info in models.items():
                model = info["model"]
                gradcam_target = info["gradcam_target"]

                probs = model.predict(input_tensor, verbose=0)
                pred = int(np.argmax(probs))
                conf = float(np.max(probs))

                prediction_rows.append(
                    {
                        "model": name,
                        "pred": pred,
                        "pred_name": _safe_get_class_name(pred),
                        "confidence": conf,
                    }
                )

                heatmap_small = make_gradcam_heatmap_robust(
                    model=model,
                    image_tensor=input_tensor,
                    gradcam_target=gradcam_target,
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
                f"Variant: {data_variant} | True: {_safe_get_class_name(true_label)} | file: {Path(processed_path).name}",
                fontsize=11,
            )
            fig.tight_layout()
            fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")

            if show:
                plt.show()
            else:
                plt.close(fig)

            print(f"[INFO] Saved: {save_path}")
            summary["items"].append(
                {
                    "data_variant": data_variant,
                    "index": int(idx),
                    "raw_path": str(raw_path),
                    "processed_path": str(processed_path),
                    "path": str(save_path),
                    "true_label": true_label,
                    "true_name": _safe_get_class_name(true_label),
                    "predictions": prediction_rows,
                    "skipped": False,
                }
            )

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    return summary


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
    skip_existing: bool = False,
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

    return run_compare_explainability(
        model_names=best_models,
        data_variants=variants,
        split_dir=split_dir,
        out_dir=out_dir,
        n_examples=n_examples,
        include_saliency=include_saliency,
        show=show,
        skip_existing=skip_existing,
    )
