# src/compare_explainability.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import (
    CLASS_INFOS,
    IMAGE_SIZE,
    MODELS_DIR,
    OUTPUT_DIR,
    SPLITS_DIR,
    get_class_name,
    get_class_names,
    ensure_dir
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
# load model
# =========================================================

def load_model_by_name(model_name: str):
    model_path = Path(MODELS_DIR) / model_name / "best_model.keras"
    if not model_path.exists():
        model_path = Path(MODELS_DIR) / model_name / "final_model.keras"

    if not model_path.exists():
        raise FileNotFoundError(f"Nincs modell: {model_name}")

    model = tf.keras.models.load_model(model_path, safe_mode = False)
    last_conv = find_last_conv_layer_name(model)

    return model, last_conv


# =========================================================
# main
# =========================================================

def run_compare_explainability(
    model_names=("resnet50", "vgg16", "efficientnetb0"),
    split_dir=SPLITS_DIR,
    out_dir=None,
    n_examples=6,
):
    if out_dir is None:
        out_dir = Path(OUTPUT_DIR) / "figures" / "compare_explainability"

    out_dir = ensure_dir(out_dir)

    print("=" * 80)
    print("COMPARE EXPLAINABILITY")
    print("=" * 80)

    # -----------------------------------------------------
    # dataset
    # -----------------------------------------------------
    _, _, test_ds, _, _, test_df = build_datasets_from_split_csvs(
        split_dir=split_dir,
        image_size=IMAGE_SIZE,
        batch_size=16,
    )

    # predikciók (egy referencia modelllel)
    ref_model, _ = load_model_by_name(model_names[0])

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        probs = ref_model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # -----------------------------------------------------
    # példák kiválasztása
    # -----------------------------------------------------
    correct_idx = np.where(y_true == y_pred)[0]
    incorrect_idx = np.where(y_true != y_pred)[0]

    selected = []

    if len(correct_idx) > 0:
        selected.extend(correct_idx[: n_examples // 2])

    if len(incorrect_idx) > 0:
        selected.extend(incorrect_idx[: n_examples // 2])

    selected = selected[:n_examples]

    print(f"[INFO] Selected examples: {len(selected)}")

    # -----------------------------------------------------
    # modellek betöltése
    # -----------------------------------------------------
    models = {}
    for name in model_names:
        model, last_conv = load_model_by_name(name)
        models[name] = (model, last_conv)

    # -----------------------------------------------------
    # vizualizáció
    # -----------------------------------------------------
    for i, idx in enumerate(selected):
        row = test_df.iloc[idx]
        filepath = row["filepath"]

        raw = load_raw_image(filepath)
        proc = load_processed_input(filepath)

        input_tensor = tf.convert_to_tensor(proc[np.newaxis, ...])

        true_label = y_true[idx]

        # layout:
        # raw | processed | model1 | model2 | model3 ...
        ncols = 2 + len(models)
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

        # raw
        axes[0].imshow(raw[..., 0], cmap="gray")
        axes[0].set_title("Raw")
        axes[0].axis("off")

        # processed
        axes[1].imshow(proc[..., 0], cmap="gray")
        axes[1].set_title("Processed")
        axes[1].axis("off")

        # modellek
        for j, (name, (model, last_conv)) in enumerate(models.items()):
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

            ax = axes[2 + j]
            ax.imshow(overlay)
            ax.set_title(
                f"{name}\nP:{get_class_name(pred)} ({conf:.2f})",
                fontsize=9,
            )
            ax.axis("off")

        fig.suptitle(
            f"True: {get_class_name(true_label)}\n{Path(filepath).name}",
            fontsize=11,
        )

        plt.tight_layout()

        save_path = out_dir / f"{i:02d}_{Path(filepath).stem}.png"
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.show()

        print(f"[INFO] Saved: {save_path}")

    print("=" * 80)
    print("DONE")
    print("=" * 80)
