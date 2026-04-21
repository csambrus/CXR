# src/train_cnn.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import tensorflow as tf

from src.config import (
    BATCH_SIZE,
    CLASS_INFOS,
    IMAGE_SIZE,
    NUM_CLASSES,
    SEED,
    MODELS_DIR,
    SPLITS_DIR,
    ensure_dir,
    get_class_names,
    save_json,
)
from src.dataloader import build_datasets_from_split_csvs


def set_global_seed(seed: int = SEED) -> None:
    tf.keras.utils.set_random_seed(seed)


# =========================================================
# Metrics
# =========================================================

def build_metrics() -> list[tf.keras.metrics.Metric]:
    return [
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    ]


# =========================================================
# Input / preprocess blokkok
# =========================================================

def build_input_block(image_size: tuple[int, int] = IMAGE_SIZE):
    """
    A dataloader 1 csatornás képet ad.
    A pretrained ImageNet backbone-okhoz 3 csatornára alakítjuk
    kizárólag built-in Keras rétegekkel, hogy a modell jól
    sorosítható / visszatölthető legyen.
    """
    inputs = tf.keras.Input(
        shape=(image_size[0], image_size[1], 1),
        name="image_input",
    )

    # [H, W, 1] -> [H, W, 3]
    x = tf.keras.layers.Concatenate(axis=-1, name="gray_to_rgb")(
        [inputs, inputs, inputs]
    )

    return inputs, x


def apply_resnet_vgg_preprocess(
    x: tf.Tensor,
    name_prefix: str,
) -> tf.Tensor:
    """
    ResNet50 / VGG16 preprocess_input built-in rétegekkel,
    Lambda nélkül.

    Mivel a bemenetünk grayscale-ből 3 azonos csatornára másolt kép,
    az RGB->BGR csatornacsere nem változtat a pixeleken.
    Így elég:
      1) [0,1] -> [0,255]
      2) csatornánként mean kivonás
    """
    x = tf.keras.layers.Rescaling(
        scale=255.0,
        name=f"{name_prefix}_scale255",
    )(x)

    # Caffe-style mean subtraction
    mean = [103.939, 116.779, 123.68]
    variance = [1.0, 1.0, 1.0]

    x = tf.keras.layers.Normalization(
        mean=mean,
        variance=variance,
        axis=-1,
        name=f"{name_prefix}_mean_subtract",
    )(x)

    return x


def apply_efficientnet_preprocess(
    x: tf.Tensor,
    name_prefix: str,
) -> tf.Tensor:
    """
    Az EfficientNetB0 Keras modellben a preprocessing a modellen belül van,
    és [0,255] skálájú inputot vár.
    A pipeline viszont [0,1]-et ad, ezért itt felszorozzuk 255-re.
    """
    x = tf.keras.layers.Rescaling(
        scale=255.0,
        name=f"{name_prefix}_scale255",
    )(x)
    return x


# =========================================================
# Classifier head
# =========================================================

def build_classifier_head(
    x: tf.Tensor,
    num_classes: int = NUM_CLASSES,
    dropout: float = 0.30,
) -> tf.Tensor:
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)
    return outputs


# =========================================================
# Modellek
# =========================================================

def build_resnet50(
    image_size: tuple[int, int] = IMAGE_SIZE,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    train_backbone: bool = False,
    dropout: float = 0.30,
):
    inputs, x = build_input_block(image_size)
    x = apply_resnet_vgg_preprocess(x, name_prefix="resnet50_preprocess")

    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_tensor=x,
    )
    backbone.trainable = train_backbone

    outputs = build_classifier_head(
        backbone.output,
        num_classes=num_classes,
        dropout=dropout,
    )
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="resnet50_classifier",
    )
    return model, backbone


def build_vgg16(
    image_size: tuple[int, int] = IMAGE_SIZE,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    train_backbone: bool = False,
    dropout: float = 0.30,
):
    inputs, x = build_input_block(image_size)
    x = apply_resnet_vgg_preprocess(x, name_prefix="vgg16_preprocess")

    backbone = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_tensor=x,
    )
    backbone.trainable = train_backbone

    outputs = build_classifier_head(
        backbone.output,
        num_classes=num_classes,
        dropout=dropout,
    )
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="vgg16_classifier",
    )
    return model, backbone


def build_efficientnetb0(
    image_size: tuple[int, int] = IMAGE_SIZE,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    train_backbone: bool = False,
    dropout: float = 0.30,
):
    inputs, x = build_input_block(image_size)
    x = apply_efficientnet_preprocess(x, name_prefix="efficientnetb0_preprocess")

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_tensor=x,
    )
    backbone.trainable = train_backbone

    outputs = build_classifier_head(
        backbone.output,
        num_classes=num_classes,
        dropout=dropout,
    )
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="efficientnetb0_classifier",
    )
    return model, backbone


def build_baseline_cnn(
    image_size: tuple[int, int] = IMAGE_SIZE,
    num_classes: int = NUM_CLASSES,
):
    inputs = tf.keras.Input(
        shape=(image_size[0], image_size[1], 1),
        name="image_input",
    )

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")
    backbone = None
    return model, backbone


def build_model(
    model_name: str = "resnet50",
    image_size: tuple[int, int] = IMAGE_SIZE,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    train_backbone: bool = False,
):
    model_name = model_name.lower().strip()

    if model_name == "resnet50":
        return build_resnet50(
            image_size=image_size,
            num_classes=num_classes,
            pretrained=pretrained,
            train_backbone=train_backbone,
        )

    if model_name == "vgg16":
        return build_vgg16(
            image_size=image_size,
            num_classes=num_classes,
            pretrained=pretrained,
            train_backbone=train_backbone,
        )

    if model_name == "efficientnetb0":
        return build_efficientnetb0(
            image_size=image_size,
            num_classes=num_classes,
            pretrained=pretrained,
            train_backbone=train_backbone,
        )

    if model_name == "baseline_cnn":
        return build_baseline_cnn(
            image_size=image_size,
            num_classes=num_classes,
        )

    raise ValueError(
        f"Ismeretlen model_name: {model_name}. "
        "Lehetséges értékek: resnet50, vgg16, efficientnetb0, baseline_cnn"
    )


# =========================================================
# Compile / callbacks
# =========================================================

def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=build_metrics(),
        jit_compile=False,
    )
    return model


def build_callbacks(
    out_dir: str | Path,
    monitor: str = "val_accuracy",
    early_stopping_patience: int = 4,
    reduce_lr_patience: int = 2,
):
    out_dir = Path(out_dir)
    callbacks: list[tf.keras.callbacks.Callback] = []

    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        )
    )

    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode="max",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        )
    )

    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best_model.keras"),
            monitor=monitor,
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
    )

    callbacks.append(
        tf.keras.callbacks.CSVLogger(
            filename=str(out_dir / "history.csv"),
            separator=",",
            append=False,
        )
    )

    return callbacks


# =========================================================
# Fine-tune helper
# =========================================================

def unfreeze_top_layers(
    backbone: tf.keras.Model | None,
    fraction: float = 0.20,
) -> int:
    """
    A backbone felső részét kinyitja fine-tuninghoz.
    """
    if backbone is None:
        return 0

    layers = backbone.layers
    n_total = len(layers)
    n_unfreeze = max(1, int(round(n_total * fraction)))
    split_idx = n_total - n_unfreeze

    backbone.trainable = True

    for i, layer in enumerate(layers):
        layer.trainable = i >= split_idx

    return n_unfreeze


# =========================================================
# History / metrics mentés
# =========================================================

def save_history(history: tf.keras.callbacks.History, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(out_dir / "history_epochwise.csv", index=False)

    history_json = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }
    save_json(history_json, out_dir / "history.json")


def evaluate_and_save(
    model: tf.keras.Model,
    train_ds,
    val_ds,
    test_ds,
    out_dir: str | Path,
) -> dict[str, dict[str, float]]:
    out_dir = Path(out_dir)

    train_metrics = model.evaluate(train_ds, verbose=0, return_dict=True)
    val_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)
    test_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)

    metrics = {
        "train": {k: float(v) for k, v in train_metrics.items()},
        "val": {k: float(v) for k, v in val_metrics.items()},
        "test": {k: float(v) for k, v in test_metrics.items()},
    }

    save_json(metrics, out_dir / "metrics.json")
    return metrics


def is_training_complete(model_out_dir: str | Path) -> bool:
    model_out_dir = Path(model_out_dir)

    required_files = [
        model_out_dir / "final_model.keras",
        model_out_dir / "metrics.json",
        model_out_dir / "training_config.json",
    ]
    return all(p.exists() for p in required_files)

# =========================================================
# Training
# =========================================================

def run_training(
    split_dir: str | Path,
    out_dir: str | Path = MODELS_DIR,
    model_name: str = "resnet50",
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    pretrained: bool = True,
    train_backbone: bool = False,
    do_fine_tuning: bool = False,
    fine_tune_fraction: float = 0.20,
    epochs_head: int = 8,
    epochs_finetune: int = 5,
    learning_rate_head: float = 1e-3,
    learning_rate_finetune: float = 1e-5,
):
    set_global_seed(seed)

    out_dir = Path(out_dir) / model_name.lower()
    ensure_dir(out_dir)

    # -----------------------------------------------------
    # Datasetek
    # -----------------------------------------------------
    train_ds, val_ds, test_ds, train_df, val_df, test_df = build_datasets_from_split_csvs(
        split_dir=split_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )
    
    # -----------------------------------------------------
    # Speed test
    # -----------------------------------------------------
    import time
    print("\n[INFO] Dataset quick check...")
    t0 = time.time()
    for i, (x, y) in enumerate(train_ds.take(3)):
        print(f"  batch {i}: x={x.shape}, y={y.shape}")
    print(f"[INFO] 3 batch load time: {time.time() - t0:.2f} sec")

    
    # -----------------------------------------------------
    # Modell
    # -----------------------------------------------------
    model, backbone = build_model(
        model_name=model_name,
        image_size=image_size,
        num_classes=NUM_CLASSES,
        pretrained=pretrained,
        train_backbone=train_backbone,
    )

    model = compile_model(model, learning_rate=learning_rate_head)

    config_snapshot = {
        "model_name": model_name,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "batch_size": int(batch_size),
        "seed": int(seed),
        "num_classes": int(NUM_CLASSES),
        "class_names": get_class_names(),
        "pretrained": bool(pretrained),
        "train_backbone": bool(train_backbone),
        "do_fine_tuning": bool(do_fine_tuning),
        "fine_tune_fraction": float(fine_tune_fraction),
        "epochs_head": int(epochs_head),
        "epochs_finetune": int(epochs_finetune),
        "learning_rate_head": float(learning_rate_head),
        "learning_rate_finetune": float(learning_rate_finetune),
        "split_dir": str(split_dir),
        "dataset_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
    }
    save_json(config_snapshot, out_dir / "training_config.json")

    # -----------------------------------------------------
    # Phase 1: head training
    # -----------------------------------------------------
    print("\n" + "=" * 80)
    print(f"TRAINING START: {model_name}")
    print("=" * 80)
    model.summary()

    callbacks = build_callbacks(out_dir=out_dir, monitor="val_accuracy")

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_head,
        callbacks=callbacks,
        verbose=1,
    )

    all_history = dict(history_head.history)

    # -----------------------------------------------------
    # Phase 2: fine tuning opcionálisan
    # -----------------------------------------------------
    if do_fine_tuning and backbone is not None:
        print("\n" + "=" * 80)
        print("FINE-TUNING PHASE")
        print("=" * 80)

        n_unfrozen = unfreeze_top_layers(backbone, fraction=fine_tune_fraction)
        print(f"[INFO] Unfrozen top layers: {n_unfrozen}")

        model = compile_model(model, learning_rate=learning_rate_finetune)

        fine_tune_callbacks = build_callbacks(
            out_dir=out_dir,
            monitor="val_accuracy",
        )

        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_head + epochs_finetune,
            initial_epoch=len(history_head.history["loss"]),
            callbacks=fine_tune_callbacks,
            verbose=1,
        )

        for key, values in history_ft.history.items():
            if key in all_history:
                all_history[key].extend(values)
            else:
                all_history[key] = list(values)

    # -----------------------------------------------------
    # Mentések
    # -----------------------------------------------------
    final_model_path = out_dir / "final_model.keras"
    model.save(final_model_path)

    save_json({"class_names": get_class_names()}, out_dir / "class_names.json")

    history_obj = type("HistoryObj", (), {"history": all_history})()
    save_history(history_obj, out_dir)

    metrics = evaluate_and_save(model, train_ds, val_ds, test_ds, out_dir)

    print("\nFinal metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    return {
        "model": model,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "metrics": metrics,
        "out_dir": out_dir,
        "best_model_path": out_dir / "best_model.keras",
        "final_model_path": final_model_path,
        "history_csv": out_dir / "history.csv",
        "history_json": out_dir / "history.json",
        "metrics_json": out_dir / "metrics.json",
        "config_json": out_dir / "training_config.json",
    }


# =========================================================
# Több modell futtatása összehasonlításhoz
# =========================================================

def run_multiple_models(
    split_dir: str | Path,
    out_dir: str | Path = MODELS_DIR,
    model_names: list[str] | tuple[str, ...] = (
        "resnet50",
        "vgg16",
        "efficientnetb0",
    ),
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    pretrained: bool = True,
    do_fine_tuning: bool = False,
    fine_tune_fraction: float = 0.20,
    epochs_head: int = 8,
    epochs_finetune: int = 5,
    learning_rate_head: float = 1e-3,
    learning_rate_finetune: float = 1e-5,
) -> pd.DataFrame:
    rows = []

    for model_name in model_names:
        model_dir = Path(out_dir) / model_name.lower()
        if is_training_complete(model_dir):
            print(f"[SKIP] {model_name} already completed: {model_dir}")

            metrics_path = model_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    saved_metrics = json.load(f)
                row = {
                    "model_name": model_name,
                    "train_loss": saved_metrics["train"]["loss"],
                    "train_accuracy": saved_metrics["train"]["accuracy"],
                    "val_loss": saved_metrics["val"]["loss"],
                    "val_accuracy": saved_metrics["val"]["accuracy"],
                    "test_loss": saved_metrics["test"]["loss"],
                    "test_accuracy": saved_metrics["test"]["accuracy"],
                    "out_dir": str(model_dir),
                }
                rows.append(row)
            continue
        
        result = run_training(
            split_dir=split_dir,
            out_dir=out_dir,
            model_name=model_name,
            image_size=image_size,
            batch_size=batch_size,
            seed=seed,
            pretrained=pretrained,
            train_backbone=False,
            do_fine_tuning=do_fine_tuning,
            fine_tune_fraction=fine_tune_fraction,
            epochs_head=epochs_head,
            epochs_finetune=epochs_finetune,
            learning_rate_head=learning_rate_head,
            learning_rate_finetune=learning_rate_finetune,
        )

        row = {
            "model_name": model_name,
            "train_loss": result["metrics"]["train"]["loss"],
            "train_accuracy": result["metrics"]["train"]["accuracy"],
            "val_loss": result["metrics"]["val"]["loss"],
            "val_accuracy": result["metrics"]["val"]["accuracy"],
            "test_loss": result["metrics"]["test"]["loss"],
            "test_accuracy": result["metrics"]["test"]["accuracy"],
            "out_dir": str(result["out_dir"]),
        }
        rows.append(row)

    comparison_df = (
        pd.DataFrame(rows)
        .sort_values("val_accuracy", ascending=False)
        .reset_index(drop=True)
    )

    comparison_dir = ensure_dir(Path(out_dir))
    comparison_path = comparison_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print(f"[INFO] Saved comparison table: {comparison_path}")
    print(comparison_df)

    return comparison_df


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    run_training(
        split_dir=SPLITS_DIR,
        out_dir=MODELS_DIR,
        model_name="resnet50",
        pretrained=True,
        do_fine_tuning=False,
    )
