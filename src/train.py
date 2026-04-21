from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from src.config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    MODELS_DIR,
    NUM_CLASSES,
    NUM_EPOCHS_DEFAULT,
    PLOT_DPI,
    SEED,
    ensure_dir,
    get_class_names,
    get_data_root,
    save_json,
    set_global_seed,
)
from src.dataloader import build_datasets_from_split_csvs, build_default_augmentation


# =========================================================
# Model builders
# =========================================================

def build_baseline_cnn(
    input_shape: tuple[int, int, int] = (224, 224, 1),
    num_classes: int = NUM_CLASSES,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="baseline_cnn")


def build_transfer_model(
    model_name: str,
    input_shape: tuple[int, int, int] = (224, 224, 1),
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    name = model_name.lower()

    base_weights = "imagenet" if pretrained else None

    rgb_input = tf.keras.Input(shape=input_shape, name="grayscale_input")
    x = tf.keras.layers.Concatenate()([rgb_input, rgb_input, rgb_input])

    if name == "resnet50":
        preprocess = tf.keras.applications.resnet50.preprocess_input
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=base_weights,
            input_shape=(input_shape[0], input_shape[1], 3),
        )
    elif name == "vgg16":
        preprocess = tf.keras.applications.vgg16.preprocess_input
        base_model = tf.keras.applications.VGG16(
            include_top=False,
            weights=base_weights,
            input_shape=(input_shape[0], input_shape[1], 3),
        )
    elif name == "efficientnetb0":
        preprocess = tf.keras.applications.efficientnet.preprocess_input
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=base_weights,
            input_shape=(input_shape[0], input_shape[1], 3),
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    y = tf.keras.layers.Lambda(preprocess, name="preprocess_input")(x)
    y = base_model(y, training=False)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    y = tf.keras.layers.Dropout(0.3)(y)
    y = tf.keras.layers.Dense(256, activation="relu")(y)
    y = tf.keras.layers.Dropout(0.2)(y)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(y)

    model = tf.keras.Model(rgb_input, outputs, name=name)
    return model, base_model


def build_model(
    model_name: str,
    input_shape: tuple[int, int, int] = (224, 224, 1),
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> tuple[tf.keras.Model, tf.keras.Model | None]:
    name = model_name.lower()

    if name == "baseline_cnn":
        return build_baseline_cnn(input_shape=input_shape, num_classes=num_classes), None

    return build_transfer_model(
        model_name=name,
        input_shape=input_shape,
        num_classes=num_classes,
        pretrained=pretrained,
    )


# =========================================================
# Compile / callbacks
# =========================================================

def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def build_callbacks(out_dir: str | Path) -> list[tf.keras.callbacks.Callback]:
    out_dir = Path(out_dir)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(out_dir / "history.csv")),
    ]


# =========================================================
# Plot helpers
# =========================================================

def plot_training_history(
    history_df: pd.DataFrame,
    save_path: str | Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # loss
    axes[0].plot(history_df["loss"], label="train")
    if "val_loss" in history_df.columns:
        axes[0].plot(history_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # accuracy
    if "accuracy" in history_df.columns:
        axes[1].plot(history_df["accuracy"], label="train")
    if "val_accuracy" in history_df.columns:
        axes[1].plot(history_df["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # recall
    if "recall" in history_df.columns:
        axes[2].plot(history_df["recall"], label="train")
    if "val_recall" in history_df.columns:
        axes[2].plot(history_df["val_recall"], label="val")
    axes[2].set_title("Recall")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Training
# =========================================================

def run_training(
    split_dir: str | Path,
    out_dir: str | Path = MODELS_DIR,
    model_name: str = "baseline_cnn",
    pretrained: bool = True,
    do_fine_tuning: bool = False,
    epochs_head: int = NUM_EPOCHS_DEFAULT,
    epochs_finetune: int = 5,
    learning_rate_head: float = 1e-3,
    learning_rate_finetune: float = 1e-5,
    data_root: str | Path | None = None,
    data_variant: str = "raw",
    batch_size: int = BATCH_SIZE,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> dict[str, Any]:
    set_global_seed(SEED)

    if data_root is None:
        data_root = get_data_root(data_variant)
    data_root = Path(data_root)

    model_out_dir = ensure_dir(Path(out_dir) / f"{model_name}_{data_variant}")

    print("=" * 72)
    print("TRAINING")
    print("=" * 72)
    print("model_name   :", model_name)
    print("data_variant :", data_variant)
    print("data_root    :", data_root)
    print("split_dir    :", Path(split_dir))
    print("out_dir      :", model_out_dir)
    print("pretrained   :", pretrained)
    print("fine_tuning  :", do_fine_tuning)
    print("=" * 72)

    augmentation = build_default_augmentation()

    train_ds, val_ds, test_ds = build_datasets_from_split_csvs(
        split_dir=split_dir,
        data_root=data_root,
        batch_size=batch_size,
        augment_fn=augmentation,
        cache=False,
        image_size=image_size,
        channels=1,
    )

    model, base_model = build_model(
        model_name=model_name,
        input_shape=(image_size[0], image_size[1], 1),
        num_classes=NUM_CLASSES,
        pretrained=pretrained,
    )

    if base_model is not None:
        base_model.trainable = False

    compile_model(model, learning_rate=learning_rate_head)

    callbacks = build_callbacks(model_out_dir)

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_head,
        callbacks=callbacks,
        verbose=1,
    )

    history_frames = [pd.DataFrame(history_head.history)]
    history_frames[0]["phase"] = "head"

    if do_fine_tuning and base_model is not None and epochs_finetune > 0:
        base_model.trainable = True

        compile_model(model, learning_rate=learning_rate_finetune)

        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_head + epochs_finetune,
            initial_epoch=epochs_head,
            callbacks=callbacks,
            verbose=1,
        )

        ft_df = pd.DataFrame(history_ft.history)
        ft_df["phase"] = "finetune"
        history_frames.append(ft_df)

    history_df = pd.concat(history_frames, ignore_index=True)
    history_df.to_csv(model_out_dir / "history_full.csv", index=False)

    plot_training_history(
        history_df=history_df,
        save_path=model_out_dir / "training_history.png",
        title=f"Training history - {model_name} ({data_variant})",
    )

    best_model_path = model_out_dir / "best_model.keras"
    final_model_path = model_out_dir / "last_model.keras"

    model.save(final_model_path)

    summary = {
        "model_name": model_name,
        "data_variant": data_variant,
        "data_root": str(data_root),
        "split_dir": str(Path(split_dir)),
        "out_dir": str(model_out_dir),
        "best_model_path": str(best_model_path),
        "last_model_path": str(final_model_path),
        "class_names": get_class_names(),
        "pretrained": bool(pretrained),
        "do_fine_tuning": bool(do_fine_tuning),
        "epochs_head": int(epochs_head),
        "epochs_finetune": int(epochs_finetune),
        "learning_rate_head": float(learning_rate_head),
        "learning_rate_finetune": float(learning_rate_finetune),
    }
    save_json(summary, model_out_dir / "train_summary.json")

    return summary
