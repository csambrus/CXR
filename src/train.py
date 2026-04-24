# train.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from src.runtime import set_seed

from src.config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    MODELS_DIR,
    NUM_CLASSES,
    EPOCHS,
    PLOT_DPI,
    SEED,
    ensure_dir,
    get_class_names,
    get_data_root,
    save_json,
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
    """
    Transfer model grayscale -> RGB.

    Fontos:
    Nem használunk Lambda(preprocess_input) réteget, mert Keras 3 / TF 2.19 alatt
    .keras mentés-betöltéskor deszerializációs hibát okozhat.
    """
    name = model_name.lower()
    base_weights = "imagenet" if pretrained else None

    inputs = tf.keras.Input(shape=input_shape, name="grayscale_input")

    x = tf.keras.layers.Concatenate(name="gray_to_rgb")(
        [inputs, inputs, inputs]
    )

    if name == "resnet50":
        # A dataloader 0..1 közé skáláz. ResNet/VGG ImageNet súlyokhoz
        # legalább 0..255 tartományra visszaskálázzuk, Lambda nélkül.
        x = tf.keras.layers.Rescaling(255.0, name="scale_to_255")(x)
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=base_weights,
            input_shape=(input_shape[0], input_shape[1], 3),
        )

    elif name == "vgg16":
        x = tf.keras.layers.Rescaling(255.0, name="scale_to_255")(x)
        base_model = tf.keras.applications.VGG16(
            include_top=False,
            weights=base_weights,
            input_shape=(input_shape[0], input_shape[1], 3),
        )

    elif name == "efficientnetb0":
        # EfficientNetB0 Keras alatt 0..1 inputtal is jól kezelhető,
        # ezért itt nem skálázzuk vissza 255-re.
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=base_weights,
            input_shape=(input_shape[0], input_shape[1], 3),
        )

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.3, name="head_dropout_1")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(0.2, name="head_dropout_2")(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = tf.keras.Model(inputs, outputs, name=name)
    return model, base_model


def build_model(
    model_name: str,
    input_shape: tuple[int, int, int] = (224, 224, 1),
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> tuple[tf.keras.Model, tf.keras.Model | None]:
    name = model_name.lower()

    if name == "baseline_cnn":
        model = build_baseline_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
        )
        return model, None

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
    """
    Sparse multiclass setup:

    - labels: integer class ids, shape: (batch,)
    - predictions: softmax probabilities, shape: (batch, NUM_CLASSES)

    Ne használjunk itt tf.keras.metrics.Recall / Precision / AUC metrikákat,
    mert azok sparse multiclass esetben shape hibát okozhatnak.
    Macro recall / F1 / ROC-AUC később evaluate.py-ban számolandó.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )


def build_callbacks(out_dir: str | Path) -> list[tf.keras.callbacks.Callback]:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

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
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "loss" in history_df.columns:
        axes[0].plot(history_df["loss"], label="train")
    if "val_loss" in history_df.columns:
        axes[0].plot(history_df["val_loss"], label="val")

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if "accuracy" in history_df.columns:
        axes[1].plot(history_df["accuracy"], label="train")
    if "val_accuracy" in history_df.columns:
        axes[1].plot(history_df["val_accuracy"], label="val")

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

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
    epochs_head: int = EPOCHS,
    epochs_finetune: int = 5,
    learning_rate_head: float = 1e-3,
    learning_rate_finetune: float = 1e-5,
    data_root: str | Path | None = None,
    data_variant: str = "raw",
    batch_size: int = BATCH_SIZE,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> dict[str, Any]:
    set_seed(SEED)

    if data_root is None:
        data_root = get_data_root(data_variant)

    data_root = Path(data_root)
    split_dir = Path(split_dir)

    model_out_dir = ensure_dir(Path(out_dir) / f"{model_name}_{data_variant}")

    print("=" * 72)
    print("TRAINING")
    print("=" * 72)
    print("model_name   :", model_name)
    print("data_variant :", data_variant)
    print("data_root    :", data_root)
    print("split_dir    :", split_dir)
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

    history_frames: list[pd.DataFrame] = []

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_head,
        callbacks=callbacks,
        verbose=1,
    )

    head_df = pd.DataFrame(history_head.history)
    head_df["phase"] = "head"
    head_df["epoch_global"] = range(1, len(head_df) + 1)
    history_frames.append(head_df)

    if do_fine_tuning and base_model is not None and epochs_finetune > 0:
        print("=" * 72)
        print("FINE-TUNING")
        print("=" * 72)

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
        ft_df["epoch_global"] = range(
            len(head_df) + 1,
            len(head_df) + len(ft_df) + 1,
        )
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
        "split_dir": str(split_dir),
        "out_dir": str(model_out_dir),
        "best_model_path": str(best_model_path),
        "last_model_path": str(final_model_path),
        "class_names": get_class_names(),
        "num_classes": int(NUM_CLASSES),
        "input_shape": [int(image_size[0]), int(image_size[1]), 1],
        "pretrained": bool(pretrained),
        "do_fine_tuning": bool(do_fine_tuning),
        "epochs_head": int(epochs_head),
        "epochs_finetune": int(epochs_finetune),
        "learning_rate_head": float(learning_rate_head),
        "learning_rate_finetune": float(learning_rate_finetune),
        "batch_size": int(batch_size),
        "metrics": ["accuracy"],
        "loss": "sparse_categorical_crossentropy",
    }

    save_json(summary, model_out_dir / "train_summary.json")

    print("=" * 72)
    print("[OK] Training finished")
    print("=" * 72)
    print("model_name      :", model_name)
    print("data_variant    :", data_variant)
    print("best_model_path :", best_model_path)
    print("last_model_path :", final_model_path)
    print("=" * 72)

    return summary
