"""Train the face mask classification model."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


DATASET_DIR = Path("dataset")
MODELS_DIR = Path("models")
CLASS_NAMES = ["with_mask", "without_mask"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 123


def build_model():
    """Create the CNN used to classify face crops."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
            keras.layers.Rescaling(1.0 / 255),
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_curves(history, output_path):
    """Save loss and accuracy plots from the training history."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for metric, axis, title in (
        ("accuracy", axes[0], "Accuracy"),
        ("loss", axes[1], "Loss"),
    ):
        axis.plot(history.history[metric], label="Training")
        axis.plot(history.history[f"val_{metric}"], label="Validation")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_options = {
        "labels": "inferred",
        "label_mode": "binary",
        "class_names": CLASS_NAMES,
        "validation_split": 0.2,
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
    }
    try:
        training_data = keras.utils.image_dataset_from_directory(
            DATASET_DIR, subset="training", **dataset_options
        )
        validation_data = keras.utils.image_dataset_from_directory(
            DATASET_DIR, subset="validation", **dataset_options
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(
            "Could not load the dataset. Add images to "
            "dataset/with_mask/ and dataset/without_mask/."
        ) from error

    autotune = tf.data.AUTOTUNE
    training_data = training_data.prefetch(buffer_size=autotune)
    validation_data = validation_data.prefetch(buffer_size=autotune)

    model = build_model()
    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ],
    )

    model.save(MODELS_DIR / "face_mask_model.h5")
    (MODELS_DIR / "classes.json").write_text(
        json.dumps(CLASS_NAMES, indent=2) + "\n", encoding="utf-8"
    )
    save_training_curves(history, MODELS_DIR / "training_curves.png")
    print(f"Training outputs saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
