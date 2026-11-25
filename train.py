"""
train.py - Face Mask Detection CNN Training Script
Trains a binary classification CNN to predict whether a face is wearing a mask.
Saves:
  - models/face_mask_model.h5
  - models/classes.json
  - models/training_curves.png
"""

import os
import sys
import json
from pathlib import Path

# Configure environment before importing frameworks
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
matplotlib_dir = Path(__file__).resolve().parent / ".matplotlib"
matplotlib_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(matplotlib_dir)

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Configuration
DATASET_DIR = Path("dataset")
MODELS_DIR = Path("models")
CLASS_WITH_MASK = "with_mask"
CLASS_WITHOUT_MASK = "without_mask"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 4
LEARNING_RATE = 0.001
RANDOM_SEED = 42


def validate_dataset_structure():
    """Validates that the dataset directories exist and contain image files."""
    if not DATASET_DIR.exists():
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.", file=sys.stderr)
        print("Please create 'dataset/with_mask/' and 'dataset/without_mask/' and populate them with images.", file=sys.stderr)
        sys.exit(1)

    dir_with_mask = DATASET_DIR / CLASS_WITH_MASK
    dir_without_mask = DATASET_DIR / CLASS_WITHOUT_MASK

    if not dir_with_mask.exists():
        print(f"Error: Directory '{dir_with_mask}' not found.", file=sys.stderr)
        print(f"Add images to {dir_with_mask}/ before training.", file=sys.stderr)
        sys.exit(1)

    if not dir_without_mask.exists():
        print(f"Error: Directory '{dir_without_mask}' not found.", file=sys.stderr)
        print(f"Add images to {dir_without_mask}/ before training.", file=sys.stderr)
        sys.exit(1)

    with_mask_count = sum(1 for p in dir_with_mask.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    without_mask_count = sum(1 for p in dir_without_mask.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)

    print(f"Found {with_mask_count} images in {CLASS_WITH_MASK}/")
    print(f"Found {without_mask_count} images in {CLASS_WITHOUT_MASK}/")

    if with_mask_count == 0:
        print(f"Error: Add images to dataset/{CLASS_WITH_MASK}/ before training.", file=sys.stderr)
        sys.exit(1)

    if without_mask_count == 0:
        print(f"Error: Add images to dataset/{CLASS_WITHOUT_MASK}/ before training.", file=sys.stderr)
        sys.exit(1)

    if with_mask_count + without_mask_count < 10:
        print("Warning: Dataset is very small. Consider adding more images for better generalization.")


def build_model():
    """Builds the CNN architecture specified in the project guide."""
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Rescaling(1.0 / 255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_training_history(history, save_path):
    """Plots training and validation accuracy and loss curves."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#1f77b4", lw=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#2ca02c", lw=2)
    plt.title("Training and Validation Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#d62728", lw=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#ff7f0e", lw=2)
    plt.title("Training and Validation Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")


def main():
    print("=" * 60)
    print("Face Mask Detection CNN Training")
    print("=" * 60)

    validate_dataset_structure()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading dataset and creating 80/20 train/validation split...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=RANDOM_SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=RANDOM_SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    class_names = train_ds.class_names
    print(f"Class order: {class_names}")

    # Save class order
    classes_path = MODELS_DIR / "classes.json"
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved {classes_path}")

    # Performance optimization
    autotune = tf.data.AUTOTUNE
    train_ds_perf = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds_perf = val_ds.cache().prefetch(buffer_size=autotune)

    print("\nBuilding CNN model...")
    model = build_model()
    model.summary()

    callbacks_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print(f"\nStarting training for up to {EPOCHS} epochs (early stopping patience={PATIENCE})...")
    history = model.fit(
        train_ds_perf,
        validation_data=val_ds_perf,
        epochs=EPOCHS,
        callbacks=callbacks_list
    )

    print("\nEvaluating model on validation set...")
    val_loss, val_acc = model.evaluate(val_ds_perf, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}; loss: {val_loss:.4f}")

    # Save trained model in H5 format as required
    model_path = MODELS_DIR / "face_mask_model.h5"
    model.save(str(model_path))
    print(f"Saved {model_path}")

    # Save training curves
    curves_path = MODELS_DIR / "training_curves.png"
    plot_training_history(history, curves_path)

    print("\nTraining completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
