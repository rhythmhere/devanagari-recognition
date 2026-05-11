"""
Training Script for Nepali Currency Note Denomination Classification.
Secondary advanced module of the Devanagari Character Recognition project.

Features:
  - MobileNetV2 transfer learning
  - Data augmentation (rotation, shift, zoom, brightness, flip)
  - ReduceLROnPlateau + EarlyStopping + ModelCheckpoint
  - Training history saved to JSON
  - Training plots saved as PNG
  - Label mapping saved to JSON

Expected data directory structure:
  data/note_train/
    rs5/       (images of Rs. 5 notes)
    rs10/      (images of Rs. 10 notes)
    rs20/      (images of Rs. 20 notes)
    rs50/      (images of Rs. 50 notes)pytho
    rs100/     (images of Rs. 100 notes)
    rs500/     (images of Rs. 500 notes)
    rs1000/    (images of Rs. 1000 notes)
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from note_model import build_note_model, get_note_model_summary_text

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "note_train")
SAVE_DIR = os.path.join(BASE_DIR, "backend", "saved_model")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30

# ── Data Generators ───────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=5,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

print(f"[INFO] Looking for note training data in: {TRAIN_DIR}")

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"[INFO] Found {num_classes} note denomination classes")
print(f"[INFO] Training samples: {train_generator.samples}")
print(f"[INFO] Validation samples: {val_generator.samples}")
print(f"[INFO] Classes: {train_generator.class_indices}")

# ── Build model ───────────────────────────────────────────────────
model = build_note_model(num_classes, input_shape=(IMG_SIZE, IMG_SIZE, 3))
model.summary()

# Save model summary
summary_text = get_note_model_summary_text(model)
with open(os.path.join(SAVE_DIR, "note_model_summary.txt"), "w") as f:
    f.write(summary_text)
print("[INFO] Note model summary saved.")

# ── Save label mapping ────────────────────────────────────────────
idx_to_label = {v: k for k, v in train_generator.class_indices.items()}
with open(os.path.join(SAVE_DIR, "note_labels.json"), "w", encoding="utf-8") as f:
    json.dump(idx_to_label, f, ensure_ascii=False, indent=2)
print("[INFO] Note label mapping saved.")

# ── Callbacks ─────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(SAVE_DIR, "note_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
]

# ── Train ─────────────────────────────────────────────────────────
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ── Final evaluation ──────────────────────────────────────────────
val_loss, val_acc = model.evaluate(val_generator, verbose=1)
print(f"\nNote Model Validation Accuracy: {val_acc:.4f}")
print(f"Note Model Validation Loss:    {val_loss:.4f}")

# Save final model
model.save(os.path.join(SAVE_DIR, "note_final_model.keras"))

# ── Save training history to JSON ─────────────────────────────────
history_data = {
    "accuracy": [round(float(v), 4) for v in history.history["accuracy"]],
    "val_accuracy": [round(float(v), 4) for v in history.history["val_accuracy"]],
    "loss": [round(float(v), 4) for v in history.history["loss"]],
    "val_loss": [round(float(v), 4) for v in history.history["val_loss"]],
    "epochs_completed": len(history.history["accuracy"]),
    "final_val_accuracy": round(float(val_acc), 4),
    "final_val_loss": round(float(val_loss), 4),
}
if "lr" in history.history:
    history_data["lr"] = [round(float(v), 8) for v in history.history["lr"]]

with open(os.path.join(SAVE_DIR, "note_training_history.json"), "w") as f:
    json.dump(history_data, f, indent=2)
print("[INFO] Note training history JSON saved.")

# ── Plot: Accuracy & Loss Curves ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
axes[0].set_title("Note Model Accuracy", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["loss"], label="Train Loss", linewidth=2)
axes[1].plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
axes[1].set_title("Note Model Loss", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "note_training_plot.png"), dpi=150)
plt.close()

print("\n" + "=" * 60)
print("  NOTE MODEL TRAINING COMPLETE")
print(f"  Best model:  {os.path.join(SAVE_DIR, 'note_model.keras')}")
print(f"  Final model: {os.path.join(SAVE_DIR, 'note_final_model.keras')}")
print(f"  Val Accuracy: {val_acc:.4f}")
print("=" * 60)
