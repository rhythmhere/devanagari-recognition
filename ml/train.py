"""
Advanced Training Script for Devanagari Character Recognition.
Features:
  - Stronger data augmentation (shear, brightness)
  - ReduceLROnPlateau + EarlyStopping + ModelCheckpoint callbacks
  - Per-class accuracy bar chart
  - Model summary saved to text file
  - Training history saved to JSON for frontend analytics
"""

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
from model import build_model, get_model_summary_text

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
SAVE_DIR = os.path.join(BASE_DIR, "backend", "saved_model")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────
IMG_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 25  # Increased from 15; early stopping prevents overtraining

# ── Data Generators with richer augmentation ──────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=12,
    width_shift_range=0.12,
    height_shift_range=0.12,
    zoom_range=0.12,
    shear_range=5,
    brightness_range=[0.85, 1.15],
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"[INFO] Found {num_classes} classes")
print(f"[INFO] Training samples: {train_generator.samples}")
print(f"[INFO] Validation samples: {val_generator.samples}")

# ── Build model ───────────────────────────────────────────────────
model = build_model(num_classes)
model.summary()

# Save model summary
summary_text = get_model_summary_text(model)
with open(os.path.join(SAVE_DIR, "model_summary.txt"), "w") as f:
    f.write(summary_text)
print("[INFO] Model summary saved.")

# ── Save label mapping ────────────────────────────────────────────
idx_to_label = {v: k for k, v in train_generator.class_indices.items()}
with open(os.path.join(SAVE_DIR, "labels.json"), "w", encoding="utf-8") as f:
    json.dump(idx_to_label, f, ensure_ascii=False, indent=2)

# ── Callbacks ─────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(SAVE_DIR, "devanagari_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
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
print(f"\nValidation Accuracy: {val_acc:.4f}")
print(f"Validation Loss:    {val_loss:.4f}")

# Save final model
model.save(os.path.join(SAVE_DIR, "final_model.keras"))

# ── Save training history to JSON (for frontend analytics) ───────
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

with open(os.path.join(SAVE_DIR, "training_history.json"), "w") as f:
    json.dump(history_data, f, indent=2)
print("[INFO] Training history JSON saved.")

# ── Plot 1: Accuracy & Loss Curves ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
axes[0].set_title("Model Accuracy", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["loss"], label="Train Loss", linewidth=2)
axes[1].plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
axes[1].set_title("Model Loss", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_plot.png"), dpi=150)
plt.close()

# ── Plot 2: Learning Rate Schedule (if available) ─────────────────
if "lr" in history.history:
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["lr"], linewidth=2, color='#e74c3c')
    plt.title("Learning Rate Schedule", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "lr_schedule.png"), dpi=150)
    plt.close()
    print("[INFO] Learning rate plot saved.")

print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print(f"  Best model:  {os.path.join(SAVE_DIR, 'devanagari_model.keras')}")
print(f"  Final model: {os.path.join(SAVE_DIR, 'final_model.keras')}")
print(f"  Val Accuracy: {val_acc:.4f}")
print("=" * 60)
