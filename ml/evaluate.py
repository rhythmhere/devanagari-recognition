"""
Advanced Evaluation Script for Devanagari Character Recognition.
Features:
  - Full classification report saved to text + JSON
  - Annotated confusion matrix heatmap
  - Top-10 most confused character pairs analysis
  - Per-class accuracy bar chart
  - Evaluation summary JSON for frontend dashboard
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
SAVE_DIR = os.path.join(BASE_DIR, "backend", "saved_model")

IMG_SIZE = 32
BATCH_SIZE = 64

# ── Load model and labels ────────────────────────────────────────
print("[INFO] Loading model...")
model = load_model(os.path.join(SAVE_DIR, "devanagari_model.keras"))

with open(os.path.join(SAVE_DIR, "labels.json"), "r", encoding="utf-8") as f:
    idx_to_label = json.load(f)

# ── Prepare validation data ──────────────────────────────────────
datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ── Predictions ───────────────────────────────────────────────────
print("[INFO] Running predictions on validation set...")
pred_probs = model.predict(val_generator, verbose=1)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = val_generator.classes

labels = [idx_to_label[str(i)] for i in range(len(idx_to_label))]
num_classes = len(labels)

# ── Overall Metrics ───────────────────────────────────────────────
overall_accuracy = accuracy_score(true_classes, pred_classes)
macro_precision = precision_score(true_classes, pred_classes, average='macro')
macro_recall = recall_score(true_classes, pred_classes, average='macro')
macro_f1 = f1_score(true_classes, pred_classes, average='macro')
weighted_f1 = f1_score(true_classes, pred_classes, average='weighted')

print(f"\n{'=' * 50}")
print(f"  Overall Accuracy:     {overall_accuracy:.4f}")
print(f"  Macro Precision:      {macro_precision:.4f}")
print(f"  Macro Recall:         {macro_recall:.4f}")
print(f"  Macro F1-Score:       {macro_f1:.4f}")
print(f"  Weighted F1-Score:    {weighted_f1:.4f}")
print(f"{'=' * 50}\n")

# ── Classification Report ────────────────────────────────────────
report_text = classification_report(true_classes, pred_classes, target_names=labels)
print(report_text)

# Save text report
with open(os.path.join(SAVE_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
    f.write(f"Macro Precision:  {macro_precision:.4f}\n")
    f.write(f"Macro Recall:     {macro_recall:.4f}\n")
    f.write(f"Macro F1-Score:   {macro_f1:.4f}\n")
    f.write(f"Weighted F1:      {weighted_f1:.4f}\n\n")
    f.write(report_text)

# Save JSON report for frontend
report_dict = classification_report(
    true_classes, pred_classes, target_names=labels, output_dict=True
)
per_class_metrics = {}
for label in labels:
    if label in report_dict:
        per_class_metrics[label] = {
            "precision": round(report_dict[label]["precision"], 4),
            "recall": round(report_dict[label]["recall"], 4),
            "f1_score": round(report_dict[label]["f1-score"], 4),
            "support": int(report_dict[label]["support"]),
        }

eval_summary = {
    "overall_accuracy": round(overall_accuracy, 4),
    "macro_precision": round(macro_precision, 4),
    "macro_recall": round(macro_recall, 4),
    "macro_f1": round(macro_f1, 4),
    "weighted_f1": round(weighted_f1, 4),
    "num_classes": num_classes,
    "total_samples": int(len(true_classes)),
    "per_class": per_class_metrics,
}

with open(os.path.join(SAVE_DIR, "evaluation_summary.json"), "w") as f:
    json.dump(eval_summary, f, indent=2)
print("[INFO] Evaluation summary JSON saved.")

# ── Confusion Matrix ─────────────────────────────────────────────
cm = confusion_matrix(true_classes, pred_classes)

# Large annotated heatmap
plt.figure(figsize=(20, 18))
sns.heatmap(cm, cmap="Blues", xticklabels=labels, yticklabels=labels,
            fmt='d', linewidths=0.3, linecolor='white',
            cbar_kws={'shrink': 0.8})
plt.title("Confusion Matrix – Devanagari Character Recognition", fontsize=16, fontweight='bold')
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(fontsize=6, rotation=90)
plt.yticks(fontsize=6, rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("[INFO] Confusion matrix saved.")

# ── Top-10 Most Confused Pairs ────────────────────────────────────
confused_pairs = []
for i in range(num_classes):
    for j in range(num_classes):
        if i != j and cm[i][j] > 0:
            confused_pairs.append({
                "true_label": labels[i],
                "predicted_label": labels[j],
                "count": int(cm[i][j]),
            })

confused_pairs.sort(key=lambda x: x["count"], reverse=True)
top_confused = confused_pairs[:10]

print("\nTop 10 Most Confused Pairs:")
print(f"{'True':<25} {'Predicted':<25} {'Count'}")
print("-" * 60)
for pair in top_confused:
    print(f"{pair['true_label']:<25} {pair['predicted_label']:<25} {pair['count']}")

with open(os.path.join(SAVE_DIR, "top_confused_pairs.json"), "w") as f:
    json.dump(top_confused, f, indent=2)

# ── Per-Class Accuracy Bar Chart ──────────────────────────────────
per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

fig, ax = plt.subplots(figsize=(18, 6))
colors = ['#e74c3c' if acc < 0.8 else '#f39c12' if acc < 0.9 else '#27ae60'
          for acc in per_class_acc]
ax.bar(range(num_classes), per_class_acc, color=colors, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(num_classes))
ax.set_xticklabels(labels, rotation=90, fontsize=6)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Per-Class Accuracy", fontsize=14, fontweight='bold')
ax.axhline(y=0.9, color='#27ae60', linestyle='--', alpha=0.5, label='90% threshold')
ax.axhline(y=0.8, color='#e74c3c', linestyle='--', alpha=0.5, label='80% threshold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "per_class_accuracy.png"), dpi=150)
plt.close()
print("[INFO] Per-class accuracy chart saved.")

print("\n[INFO] Evaluation complete. All artifacts saved to:", SAVE_DIR)
