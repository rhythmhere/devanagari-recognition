"""
Multi-Digit Recognition Evaluation Script.

Evaluates the segmentation-based multi-digit pipeline on the
synthetic test set produced by generate_multidigit_dataset.py.

Metrics computed:
  - Full-sequence accuracy   – entire numeral string correct
  - Digit-level accuracy     – each digit position independently correct
  - Segmentation success rate – correct number of digit segments found

Outputs:
  backend/saved_model/multidigit_evaluation_summary.json
  backend/saved_model/multidigit_confusion_matrix.png

Usage (from project root):
  python ml/evaluate_multidigit.py
"""

import sys
import os
import csv
import json
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup ─────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

DATASET_DIR = PROJECT_ROOT / "data" / "multidigit_dataset"
LABELS_CSV = DATASET_DIR / "labels.csv"
OUTPUT_DIR = BACKEND_DIR / "saved_model"


# ── Data loading ───────────────────────────────────────────────

def load_test_entries() -> list:
    if not LABELS_CSV.exists():
        print(f"[ERROR] Labels CSV not found: {LABELS_CSV}")
        print("  Run  ml/generate_multidigit_dataset.py  first.")
        sys.exit(1)

    entries = []
    with open(LABELS_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "test":
                entries.append(row)

    if not entries:
        print("[ERROR] No test-split entries found in labels.csv.")
        sys.exit(1)

    return entries


# ── Evaluation ─────────────────────────────────────────────────

def evaluate():
    from multidigit_predictor import predict_multidigit

    entries = load_test_entries()
    print(f"[INFO] Evaluating {len(entries)} test images…\n")

    total = 0
    full_seq_correct = 0
    seg_success = 0
    digit_total = 0
    digit_correct = 0
    errors = []

    # Per-number breakdown
    per_class: dict = defaultdict(lambda: {
        "total": 0, "full_seq_correct": 0,
        "digit_total": 0, "digit_correct": 0,
    })

    # Confusion: (true_arabic_digit, pred_arabic_digit) → count
    confusion: dict = defaultdict(int)

    for i, entry in enumerate(entries):
        # entry["filename"] is like "test/10/10_0042.png"
        img_path = DATASET_DIR / entry["filename"]
        if not img_path.exists():
            errors.append(f"Missing file: {img_path}")
            continue

        gt_arabic: str = entry["label_arabic"]     # e.g. "125"
        gt_deva: str = entry["label_devanagari"]   # e.g. "१२५"
        gt_n_digits: int = int(entry["digit_count"])

        try:
            with open(img_path, "rb") as fh:
                result = predict_multidigit(BytesIO(fh.read()))
        except Exception as exc:
            errors.append(f"Exception on {img_path}: {exc}")
            continue

        total += 1
        pc = per_class[gt_arabic]
        pc["total"] += 1

        if not result.get("success"):
            # Complete pipeline failure (e.g. 0 segments)
            continue

        pred_deva: str = result.get("full_number", "")
        pred_count: int = result.get("digit_count", 0)

        # Full-sequence accuracy
        if pred_deva == gt_deva:
            full_seq_correct += 1
            pc["full_seq_correct"] += 1

        # Segmentation success
        if pred_count == gt_n_digits:
            seg_success += 1

        # Digit-level accuracy
        segments = result.get("segments", [])
        gt_digits = list(gt_arabic)           # ["1", "2", "5"]
        pred_digits = [s["roman"] for s in segments]
        compare_len = min(len(gt_digits), len(pred_digits), gt_n_digits)

        for pos in range(compare_len):
            digit_total += 1
            pc["digit_total"] += 1
            td = gt_digits[pos] if pos < len(gt_digits) else "?"
            pd = pred_digits[pos] if pos < len(pred_digits) else "?"
            if td == pd:
                digit_correct += 1
                pc["digit_correct"] += 1
            confusion[(td, pd)] += 1

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(entries)}…")

    if total == 0:
        print("[ERROR] No images were successfully processed.")
        sys.exit(1)

    # ── Print results ──────────────────────────────────────────
    fsa = full_seq_correct / total
    ssr = seg_success / total
    dla = digit_correct / digit_total if digit_total > 0 else 0.0

    print("\n" + "=" * 52)
    print("  MULTI-DIGIT RECOGNITION — EVALUATION SUMMARY")
    print("=" * 52)
    print(f"  Test images processed     : {total}")
    print(f"  Full-sequence accuracy    : {fsa * 100:>6.2f}%")
    print(f"  Digit-level accuracy      : {dla * 100:>6.2f}%")
    print(f"  Segmentation success rate : {ssr * 100:>6.2f}%")
    print(f"  Digits evaluated          : {digit_total}")
    if errors:
        print(f"  Errors / skipped          : {len(errors)}")
    print("=" * 52)

    print("\n  Per-number breakdown:")
    print(f"  {'Number':<8}  {'Total':>6}  {'Full-seq%':>10}  {'Digit%':>8}")
    print("  " + "-" * 38)
    for num in sorted(per_class.keys(), key=lambda x: int(x)):
        r = per_class[num]
        if r["total"] == 0:
            continue
        fsp = r["full_seq_correct"] / r["total"] * 100
        dp = (r["digit_correct"] / r["digit_total"] * 100
              if r["digit_total"] > 0 else 0)
        print(f"  {num:<8}  {r['total']:>6}  {fsp:>9.1f}%  {dp:>7.1f}%")

    # ── Save JSON summary ──────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_class_out = {}
    for num, r in per_class.items():
        if r["total"] == 0:
            continue
        per_class_out[num] = {
            "total": r["total"],
            "full_seq_accuracy": round(r["full_seq_correct"] / r["total"], 4),
            "digit_accuracy": (
                round(r["digit_correct"] / r["digit_total"], 4)
                if r["digit_total"] > 0 else 0.0
            ),
        }

    summary = {
        "total_test_images": total,
        "full_sequence_accuracy": round(fsa, 4),
        "digit_level_accuracy": round(dla, 4),
        "segmentation_success_rate": round(ssr, 4),
        "total_digits_evaluated": digit_total,
        "per_class_results": per_class_out,
        "errors_count": len(errors),
    }

    json_path = OUTPUT_DIR / "multidigit_evaluation_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] Evaluation summary  → {json_path}")

    # ── Confusion matrix (digit level, 0-9) ───────────────────
    digit_labels_arabic = [str(i) for i in range(10)]
    cm = np.zeros((10, 10), dtype=int)
    for (td, pd), cnt in confusion.items():
        if td in digit_labels_arabic and pd in digit_labels_arabic:
            cm[int(td)][int(pd)] += cnt

    deva_labels = ["०", "१", "२", "३", "४", "५", "६", "७", "८", "९"]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(deva_labels, fontsize=11)
    ax.set_yticklabels(deva_labels, fontsize=11)
    ax.set_xlabel("Predicted Digit", fontsize=11)
    ax.set_ylabel("True Digit", fontsize=11)
    ax.set_title("Multi-Digit Extension — Digit-Level Confusion Matrix",
                 fontsize=12, pad=12)

    max_val = cm.max() if cm.max() > 0 else 1
    for r in range(10):
        for c in range(10):
            if cm[r][c] > 0:
                ax.text(c, r, str(cm[r][c]),
                        ha="center", va="center", fontsize=8,
                        color="white" if cm[r][c] > max_val * 0.55
                        else "black")

    plt.tight_layout()
    cm_path = OUTPUT_DIR / "multidigit_confusion_matrix.png"
    plt.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Confusion matrix    → {cm_path}")

    return summary


if __name__ == "__main__":
    evaluate()
