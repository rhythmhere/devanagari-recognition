"""
Synthetic Multi-Digit Devanagari Dataset Generator.

Creates composite images by combining single-digit images from
  data/train/digit_0  through  data/train/digit_9.

Each composite places two or three digit images side-by-side with
random spacing, slight vertical jitter, and mild noise so the
pipeline is tested on realistic variation without exposing the
test set to training images (80/20 source-level split).

Output structure:
  data/multidigit_dataset/
    train/
      10/  11/  12/  ...  999/
    test/
      10/  11/  12/  ...  999/
    labels.csv        (filename, label_arabic, label_devanagari,
                       digit_count, split)

Usage (from project root):
  python ml/generate_multidigit_dataset.py
Or from the ml/ directory:
  python generate_multidigit_dataset.py
"""

import os
import sys
import random
import csv
import numpy as np
from pathlib import Path
from PIL import Image

# ── Resolve directories ────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DIGIT_TRAIN_DIR = DATA_DIR / "train"
OUTPUT_DIR = DATA_DIR / "multidigit_dataset"
TRAIN_OUT = OUTPUT_DIR / "train"
TEST_OUT = OUTPUT_DIR / "test"

# ── Generation config ──────────────────────────────────────────
RANDOM_SEED = 42
TRAIN_POOL_RATIO = 0.80          # fraction of source images used for train composites
SAMPLES_TRAIN = 200              # composites per class in train split
SAMPLES_TEST = 50                # composites per class in test split
DIGIT_HEIGHT = 52                # each digit is scaled to this height
GAP_MIN = 4                      # min horizontal gap between digits (px)
GAP_MAX = 16                     # max horizontal gap
PADDING = 6                      # canvas border padding
JITTER_Y = 4                     # max vertical jitter (px)
NOISE_STD = 4.0                  # gaussian noise std applied to final canvas

# Numbers to generate (2-digit and 3-digit)
TARGET_NUMBERS = [
    # common 2-digit numbers
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 25, 30, 50, 51, 75, 99,
    # selected 3-digit numbers
    100, 111, 123, 125, 200, 500, 999,
]

DEVANAGARI = {
    0: "०", 1: "१", 2: "२", 3: "३", 4: "४",
    5: "५", 6: "६", 7: "७", 8: "८", 9: "९",
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Helper functions ───────────────────────────────────────────

def number_to_devanagari(n: int) -> str:
    return "".join(DEVANAGARI[int(d)] for d in str(n))


def load_image_paths(digit: int) -> list:
    folder = DIGIT_TRAIN_DIR / f"digit_{digit}"
    if not folder.exists():
        return []
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        paths.extend(str(p) for p in folder.glob(ext))
    return paths


def preprocess_single_digit(path: str) -> Image.Image:
    """
    Load one digit image: composite on white, convert to grayscale,
    invert if needed, threshold, and crop to content bounding box.
    Returns a PIL 'L' image with dark strokes on white background.
    """
    img = Image.open(path).convert("RGBA")
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white, img).convert("L")

    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape
    # corner-based background detection (matches predictor.py)
    corners = np.concatenate([
        arr[0:min(4, h), 0:min(4, w)].flatten(),
        arr[0:min(4, h), max(0, w - 4):w].flatten(),
        arr[max(0, h - 4):h, 0:min(4, w)].flatten(),
        arr[max(0, h - 4):h, max(0, w - 4):w].flatten(),
    ])
    if float(np.median(corners)) > 128:
        arr = 255 - arr           # invert: strokes become bright
    arr = np.where(arr > 30, arr, 0).astype(np.uint8)

    img_out = Image.fromarray(arr)
    bbox = img_out.getbbox()
    if bbox:
        img_out = img_out.crop(bbox)
    return img_out


def compose_row(digit_images: list) -> Image.Image:
    """
    Place scaled digit images side-by-side on a white canvas with
    random gap and vertical jitter.  Returns a PIL 'L' image.
    """
    gap = random.randint(GAP_MIN, GAP_MAX)
    canvas_h = DIGIT_HEIGHT + 2 * PADDING + 2 * JITTER_Y

    # Scale each digit to DIGIT_HEIGHT, preserving aspect ratio
    scaled = []
    for img in digit_images:
        ow, oh = img.size
        if oh == 0:
            continue
        nw = max(1, int(ow * DIGIT_HEIGHT / oh))
        scaled.append(img.resize((nw, DIGIT_HEIGHT), Image.LANCZOS))
    if not scaled:
        return None

    total_w = (sum(s.width for s in scaled)
               + gap * (len(scaled) - 1)
               + 2 * PADDING)
    canvas = Image.new("L", (total_w, canvas_h), 255)

    x = PADDING
    for s in scaled:
        jitter = random.randint(-JITTER_Y, JITTER_Y)
        y = PADDING + jitter
        y = max(0, min(canvas_h - s.height, y))
        canvas.paste(s, (x, y))
        x += s.width + gap

    # Mild gaussian noise for realism
    arr = np.array(canvas, dtype=np.float32)
    arr += np.random.normal(0, NOISE_STD, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ── Main generation routine ────────────────────────────────────

def generate():
    print("=" * 55)
    print(" Synthetic Multi-Digit Dataset Generator")
    print("=" * 55)

    # Load and split image pools per digit
    pools: dict = {}
    for d in range(10):
        paths = load_image_paths(d)
        if not paths:
            print(f"  [WARN] No images for digit_{d} – skipping.")
            continue
        random.shuffle(paths)
        split_idx = max(1, int(len(paths) * TRAIN_POOL_RATIO))
        pools[d] = {
            "train": paths[:split_idx],
            "test": paths[split_idx:] or paths[:1],  # ensure at least 1
        }
        print(f"  digit_{d}: {len(paths)} images "
              f"(train pool {split_idx}, test pool {len(paths)-split_idx})")

    if not pools:
        print("[ERROR] No digit images found. "
              "Ensure data/train/digit_0 … digit_9 folders exist.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    TEST_OUT.mkdir(parents=True, exist_ok=True)

    csv_rows: list = []
    generated_counts: dict = {"train": 0, "test": 0}

    for number in TARGET_NUMBERS:
        digits = [int(d) for d in str(number)]
        if any(d not in pools for d in digits):
            print(f"  [SKIP] {number}: missing digit pool.")
            continue

        label_arabic = str(number)
        label_deva = number_to_devanagari(number)
        n_digits = len(digits)

        for split, n_samples in (("train", SAMPLES_TRAIN),
                                  ("test", SAMPLES_TEST)):
            out_dir = (TRAIN_OUT if split == "train" else TEST_OUT) / label_arabic
            out_dir.mkdir(parents=True, exist_ok=True)

            for idx in range(n_samples):
                # Sample one image per digit position from this split's pool
                digit_imgs = []
                ok = True
                for d in digits:
                    pool = pools[d][split]
                    try:
                        path = random.choice(pool)
                        digit_imgs.append(preprocess_single_digit(path))
                    except Exception as exc:
                        print(f"  [WARN] {path}: {exc}")
                        ok = False
                        break

                if not ok or not digit_imgs:
                    continue

                composed = compose_row(digit_imgs)
                if composed is None:
                    continue

                filename = f"{label_arabic}_{idx:04d}.png"
                composed.save(out_dir / filename)
                rel = f"{split}/{label_arabic}/{filename}"
                csv_rows.append([rel, label_arabic, label_deva, n_digits, split])
                generated_counts[split] += 1

        print(f"  Generated: {number:>4}  ({label_deva})  "
              f"→ {SAMPLES_TRAIN} train + {SAMPLES_TEST} test")

    # Write labels CSV (UTF-8 BOM for Excel compatibility)
    csv_path = OUTPUT_DIR / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label_arabic", "label_devanagari",
                    "digit_count", "split"])
        w.writerows(csv_rows)

    print("\n" + "=" * 55)
    print(f" Dataset saved to : {OUTPUT_DIR}")
    print(f" Train composites : {generated_counts['train']}")
    print(f" Test  composites : {generated_counts['test']}")
    print(f" Labels CSV       : {csv_path}")
    print(f" Total rows in CSV: {len(csv_rows)}")
    print("=" * 55)


if __name__ == "__main__":
    generate()
