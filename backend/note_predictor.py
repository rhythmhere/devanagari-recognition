"""
Note Denomination Predictor Module.
Secondary advanced module of the Devanagari Character Recognition project.

Features:
  - RGB image preprocessing for currency notes (EXIF-safe, BICUBIC resize)
  - MobileNetV2-based denomination classification (when model is trained)
  - Color-histogram fallback classification (when model is not trained)
  - Three-tier confidence: high (>=65%), medium (>=30%), low (>=10%)
  - Rejection only when confidence < 10% (true garbage prediction)
  - Top-3 predictions always included for transparency
  - Rich metadata integration for voice output
  - Label-to-metadata key mapping (handles different folder naming schemes)
"""

import os
import io
import json
import base64
import numpy as np
from PIL import Image, ImageOps

from note_metadata import get_note_metadata, get_note_voice_text

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
NOTE_MODEL_PATH  = os.path.join(BASE_DIR, "saved_model", "note_model.keras")
NOTE_LABELS_PATH = os.path.join(BASE_DIR, "saved_model", "note_labels.json")

# ── Tiered confidence thresholds ──────────────────────────────────
# With 7 classes, random chance = 14.3 %.
# TIER_HIGH    – model is confident → show denomination, green UI
# TIER_MEDIUM  – model is somewhat sure → show "Likely ₨X" with warning
# TIER_LOW     – weak signal → show "Possibly ₨X — please verify"
# Below TIER_LOW → reject ("Not Recognized")
TIER_HIGH   = 65.0
TIER_MEDIUM = 30.0
TIER_LOW    = 10.0   # rejection boundary

# Kept for backward compatibility
REJECTION_THRESHOLD_DL       = TIER_LOW
REJECTION_THRESHOLD_FALLBACK = TIER_LOW

# ── Mapping from training folder names → metadata keys ────────────
TRAINING_LABEL_TO_META_KEY = {
    "five":         "rs5",
    "ten":          "rs10",
    "twenty":       "rs20",
    "fifty":        "rs50",
    "hundred":      "rs100",
    "fivehundred":  "rs500",
    "thousand":     "rs1000",
    "rs5":          "rs5",
    "rs10":         "rs10",
    "rs20":         "rs20",
    "rs50":         "rs50",
    "rs100":        "rs100",
    "rs500":        "rs500",
    "rs1000":       "rs1000",
}


def resolve_label(raw_label: str) -> str:
    """Map any raw label from note_labels.json to the canonical metadata key."""
    key = raw_label.strip().lower()
    return TRAINING_LABEL_TO_META_KEY.get(key, raw_label)


# ── Load model and labels ─────────────────────────────────────────
note_model        = None
note_idx_to_label = None
NOTE_MODEL_AVAILABLE = False

try:
    if os.path.exists(NOTE_MODEL_PATH) and os.path.exists(NOTE_LABELS_PATH):
        from tensorflow.keras.models import load_model
        print("[INFO] Loading note denomination model...")
        note_model = load_model(NOTE_MODEL_PATH)
        print(f"[INFO] Note model loaded. Parameters: {note_model.count_params():,}")

        with open(NOTE_LABELS_PATH, "r", encoding="utf-8") as f:
            note_idx_to_label = json.load(f)

        NOTE_MODEL_AVAILABLE = True
        print(f"[INFO] Note model ready with {len(note_idx_to_label)} classes.")
    else:
        print("[WARN] Note model or labels not found. Using color fallback classifier.")
        if not os.path.exists(NOTE_MODEL_PATH):
            print(f"[WARN]  Missing model: {NOTE_MODEL_PATH}")
        if not os.path.exists(NOTE_LABELS_PATH):
            print(f"[WARN]  Missing labels: {NOTE_LABELS_PATH}")

        if os.path.exists(NOTE_LABELS_PATH):
            with open(NOTE_LABELS_PATH, "r", encoding="utf-8") as f:
                note_idx_to_label = json.load(f)
            print(f"[INFO] Note labels loaded ({len(note_idx_to_label)} classes) for fallback mode.")
except Exception as e:
    print(f"[WARN] Could not load note model: {e}")
    print("[WARN] Using color fallback classifier.")


# ── Utilities ─────────────────────────────────────────────────────
def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def note_confidence_level(conf: float) -> str:
    if conf >= 80:
        return "high"
    elif conf >= 50:
        return "medium"
    else:
        return "low"


def note_confidence_tier(conf: float) -> str:
    """
    Map raw confidence % to a display tier.
      high     – conf >= 65 %  → show denomination normally
      medium   – conf >= 30 %  → show "Likely ₨X (medium confidence)"
      low      – conf >= 10 %  → show "Possibly ₨X (low confidence)"
      rejected – conf <  10 %  → show "Not Recognized"
    """
    if conf >= TIER_HIGH:
        return "high"
    if conf >= TIER_MEDIUM:
        return "medium"
    if conf >= TIER_LOW:
        return "low"
    return "rejected"


# ── Preprocessing ─────────────────────────────────────────────────
def preprocess_note_image(image_bytes) -> dict:
    """
    Load and preprocess a currency-note image for the MobileNetV2 model.

    Key decisions:
      - ImageOps.exif_transpose fixes phone photos stored sideways.
      - BICUBIC resize matches the Keras ImageDataGenerator default used
        during training (avoids a train/inference downsampling mismatch).
      - Normalisation to [0, 1] matches the training rescale=1/255.
    """
    raw = Image.open(image_bytes)
    # Honour EXIF orientation so portrait/landscape notes are upright
    original = ImageOps.exif_transpose(raw).convert("RGB")

    original_thumb = original.copy()
    original_thumb.thumbnail((256, 256), Image.LANCZOS)

    # BICUBIC matches Keras ImageDataGenerator default interpolation
    resized = original.resize((128, 128), Image.BICUBIC)
    arr = np.array(resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    resized_preview = resized.resize((256, 256), Image.NEAREST)

    return {
        "input_array":    arr,
        "original_image": original,
        "previews": {
            "original": pil_to_base64(original_thumb),
            "resized":  pil_to_base64(resized_preview),
        },
    }


# ══════════════════════════════════════════════════════════════════
#  FALLBACK CLASSIFIER  (color-histogram based)
# ══════════════════════════════════════════════════════════════════

# Hue ranges for Nepali notes (approximate dominant hue in degrees, 0-360)
NOTE_COLOR_PROFILES = {
    "rs5":    {"hue_range": (0,   40),   "sat_min": 15},
    "rs10":   {"hue_range": (10,  50),   "sat_min": 12},
    "rs20":   {"hue_range": (20,  60),   "sat_min": 15},
    "rs50":   {"hue_range": (195, 285),  "sat_min": 12},   # blue-purple
    "rs100":  {"hue_range": (75,  175),  "sat_min": 12},   # green
    "rs500":  {"hue_range": (335, 395),  "sat_min": 15},   # wraps around red
    "rs1000": {"hue_range": (175, 255),  "sat_min": 8},    # grey-blue
}


def fallback_classify(original_image: Image.Image) -> list:
    """
    Simple color-histogram fallback.
    Returns [(label, confidence_pct), ...] sorted by confidence descending.
    """
    img = original_image.resize((64, 64), Image.LANCZOS)
    arr = np.array(img).astype("float32")

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    avg_r = float(np.mean(r))
    avg_g = float(np.mean(g))
    avg_b = float(np.mean(b))

    max_c = max(avg_r, avg_g, avg_b)
    min_c = min(avg_r, avg_g, avg_b)
    diff  = max_c - min_c

    if diff < 8:
        hue = 0
    elif max_c == avg_r:
        hue = 60.0 * (((avg_g - avg_b) / diff) % 6)
    elif max_c == avg_g:
        hue = 60.0 * (((avg_b - avg_r) / diff) + 2)
    else:
        hue = 60.0 * (((avg_r - avg_g) / diff) + 4)

    if hue < 0:
        hue += 360.0

    saturation = (diff / max_c * 100.0) if max_c > 0 else 0.0

    scores = {}
    for label, profile in NOTE_COLOR_PROFILES.items():
        h_low, h_high = profile["hue_range"]
        if h_high > 360:
            in_range = (hue >= h_low) or (hue <= (h_high - 360))
        else:
            in_range = h_low <= hue <= h_high

        if in_range and saturation >= profile["sat_min"]:
            scores[label] = 35.0 + min(saturation, 65.0) * 0.4
        else:
            if h_high > 360:
                dist = min(abs(hue - h_low), abs(hue - (h_high - 360)))
            else:
                dist = min(abs(hue - h_low), abs(hue - h_high))
            scores[label] = max(4.0, 28.0 - dist * 0.22)

    total   = sum(scores.values())
    results = [
        (lbl, round(score / total * 100.0, 2))
        for lbl, score in scores.items()
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ── Main prediction function ──────────────────────────────────────
def predict_note(image_source) -> dict:
    """
    Run the full note denomination prediction pipeline.

    Returns a dict with:
      success           – bool
      recognized        – bool  (False only when conf < TIER_LOW = 10 %)
      confidence_tier   – "high" | "medium" | "low" | "rejected"
      prediction        – denomination details (always the model's top-1 guess;
                          denomination = 'unknown' only when truly rejected)
      top3              – top-3 candidates (always included)
      preprocessing, voice, method, note_model_trained
    """
    try:
        prep           = preprocess_note_image(image_source)
        x              = prep["input_array"]
        previews       = prep["previews"]
        original_image = prep["original_image"]

        if NOTE_MODEL_AVAILABLE and note_model is not None:
            # ── Trained DL model ──────────────────────────────────
            probs      = note_model.predict(x, verbose=0)[0]
            pred_index = int(np.argmax(probs))
            raw_label  = note_idx_to_label[str(pred_index)]
            meta_key   = resolve_label(raw_label)
            conf       = round(float(probs[pred_index] * 100.0), 2)

            top3_idx = probs.argsort()[-3:][::-1]
            top3 = []
            for i in top3_idx:
                rl = note_idx_to_label[str(int(i))]
                mk = resolve_label(rl)
                m  = get_note_metadata(mk)
                top3.append({
                    "denomination": mk,
                    "value":        m["value"],
                    "english_name": m["english_name"],
                    "nepali_name":  m["nepali_name"],
                    "confidence":   round(float(probs[i] * 100.0), 2),
                })

            method = "deep_learning"

            # Secondary color check (for logging / future ensemble use)
            fallback = fallback_classify(original_image)
            fb_label = fallback[0][0]
            fb_conf  = fallback[0][1]
            if fb_label == meta_key:
                print(f"[INFO] Note: DL={meta_key}({conf}%) COLOR={fb_label}({fb_conf}%) — agreement")
            else:
                print(f"[INFO] Note: DL={meta_key}({conf}%) COLOR={fb_label}({fb_conf}%) — disagreement")

        else:
            # ── Color-histogram fallback ──────────────────────────
            results  = fallback_classify(original_image)
            meta_key = results[0][0]
            conf     = results[0][1]

            top3 = []
            for label, c in results[:3]:
                m = get_note_metadata(label)
                top3.append({
                    "denomination": label,
                    "value":        m["value"],
                    "english_name": m["english_name"],
                    "nepali_name":  m["nepali_name"],
                    "confidence":   c,
                })

            method = "color_fallback"

        # ── Confidence tier & recognition gate ───────────────────
        tier       = note_confidence_tier(conf)
        recognized = (tier != "rejected")

        # Keep the actual prediction for transparency; only fall back to
        # "unknown" when we truly have nothing useful to show.
        display_key = meta_key if recognized else "unknown"
        meta        = get_note_metadata(display_key)

        print(f"[INFO] Note result: tier={tier} conf={conf}% denomination={display_key} recognized={recognized}")

        # ── Voice data ────────────────────────────────────────────
        voice_key  = meta_key if recognized else "unknown"
        voice_data = {
            "english":    get_note_voice_text(voice_key, "english",    conf),
            "nepali":     get_note_voice_text(voice_key, "nepali",     conf),
            "mixed":      get_note_voice_text(voice_key, "mixed",      conf),
            "confidence": get_note_voice_text(voice_key, "confidence", conf),
        }

        return {
            "success":          True,
            "recognized":       recognized,
            "confidence_tier":  tier,
            "prediction": {
                "denomination":        display_key,
                "actual_denomination": meta_key,     # always the model's raw top-1
                "value":               meta["value"],
                "english_name":        meta["english_name"],
                "nepali_name":         meta["nepali_name"],
                "nepali_numeral":      meta["nepali_numeral"],
                "color_hint":          meta["color_hint"],
                "confidence":          conf,
                "confidence_level":    note_confidence_level(conf),
                "confidence_tier":     tier,
            },
            "top3":               top3,
            "preprocessing":      previews,
            "voice":              voice_data,
            "method":             method,
            "note_model_trained": NOTE_MODEL_AVAILABLE,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
