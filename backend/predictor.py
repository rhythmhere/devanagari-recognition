"""
Advanced Predictor Module for Devanagari Character Recognition.
Features:
  - Multi-stage preprocessing with intelligent inversion
  - Grad-CAM heatmap generation for model interpretability
  - Base64 preview generation for all pipeline stages
  - Batch prediction support
  - Confidence-aware response with confusion analysis
"""

import os
import io
import json
import base64
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

from character_metadata import get_metadata, get_confusion_chars

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "devanagari_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "saved_model", "labels.json")

# ── Load model and labels ─────────────────────────────────────────
print("[INFO] Loading Devanagari model...")
model = load_model(MODEL_PATH)
print(f"[INFO] Model loaded successfully. Parameters: {model.count_params():,}")

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    idx_to_label = json.load(f)

NUM_CLASSES = len(idx_to_label)


# ── Utility: PIL image -> base64 data URI ──────────────────────────
def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL image to a base64-encoded data URI string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


# ── Grad-CAM Implementation ──────────────────────────────────────
def generate_gradcam(input_array: np.ndarray, pred_index: int) -> Image.Image:
    """
    Generate a Grad-CAM heatmap for the predicted class.
    Returns a PIL Image (128x128) with the heatmap overlay.
    """
    try:
        # Find the last convolutional layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            return None

        # Build gradient model
        grad_model = Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, model.output]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_array)
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None

        # Pool gradients over spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Resize heatmap to 128x128
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize((128, 128), Image.BILINEAR)

        # Apply colormap
        heatmap_array = np.array(heatmap_img)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.cm as cm
        colored = cm.jet(heatmap_array / 255.0)[:, :, :3]
        colored = (colored * 255).astype(np.uint8)

        # Get the original input as grayscale and overlay
        input_img = (input_array[0, :, :, 0] * 255).astype(np.uint8)
        input_pil = Image.fromarray(input_img).resize((128, 128), Image.NEAREST)
        input_rgb = input_pil.convert("RGB")

        overlay = Image.fromarray(colored)
        blended = Image.blend(input_rgb, overlay, alpha=0.5)

        return blended

    except Exception as e:
        print(f"[WARN] Grad-CAM failed: {e}")
        return None


# ── Preprocessing pipeline (returns all stages) ──────────────────
def preprocess_image(image_bytes, source_type: str = "upload") -> dict:
    """
    Preprocessing pipeline matching training data format.

    Training format: 32x32 grayscale, dark background (~0),
    lighter character strokes, rescaled to [0, 1].

    Steps:
        1. Open and composite onto white background
        2. Convert to grayscale
        3. Detect background brightness and invert if needed
        4. Noise threshold cleanup
        5. Optional morphological enhancement
        6. Crop to content bounding box with padding
        7. Centre in square
        8. Resize to 32x32
        9. Normalise to [0, 1]

    Returns:
        dict with 'input_array' (1,32,32,1) and 'previews' dict
    """
    # 1. Open as RGBA, composite onto white
    original = Image.open(image_bytes).convert("RGBA")
    white_bg = Image.new("RGBA", original.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(white_bg, original)
    original_rgb = composited.convert("RGB")

    original_thumb = original_rgb.copy()
    original_thumb.thumbnail((128, 128), Image.LANCZOS)

    # 2. Grayscale
    gray = original_rgb.convert("L")
    gray_thumb = gray.copy()
    gray_thumb.thumbnail((128, 128), Image.LANCZOS)

    # 3. Background brightness detection
    arr_gray = np.array(gray)
    h, w = arr_gray.shape
    bw = max(5, w // 12)
    bh = max(5, h // 12)
    border_pixels = np.concatenate([
        arr_gray[:bh, :].flatten(),
        arr_gray[-bh:, :].flatten(),
        arr_gray[:, :bw].flatten(),
        arr_gray[:, -bw:].flatten(),
    ])
    border_median = float(np.median(border_pixels))

    if source_type == "camera":
        # Border-band alone fails when the desk/hand occupies the frame edges
        # and makes the border look dark even though the paper is white.
        # The 75th percentile of the full image is a robust tiebreaker: for a
        # white-paper photo most pixels ARE background, so p75 will be bright.
        overall_p75 = float(np.percentile(arr_gray.flatten(), 75))
        bg_brightness = max(border_median, overall_p75)
    else:
        bg_brightness = border_median

    if bg_brightness > 128:
        arr_inv = np.array(ImageOps.invert(gray))
    else:
        arr_inv = arr_gray.copy()

    # 4. Threshold + dilation
    if source_type == "camera":
        # CLAHE normalises local contrast so ink strokes stand out against
        # uneven phone-camera illumination before Otsu sees the image.
        # Only used for computing the threshold — model input still uses the
        # unenhanced arr_inv values, matching the training distribution.
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        arr_clahe = clahe.apply(arr_inv)
        # Stronger blur reduces paper grain / texture before Otsu
        arr_for_otsu = cv2.GaussianBlur(arr_clahe, (5, 5), 0)
    else:
        arr_for_otsu = arr_inv

    _, otsu_mask = cv2.threshold(arr_for_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if source_type == "camera":
        # Remove ruled notebook lines: morphological opening with a wide
        # horizontal kernel detects only strokes spanning > 1/3 of the image
        # width — character strokes never reach that length.
        line_len = max(w // 3, 20)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
        h_lines = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, horiz_kernel)
        otsu_mask = cv2.bitwise_and(otsu_mask, cv2.bitwise_not(h_lines))

    # Apply mask to the original (unblurred) values so the model sees sharp strokes
    arr_proc = (arr_inv * (otsu_mask > 0)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    arr_proc = cv2.dilate(arr_proc, kernel, iterations=1)

    if source_type == "camera":
        # Adaptive area filter: keep components >= 2% of the largest component.
        # This is more robust than the old fixed (h+w)//20 formula which was
        # too aggressive for small characters photographed from a distance.
        bin_clean = (arr_proc > 0).astype(np.uint8) * 255
        n_comp, labels_map, stats, _ = cv2.connectedComponentsWithStats(
            bin_clean, connectivity=8)
        if n_comp > 1:
            comp_areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n_comp)]
            max_comp_area = max(comp_areas)
            min_area = max(8, int(max_comp_area * 0.02))
            clean_mask = np.zeros_like(bin_clean)
            for i in range(1, n_comp):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    clean_mask[labels_map == i] = 255
            arr_proc = (arr_proc * (clean_mask > 0)).astype(np.uint8)

    processed = Image.fromarray(arr_proc)

    processed_thumb = processed.copy()
    processed_thumb.thumbnail((128, 128), Image.LANCZOS)

    # 5. Crop to content bounding box + centre in square
    bbox = processed.getbbox()
    if bbox:
        cropped = processed.crop(bbox)
        cw, ch = cropped.size
        pad = max(4, int(max(cw, ch) * 0.1))
        max_dim = max(cw, ch) + 2 * pad
        centered = Image.new("L", (max_dim, max_dim), 0)
        paste_x = (max_dim - cw) // 2
        paste_y = (max_dim - ch) // 2
        centered.paste(cropped, (paste_x, paste_y))
    else:
        centered = processed

    # 6. Resize to 32x32
    resized = centered.resize((32, 32), Image.LANCZOS)

    # 6b. Camera: contrast-stretch so brightest stroke pixel reaches 255.
    # LANCZOS downsampling blurs strokes to intermediate values; this
    # restores the high-contrast appearance that matches the training data.
    if source_type == "camera":
        arr_r = np.array(resized)
        vmax = int(arr_r.max())
        if vmax > 20:
            arr_r = np.clip(arr_r.astype(np.float32) / vmax * 255.0, 0, 255).astype(np.uint8)
            resized = Image.fromarray(arr_r)

    # 7. Normalise
    arr = np.array(resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)

    # Build previews
    final_preview = resized.resize((128, 128), Image.NEAREST)

    previews = {
        "original": pil_to_base64(original_thumb),
        "grayscale": pil_to_base64(gray_thumb),
        "processed": pil_to_base64(processed_thumb),
        "model_input": pil_to_base64(final_preview),
    }

    return {
        "input_array": arr,
        "previews": previews,
        "bg_brightness": bg_brightness,
        "was_inverted": bg_brightness > 128,
        "source_type": source_type,
    }


# ── Confidence level helper ───────────────────────────────────────
def confidence_level(conf: float) -> str:
    if conf >= 85:
        return "high"
    elif conf >= 50:
        return "medium"
    else:
        return "low"


# ── Main prediction function ──────────────────────────────────────
def predict_character(image_source, include_gradcam: bool = True, source_type: str = "upload") -> dict:
    """
    Run the full prediction pipeline.

    Returns a rich JSON-serializable dict with:
      - prediction (char, roman, confidence, metadata)
      - top3 alternatives
      - top5 alternatives
      - confusion hints
      - preprocessing previews
      - Grad-CAM heatmap (optional)
      - preprocessing metadata
    """
    try:
        # Preprocess
        prep = preprocess_image(image_source, source_type=source_type)
        x = prep["input_array"]
        previews = prep["previews"]

        # Predict
        probs = model.predict(x, verbose=0)[0]

        # Top prediction
        pred_index = int(np.argmax(probs))
        raw_label = idx_to_label[str(pred_index)]
        conf = round(float(probs[pred_index] * 100), 2)
        meta = get_metadata(raw_label)

        # Top-3
        top3_idx = probs.argsort()[-3:][::-1]
        top3 = []
        for i in top3_idx:
            rl = idx_to_label[str(int(i))]
            m = get_metadata(rl)
            top3.append({
                "raw_label": rl,
                "char": m["char"],
                "roman": m["roman"],
                "nepali_name": m["nepali_name"],
                "type": m["type"],
                "confidence": round(float(probs[i] * 100), 2),
            })

        # Top-5
        top5_idx = probs.argsort()[-5:][::-1]
        top5 = []
        for i in top5_idx:
            rl = idx_to_label[str(int(i))]
            m = get_metadata(rl)
            top5.append({
                "raw_label": rl,
                "char": m["char"],
                "roman": m["roman"],
                "type": m["type"],
                "confidence": round(float(probs[i] * 100), 2),
            })

        # Confusion hints
        confusions = get_confusion_chars(raw_label)

        # Grad-CAM
        gradcam_b64 = None
        if include_gradcam:
            gradcam_img = generate_gradcam(x, pred_index)
            if gradcam_img is not None:
                gradcam_b64 = pil_to_base64(gradcam_img)

        # Entropy of prediction distribution (uncertainty measure)
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        max_entropy = float(np.log(NUM_CLASSES))
        normalised_entropy = round(entropy / max_entropy, 4)

        # Build response
        return {
            "success": True,
            "prediction": {
                "raw_label": raw_label,
                "char": meta["char"],
                "roman": meta["roman"],
                "nepali_name": meta["nepali_name"],
                "type": meta["type"],
                "confidence": conf,
                "confidence_level": confidence_level(conf),
                "example_word": meta["example_word"],
                "note": meta["note"],
            },
            "top3": top3,
            "top5": top5,
            "confusions": confusions,
            "preprocessing": previews,
            "gradcam": gradcam_b64,
            "analysis": {
                "entropy": round(entropy, 4),
                "normalised_entropy": normalised_entropy,
                "uncertainty": "low" if normalised_entropy < 0.1 else "medium" if normalised_entropy < 0.3 else "high",
                "bg_brightness": prep["bg_brightness"],
                "was_inverted": prep["was_inverted"],
                "model_params": model.count_params(),
                "num_classes": NUM_CLASSES,
            },
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
