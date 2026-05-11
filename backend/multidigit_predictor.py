"""
Multi-Digit Devanagari Numeral Recognition – v2.

Root-cause fixes vs v1:
  • Segmentation uses the raw Otsu binary (NO dilation).
    The v1 binary was built from the dilated masked image, so a 2x2
    dilation bridged narrow gaps between adjacent digit strokes,
    collapsing 4 components into 1-2 blobs.
  • merge_gap reduced 12 -> 4 px (only true intra-digit fragments merge).
  • Vertical projection-profile splitter added for oversized segments.
  • Warning messages no longer expose raw internal label names.
"""

import io
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw

from predictor import model, idx_to_label, pil_to_base64, confidence_level

# ── Digit label constants ──────────────────────────────────────
DIGIT_LABELS = {f"digit_{i}" for i in range(10)}

DEVANAGARI_DIGITS = {
    "digit_0": "०", "digit_1": "१", "digit_2": "२",
    "digit_3": "३", "digit_4": "४", "digit_5": "५",
    "digit_6": "६", "digit_7": "७", "digit_8": "८",
    "digit_9": "९",
}

ARABIC_DIGITS = {
    "digit_0": "0", "digit_1": "1", "digit_2": "2",
    "digit_3": "3", "digit_4": "4", "digit_5": "5",
    "digit_6": "6", "digit_7": "7", "digit_8": "8",
    "digit_9": "9",
}

MAX_SEGMENTS = 10


# ── Preprocessing ──────────────────────────────────────────────

def _preprocess_full_image(image_source, source_type: str = "canvas") -> dict:
    """
    Preprocess the full multi-digit image.

    Returns two binaries:
      binary_seg  – Otsu threshold only, NO dilation.
                    Use this for connected-component segmentation.
                    Dilation was the root cause of digit merging: a 2x2
                    ellipse kernel expands each stroke edge by ~1 px so
                    adjacent digit strokes within 2 px fuse into one blob.
      binary      – Dilated version, kept only for preview annotation.
      masked      – Dilated stroke values, used for per-digit model input.
    """
    original = Image.open(image_source).convert("RGBA")
    white_bg = Image.new("RGBA", original.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(white_bg, original).convert("RGB")
    gray_pil = composited.convert("L")
    arr = np.array(gray_pil, dtype=np.uint8)

    h, w = arr.shape
    bw = max(5, w // 12)
    bh = max(5, h // 12)
    border_pixels = np.concatenate([
        arr[:bh, :].flatten(), arr[-bh:, :].flatten(),
        arr[:, :bw].flatten(), arr[:, -bw:].flatten(),
    ])
    bg_brightness = float(np.median(border_pixels))
    was_inverted = bg_brightness > 128

    if was_inverted:
        arr = 255 - arr

    arr_for_otsu = cv2.GaussianBlur(arr, (3, 3), 0) if source_type == "camera" else arr
    _, otsu_mask = cv2.threshold(arr_for_otsu, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Segmentation binary: Otsu only — NO dilation.
    # This is the critical fix: without dilation the narrow pixel gaps
    # between adjacent handwritten digits stay open, so each digit
    # remains its own connected component.
    binary_seg = otsu_mask.copy()

    # Model-input masked: dilated for stroke quality (matches predictor.py)
    masked = (arr * (otsu_mask > 0)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    masked = cv2.dilate(masked, kernel, iterations=1)

    # Preview binary (dilated, used only for annotation overlay)
    _, binary = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY)

    return {
        "binary_seg": binary_seg,
        "binary":     binary,
        "masked":     masked,
        "original_pil": composited,
        "was_inverted": was_inverted,
        "bg_brightness": bg_brightness,
    }


# ── Segmentation helpers ───────────────────────────────────────

def _find_column_valleys(col_smooth: np.ndarray,
                          threshold: float,
                          min_run: int) -> list:
    """
    Return midpoints of consecutive 'valley' runs (columns below threshold)
    that are at least min_run columns wide.

    The min_run requirement prevents intra-digit narrow dips (e.g., the
    interior of ० or the waist of ३) from being mistaken for digit gaps.
    """
    valleys = []
    in_valley = False
    vs = 0
    n = len(col_smooth)
    for c in range(n):
        if col_smooth[c] < threshold and not in_valley:
            vs = c
            in_valley = True
        elif col_smooth[c] >= threshold and in_valley:
            if c - vs >= min_run:
                valleys.append((vs + c) // 2)
            in_valley = False
    if in_valley and n - vs >= min_run:
        valleys.append((vs + n) // 2)
    return valleys


def _split_by_projection(binary_seg: np.ndarray, box: list,
                          min_area: int = 50) -> list:
    """
    Attempt to split a bounding box that likely contains multiple merged
    digits by analysing the vertical (column) projection profile.

    Only returns multiple sub-boxes when a genuine low-ink valley of
    sufficient width is found.  Falls back to the original box otherwise
    so the caller never loses a valid digit.
    """
    x, y, bw, bh = box
    h_img, w_img = binary_seg.shape
    x2 = min(w_img, x + bw)
    y2 = min(h_img, y + bh)
    roi = binary_seg[y:y2, x:x2].astype(float)

    if roi.size == 0:
        return [box]

    col_proj = roi.sum(axis=0)
    max_val = col_proj.max()
    if max_val == 0:
        return [box]

    norm = col_proj / max_val

    # Gaussian smoothing proportional to segment width
    k = max(3, bw // 25)
    if k % 2 == 0:
        k += 1
    smooth = cv2.GaussianBlur(
        norm.reshape(1, -1).astype(np.float32),
        (k, 1), k / 3.0
    ).flatten()

    # A real inter-digit gap spans many columns; a within-digit dip is 1-3 cols.
    # min_run ~ h_img/14 ≈ 14 px on the 200 px canvas — catches gaps >= 14 px.
    min_run = max(6, h_img // 14)
    min_seg_w = max(8, bw // 8)   # sub-segment must be at least this wide

    valleys = _find_column_valleys(smooth, threshold=0.05, min_run=min_run)
    valleys = [v for v in valleys if min_seg_w < v < bw - min_seg_w]

    if not valleys:
        return [box]

    split_pts = [0] + valleys + [bw]
    sub_boxes = []
    for i in range(len(split_pts) - 1):
        sx1, sx2 = split_pts[i], split_pts[i + 1]
        sub_roi = roi[:, sx1:sx2]
        rows = np.any(sub_roi > 0, axis=1)
        if not rows.any():
            continue
        rmin = int(np.where(rows)[0][0])
        rmax = int(np.where(rows)[0][-1])
        sw = sx2 - sx1
        sh = rmax - rmin + 1
        if sw * sh < min_area:
            continue
        sub_boxes.append([x + sx1, y + rmin, sw, sh])

    return sub_boxes if len(sub_boxes) >= 2 else [box]


def _find_segments(binary_seg: np.ndarray,
                   min_area: int = 50,
                   merge_gap: int = 4) -> list:
    """
    Detect digit bounding boxes in the segmentation binary.

    Steps
    -----
    1. Connected-component analysis (8-connectivity) on the undilated binary.
    2. Area filter — drop tiny noise blobs.
    3. Sort left-to-right; merge boxes whose horizontal gap <= merge_gap px.
       (merge_gap=4 catches intra-digit stroke fragments without joining
        different digits — much safer than the old value of 12.)
    4. For any merged box wider than 1.2x the image height (a heuristic for
       "this box is probably 2+ merged digits"), attempt projection split.
    5. Final area-ratio noise filter.
    """
    h_img, w_img = binary_seg.shape
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary_seg, connectivity=8
    )

    boxes = []
    for i in range(1, num_labels):
        x   = int(stats[i, cv2.CC_STAT_LEFT])
        y   = int(stats[i, cv2.CC_STAT_TOP])
        bw  = int(stats[i, cv2.CC_STAT_WIDTH])
        bh  = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if bw > w_img * 0.95 and bh > h_img * 0.95:
            continue
        boxes.append([x, y, bw, bh])

    if not boxes:
        return []

    boxes.sort(key=lambda b: b[0])

    # Merge intra-digit fragments (small gap only)
    merged = [boxes[0][:]]
    for x, y, bw, bh in boxes[1:]:
        prev = merged[-1]
        prev_right = prev[0] + prev[2]
        if x - prev_right <= merge_gap:
            nx = min(prev[0], x)
            ny = min(prev[1], y)
            nr = max(prev[0] + prev[2], x + bw)
            nb = max(prev[1] + prev[3], y + bh)
            merged[-1] = [nx, ny, nr - nx, nb - ny]
        else:
            merged.append([x, y, bw, bh])

    # Projection-profile split for suspiciously wide segments.
    # Threshold: 1.2x the image height — a single upright Devanagari digit
    # is roughly square, so anything wider than 1.2x the canvas height is
    # probably two or more merged digits.
    max_single_w = h_img * 1.2
    expanded = []
    for box in merged:
        if box[2] > max_single_w:
            subs = _split_by_projection(binary_seg, box, min_area)
            expanded.extend(subs)
        else:
            expanded.append(box)

    expanded.sort(key=lambda b: b[0])

    # Drop tiny fragments relative to the largest segment
    if expanded:
        max_area = max(b[2] * b[3] for b in expanded)
        expanded = [b for b in expanded if b[2] * b[3] >= max_area * 0.05]

    return [tuple(b) for b in expanded]


# ── Per-digit model input ──────────────────────────────────────

def _segment_to_model_input(masked: np.ndarray, bbox: tuple,
                             target_size: int = 32):
    """
    Crop one digit from the dilated masked array and return
    (model_input_array, preview_b64).

    Uses the dilated 'masked' (not binary_seg) so that the model input
    retains the same stroke quality as single-character preprocessing.
    """
    x, y, bw, bh = bbox
    h_img, w_img = masked.shape

    initial_pad = max(6, int(max(bw, bh) * 0.15))
    x1 = max(0, x - initial_pad)
    y1 = max(0, y - initial_pad)
    x2 = min(w_img, x + bw + initial_pad)
    y2 = min(h_img, y + bh + initial_pad)

    crop = masked[y1:y2, x1:x2]
    if crop.size == 0 or crop.max() == 0:
        return None, None

    rows = np.any(crop > 0, axis=1)
    cols = np.any(crop > 0, axis=0)
    if not rows.any() or not cols.any():
        return None, None
    rmin = int(np.where(rows)[0][0]);  rmax = int(np.where(rows)[0][-1])
    cmin = int(np.where(cols)[0][0]);  cmax = int(np.where(cols)[0][-1])
    tight = crop[rmin:rmax + 1, cmin:cmax + 1]

    th, tw = tight.shape
    content_pad = max(3, int(max(tw, th) * 0.12))
    side = max(tw, th) + 2 * content_pad
    square = np.zeros((side, side), dtype=np.uint8)
    oy = (side - th) // 2
    ox = (side - tw) // 2
    square[oy:oy + th, ox:ox + tw] = tight

    pil_img = Image.fromarray(square).resize((target_size, target_size), Image.LANCZOS)

    # 64x64 debug preview (upscaled with nearest-neighbour so pixels stay sharp)
    preview_b64 = pil_to_base64(pil_img.resize((64, 64), Image.NEAREST))

    arr = np.array(pil_img).astype("float32") / 255.0
    arr = arr[np.newaxis, :, :, np.newaxis]   # (1, 32, 32, 1)
    return arr, preview_b64


# ── Annotation helper ──────────────────────────────────────────

def _annotate_segmentation(original_pil: Image.Image, boxes: list) -> str:
    annotated = original_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    colours = ["#ef4444", "#3b82f6", "#22c55e",
               "#f59e0b", "#8b5cf6", "#ec4899"]
    for i, (x, y, bw, bh) in enumerate(boxes):
        colour = colours[i % len(colours)]
        draw.rectangle([x, y, x + bw, y + bh], outline=colour, width=2)
        draw.text((x + 3, y + 2), str(i + 1), fill=colour)
    annotated.thumbnail((400, 300), Image.LANCZOS)
    return pil_to_base64(annotated)


def _geometric_mean(values: list) -> float:
    if not values:
        return 0.0
    log_sum = sum(math.log(max(v, 0.01)) for v in values)
    return round(math.exp(log_sum / len(values)), 2)


# ── Public API ─────────────────────────────────────────────────

def predict_multidigit(image_source, source_type: str = "canvas") -> dict:
    """
    Full multi-digit recognition pipeline.

    1. Global preprocessing  – invert, Otsu-threshold, produce two binaries
    2. Segmentation          – connected components on undilated binary,
                               small merge_gap, projection-profile splitting
    3. Per-segment predict   – crop from dilated masked → resize → model.predict
    4. Digit filter          – fall back to best digit class if top-1 is non-digit
    5. Reconstruct           – build Devanagari + Arabic numeral strings
    6. Assemble response
    """
    try:
        # ── Step 1: Preprocessing ──────────────────────────────
        prep = _preprocess_full_image(image_source, source_type=source_type)
        binary_seg = prep["binary_seg"]   # undilated — for segmentation
        binary     = prep["binary"]       # dilated  — for preview only
        masked     = prep["masked"]       # dilated  — for per-digit model input
        original_pil = prep["original_pil"]

        # ── Step 2: Segmentation ───────────────────────────────
        boxes = _find_segments(binary_seg, min_area=50, merge_gap=4)
        warnings = []

        if len(boxes) == 0:
            return {
                "success": False,
                "error": (
                    "No digit segments detected. "
                    "Draw digits more clearly with strong, bold strokes "
                    "and leave a small gap between each digit."
                ),
                "warnings": ["no_segments"],
            }

        if len(boxes) > MAX_SEGMENTS:
            warnings.append(
                f"Found {len(boxes)} segments — keeping the first {MAX_SEGMENTS}. "
                "Redraw with clearer spacing if a digit is missing."
            )
            boxes = boxes[:MAX_SEGMENTS]

        # ── Steps 3 + 4: Per-segment prediction ───────────────
        segments = []
        confidences = []

        for i, bbox in enumerate(boxes):
            arr, input_preview = _segment_to_model_input(masked, bbox)
            if arr is None:
                warnings.append(f"Digit {i + 1}: could not crop segment, skipped.")
                continue

            probs = model.predict(arr, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            raw_label = idx_to_label[str(pred_idx)]
            conf = round(float(probs[pred_idx] * 100), 2)

            # If top-1 is not a digit, scan all ranks for the best digit class
            if raw_label not in DIGIT_LABELS:
                best_digit_label = None
                best_digit_conf = 0.0
                for idx in probs.argsort()[::-1]:
                    candidate = idx_to_label[str(int(idx))]
                    if candidate in DIGIT_LABELS:
                        best_digit_label = candidate
                        best_digit_conf = round(float(probs[idx] * 100), 2)
                        break

                if best_digit_label:
                    warnings.append(
                        f"Digit {i + 1}: model was uncertain — "
                        f"best match is {DEVANAGARI_DIGITS[best_digit_label]} "
                        f"({ARABIC_DIGITS[best_digit_label]}) "
                        f"at {best_digit_conf:.1f}% confidence."
                    )
                    raw_label = best_digit_label
                    conf = best_digit_conf
                else:
                    warnings.append(f"Digit {i + 1}: no digit class found in predictions.")
                    raw_label = "digit_0"
                    conf = 0.0

            # Top-3 digit classes (scan all 59 ranks)
            top3 = []
            for idx in probs.argsort()[::-1]:
                rl = idx_to_label[str(int(idx))]
                if rl in DIGIT_LABELS:
                    top3.append({
                        "raw_label": rl,
                        "char": DEVANAGARI_DIGITS[rl],
                        "roman": ARABIC_DIGITS[rl],
                        "confidence": round(float(probs[idx] * 100), 2),
                    })
                if len(top3) == 3:
                    break

            confidences.append(conf)
            x, y, bw, bh = bbox
            segments.append({
                "position": i,
                "raw_label": raw_label,
                "char": DEVANAGARI_DIGITS[raw_label],
                "roman": ARABIC_DIGITS[raw_label],
                "confidence": conf,
                "confidence_level": confidence_level(conf),
                "bbox": {"x": x, "y": y, "w": bw, "h": bh},
                "top3": top3,
                "model_input_preview": input_preview,
            })

        if not segments:
            return {
                "success": False,
                "error": "Could not extract any valid digit from the detected segments.",
                "warnings": warnings,
            }

        # ── Step 5: Reconstruct ────────────────────────────────
        full_deva   = "".join(s["char"]  for s in segments)
        full_arabic = "".join(s["roman"] for s in segments)
        overall_conf  = _geometric_mean(confidences)
        overall_level = confidence_level(overall_conf)

        if overall_conf < 50:
            warnings.append(
                "Overall confidence is low — try redrawing with bolder strokes "
                "and clearer spacing between digits."
            )
        for seg in segments:
            if seg["confidence"] < 50:
                warnings.append(
                    f"Digit {seg['position'] + 1} ({seg['char']}) "
                    f"has low confidence ({seg['confidence']}%)."
                )

        # ── Step 6: Previews ───────────────────────────────────
        seg_img_b64 = _annotate_segmentation(original_pil, boxes)

        orig_thumb = original_pil.copy()
        orig_thumb.thumbnail((128, 128), Image.LANCZOS)

        binary_pil = Image.fromarray(binary).convert("RGB")
        binary_thumb = binary_pil.copy()
        binary_thumb.thumbnail((128, 128), Image.LANCZOS)

        return {
            "success": True,
            "full_number":             full_deva,
            "full_number_arabic":      full_arabic,
            "digit_count":             len(segments),
            "overall_confidence":      overall_conf,
            "overall_confidence_level": overall_level,
            "segments":                segments,
            "segmentation_image":      seg_img_b64,
            "preprocessing": {
                "original": pil_to_base64(orig_thumb),
                "binary":   pil_to_base64(binary_thumb),
            },
            "warnings":    warnings,
            "was_inverted": prep["was_inverted"],
        }

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "warnings": [],
        }
