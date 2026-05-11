"""
Advanced FastAPI Backend for Devanagari Character Recognition.
Endpoints:
  /predict              – Predict from uploaded file
  /predict-base64       – Predict from base64 image (canvas/camera)
  /batch-predict        – Predict multiple images at once
  /health               – API health & model status
  /metrics              – Training plots & evaluation artifacts
  /evaluation-summary   – Full model evaluation metrics (JSON)
  /training-history     – Training history data (JSON)
  /session-stats        – Session analytics
  /character-info/{lbl} – Character metadata lookup
  /model-info           – Model architecture & parameter summary
  /feedback             – Collect user feedback on predictions
  /voice-data/{label}   – Return pronunciation text for TTS

  === Note Reader (Secondary Module) ===
  /predict-note         – Predict note denomination from uploaded file
  /predict-note-base64  – Predict note denomination from base64 image
  /note-info/{label}    – Note denomination metadata lookup
  /note-voice-data/{label} – Voice text for note TTS
"""

import os
import io
import base64
import time
import json
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from predictor import predict_character
from character_metadata import CHAR_METADATA, get_metadata
from note_predictor import predict_note, NOTE_MODEL_AVAILABLE
from note_metadata import get_note_metadata, get_note_voice_text, NOTE_METADATA

# ── Multi-Digit Extension (advanced module) ──────────────────
try:
    from multidigit_predictor import predict_multidigit
    MULTIDIGIT_AVAILABLE = True
    print("[INFO] Multi-digit predictor loaded.")
except Exception as _md_err:
    print(f"[WARN] Multi-digit predictor not available: {_md_err}")
    MULTIDIGIT_AVAILABLE = False

# ── App setup ─────────────────────────────────────────────────────
app = FastAPI(
    title="Devanagari Character Recognition API",
    description="Advanced deep learning handwritten Devanagari character recognition with Grad-CAM, analytics, educational metadata, and Nepali currency note reader",
    version="4.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "saved_model")
app.mount("/artifacts", StaticFiles(directory=STATIC_DIR), name="artifacts")

# ── In-memory stores ─────────────────────────────────────────────
session_stats = {
    "total_predictions": 0,
    "total_confidence": 0.0,
    "highest_confidence": 0.0,
    "highest_confidence_char": "",
    "lowest_confidence": 100.0,
    "lowest_confidence_char": "",
    "last_prediction": None,
    "start_time": time.time(),
    "class_distribution": {},
    "confidence_history": [],
}

note_session_stats = {
    "total_predictions": 0,
    "last_prediction": None,
}

feedback_store = []

multidigit_session_stats = {
    "total_predictions": 0,
    "last_prediction": None,
}


def update_session(result: dict):
    if not result.get("success"):
        return
    pred = result["prediction"]
    conf = pred["confidence"]
    char = pred["char"]

    session_stats["total_predictions"] += 1
    session_stats["total_confidence"] += conf

    if conf > session_stats["highest_confidence"]:
        session_stats["highest_confidence"] = conf
        session_stats["highest_confidence_char"] = char

    if conf < session_stats["lowest_confidence"]:
        session_stats["lowest_confidence"] = conf
        session_stats["lowest_confidence_char"] = char

    session_stats["last_prediction"] = {
        "char": char,
        "roman": pred["roman"],
        "confidence": conf,
        "timestamp": time.time(),
    }

    label = pred["raw_label"]
    session_stats["class_distribution"][label] = \
        session_stats["class_distribution"].get(label, 0) + 1

    session_stats["confidence_history"].append(conf)
    if len(session_stats["confidence_history"]) > 100:
        session_stats["confidence_history"] = session_stats["confidence_history"][-100:]


def update_note_session(result: dict):
    if not result.get("success"):
        return
    pred = result["prediction"]
    note_session_stats["total_predictions"] += 1
    note_session_stats["last_prediction"] = {
        "denomination": pred["denomination"],
        "value": pred["value"],
        "english_name": pred["english_name"],
        "confidence": pred["confidence"],
        "timestamp": time.time(),
    }


def update_multidigit_session(result: dict):
    if not result.get("success"):
        return
    multidigit_session_stats["total_predictions"] += 1
    multidigit_session_stats["last_prediction"] = {
        "full_number": result.get("full_number"),
        "full_number_arabic": result.get("full_number_arabic"),
        "digit_count": result.get("digit_count"),
        "overall_confidence": result.get("overall_confidence"),
        "timestamp": time.time(),
    }


# ── Pydantic models ──────────────────────────────────────────────
class Base64Request(BaseModel):
    image: str
    input_type: Optional[str] = "canvas"

class BatchBase64Request(BaseModel):
    images: List[str]
    input_type: Optional[str] = "batch"

class FeedbackRequest(BaseModel):
    predicted_label: str
    correct_label: Optional[str] = None
    is_correct: bool
    confidence: Optional[float] = None
    input_type: Optional[str] = None
    notes: Optional[str] = None


# ══════════════════════════════════════════════════════════════════
#  EXISTING CHARACTER RECOGNITION ROUTES
# ══════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "Devanagari Character Recognition API v4.1 running.",
        "version": "4.1.0",
        "endpoints": [
            "/predict", "/predict-base64", "/batch-predict",
            "/health", "/metrics", "/evaluation-summary",
            "/training-history", "/session-stats",
            "/character-info/{label}", "/model-info",
            "/feedback", "/voice-data/{label}",
            "/predict-note", "/predict-note-base64",
            "/note-info/{label}", "/note-voice-data/{label}",
            "/predict-multidigit", "/predict-multidigit-base64",
        ],
    }


@app.get("/health")
def health():
    uptime = time.time() - session_stats["start_time"]
    return {
        "status": "healthy",
        "model_loaded": True,
        "note_model_loaded": NOTE_MODEL_AVAILABLE,
        "note_reader_available": True,  # Always true now (fallback exists)
        "uptime_seconds": round(uptime, 1),
        "total_classes": len(CHAR_METADATA),
        "total_predictions": session_stats["total_predictions"],
        "total_note_predictions": note_session_stats["total_predictions"],
        "total_multidigit_predictions": multidigit_session_stats["total_predictions"],
        "multidigit_available": MULTIDIGIT_AVAILABLE,
        "version": "4.2.0",
    }


@app.get("/metrics")
def metrics():
    artifacts = {}
    artifact_files = {
        "training_plot": "training_plot.png",
        "confusion_matrix": "confusion_matrix.png",
        "per_class_accuracy": "per_class_accuracy.png",
        "lr_schedule": "lr_schedule.png",
        "note_training_plot": "note_training_plot.png",
    }
    for key, filename in artifact_files.items():
        path = os.path.join(STATIC_DIR, filename)
        if os.path.exists(path):
            artifacts[key] = f"/artifacts/{filename}"
    return artifacts


@app.get("/evaluation-summary")
def evaluation_summary():
    path = os.path.join(STATIC_DIR, "evaluation_summary.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Evaluation summary not generated yet. Run ml/evaluate.py first.")
    with open(path, "r") as f:
        return json.load(f)


@app.get("/training-history")
def training_history():
    path = os.path.join(STATIC_DIR, "training_history.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Training history not found. Run ml/train.py first.")
    with open(path, "r") as f:
        return json.load(f)


@app.get("/model-info")
def model_info():
    from predictor import model as loaded_model
    summary_path = os.path.join(STATIC_DIR, "model_summary.txt")
    summary_text = ""
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary_text = f.read()
    return {
        "total_parameters": loaded_model.count_params(),
        "input_shape": str(loaded_model.input_shape),
        "output_shape": str(loaded_model.output_shape),
        "num_layers": len(loaded_model.layers),
        "num_classes": len(CHAR_METADATA),
        "note_model_available": NOTE_MODEL_AVAILABLE,
        "note_reader_available": True,
        "summary": summary_text,
    }


@app.get("/session-stats")
def get_session_stats():
    total = session_stats["total_predictions"]
    avg = round(session_stats["total_confidence"] / total, 2) if total > 0 else 0
    return {
        "total_predictions": total,
        "average_confidence": avg,
        "highest_confidence": session_stats["highest_confidence"],
        "highest_confidence_char": session_stats["highest_confidence_char"],
        "lowest_confidence": session_stats["lowest_confidence"] if total > 0 else 0,
        "lowest_confidence_char": session_stats["lowest_confidence_char"] if total > 0 else "",
        "last_prediction": session_stats["last_prediction"],
        "class_distribution": session_stats["class_distribution"],
        "confidence_history": session_stats["confidence_history"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    result = predict_character(io.BytesIO(contents), source_type="upload")
    update_session(result)
    return result


@app.post("/predict-base64")
async def predict_base64(req: Base64Request):
    try:
        img_data = req.image
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        decoded = base64.b64decode(img_data)
        result = predict_character(io.BytesIO(decoded), source_type=req.input_type or "canvas")
        update_session(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(req: BatchBase64Request):
    results = []
    for img_str in req.images[:10]:
        try:
            img_data = img_str
            if "," in img_data:
                img_data = img_data.split(",", 1)[1]
            decoded = base64.b64decode(img_data)
            result = predict_character(io.BytesIO(decoded), include_gradcam=False)
            update_session(result)
            results.append(result)
        except Exception as e:
            results.append({"success": False, "error": str(e)})
    return {"results": results, "count": len(results)}


@app.get("/character-info/{label}")
def character_info(label: str):
    meta = get_metadata(label)
    if meta["type"] == "unknown":
        raise HTTPException(status_code=404, detail=f"Label '{label}' not found.")
    return {"label": label, **meta}


@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    entry = {
        "predicted_label": req.predicted_label,
        "correct_label": req.correct_label,
        "is_correct": req.is_correct,
        "confidence": req.confidence,
        "input_type": req.input_type,
        "notes": req.notes,
        "timestamp": time.time(),
    }
    feedback_store.append(entry)
    return {
        "message": "Feedback recorded. Thank you!",
        "total_feedback": len(feedback_store),
    }


@app.get("/feedback-stats")
def feedback_stats():
    total = len(feedback_store)
    correct = sum(1 for f in feedback_store if f["is_correct"])
    return {
        "total_feedback": total,
        "correct_count": correct,
        "incorrect_count": total - correct,
        "user_accuracy": round(correct / total * 100, 2) if total > 0 else 0,
    }


@app.get("/voice-data/{label}")
def voice_data(label: str):
    meta = get_metadata(label)
    if meta["type"] == "unknown":
        raise HTTPException(status_code=404, detail=f"Label '{label}' not found.")
    speech_text = f"The character is {meta['char']}, romanised as {meta['roman']}."
    if meta.get("nepali_name"):
        speech_text += f" In Nepali, it is called {meta['nepali_name']}."
    if meta.get("example_word"):
        speech_text += f" An example word is {meta['example_word']}."
    return {
        "label": label,
        "char": meta["char"],
        "roman": meta["roman"],
        "speech_text": speech_text,
    }


# ══════════════════════════════════════════════════════════════════
#  NOTE READER ENDPOINTS (Secondary Advanced Module)
# ══════════════════════════════════════════════════════════════════

@app.post("/predict-note")
async def predict_note_upload(file: UploadFile = File(...)):
    contents = await file.read()
    result = predict_note(io.BytesIO(contents))
    update_note_session(result)
    return result


@app.post("/predict-note-base64")
async def predict_note_base64(req: Base64Request):
    try:
        img_data = req.image
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        decoded = base64.b64decode(img_data)
        result = predict_note(io.BytesIO(decoded))
        update_note_session(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


@app.get("/note-info/{label}")
def note_info(label: str):
    meta = get_note_metadata(label)
    if meta["value"] == 0:
        raise HTTPException(status_code=404, detail=f"Denomination '{label}' not found.")
    return {"label": label, **meta}


@app.get("/note-voice-data/{label}")
def note_voice_data(
    label: str,
    mode: str = Query("english", pattern="^(english|nepali|mixed|confidence)$"),
    confidence: float = Query(100.0)
):
    meta = get_note_metadata(label)
    if meta["value"] == 0:
        raise HTTPException(status_code=404, detail=f"Denomination '{label}' not found.")

    speech_text = get_note_voice_text(label, mode, confidence)

    return {
        "denomination": label,
        "mode": mode,
        "speech_text": speech_text,
        "lang": "ne-NP" if mode == "nepali" else "en-US",
        "all_modes": {
            "english": get_note_voice_text(label, "english", confidence),
            "nepali": get_note_voice_text(label, "nepali", confidence),
            "mixed": get_note_voice_text(label, "mixed", confidence),
            "confidence": get_note_voice_text(label, "confidence", confidence),
        },
    }


# ══════════════════════════════════════════════════════════════
#  MULTI-DIGIT RECOGNITION ENDPOINTS (Advanced Extension)
# ══════════════════════════════════════════════════════════════

@app.post("/predict-multidigit")
async def predict_multidigit_upload(file: UploadFile = File(...)):
    """Predict a multi-digit Devanagari numeral from an uploaded image."""
    if not MULTIDIGIT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Multi-digit predictor not available. "
                   "Ensure backend/multidigit_predictor.py is present and "
                   "opencv-python is installed."
        )
    contents = await file.read()
    result = predict_multidigit(io.BytesIO(contents), source_type="upload")
    update_multidigit_session(result)
    return result


@app.post("/predict-multidigit-base64")
async def predict_multidigit_b64(req: Base64Request):
    """Predict a multi-digit Devanagari numeral from a base64-encoded image."""
    if not MULTIDIGIT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Multi-digit predictor not available."
        )
    try:
        img_data = req.image
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        decoded = base64.b64decode(img_data)
        result = predict_multidigit(io.BytesIO(decoded), source_type=req.input_type or "canvas")
        update_multidigit_session(result)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image data: {str(e)}"
        )


# ── Run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
