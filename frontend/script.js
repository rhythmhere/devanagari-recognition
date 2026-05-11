/* ══════════════════════════════════════════════════════════════
   Devanagari Character Recognition – Advanced Frontend Logic v4.5
   Features: Grad-CAM, Voice Output, Feedback, Top-5, Eval Dashboard,
             Confidence Trend Chart, Uncertainty Analysis, Model Info,
             Note Reader (with fallback), Bilingual Voice, Voice Selector,
             Multi-Digit Drawing Canvas (draw tab only)
   ══════════════════════════════════════════════════════════════ */

const API_BASE = "http://127.0.0.1:8000";

// ── DOM references ────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const themeToggle     = $("#themeToggle");
const toastContainer  = $("#toastContainer");

// Tabs
const tabPills  = $$(".tab-pill");
const tabPanels = $$(".tab-panel");

// Upload
const fileInput       = $("#fileInput");
const dropZone        = $("#dropZone");
const browseBtn       = $("#browseBtn");
const uploadPreview   = $("#uploadPreview");
const uploadPreviewWrap = $("#uploadPreviewWrap");

// Canvas — single character
const drawCanvas      = $("#drawCanvas");
const clearCanvasBtn  = $("#clearCanvasBtn");
const strokeSlider    = $("#strokeSlider");
const ctx             = drawCanvas.getContext("2d");

// Canvas — multi-digit (wider canvas, only in draw tab)
const drawCanvasMulti  = $("#drawCanvasMulti");
const singleCanvasWrap = $("#singleCanvasWrap");
const multiCanvasWrap  = $("#multiCanvasWrap");
let ctxMulti           = null;   // lazily initialized when multi mode first used

// Camera
const cameraVideo      = $("#cameraVideo");
const cameraSnap       = $("#cameraSnap");
const startCameraBtn   = $("#startCameraBtn");
const captureBtn       = $("#captureBtn");
const stopCameraBtn    = $("#stopCameraBtn");
const cameraPreview    = $("#cameraPreview");
const cameraPreviewWrap  = $("#cameraPreviewWrap");
const cameraGuideFrame = $("#cameraGuideFrame");

// Note reader
const noteFileInput    = $("#noteFileInput");
const noteDropZone     = $("#noteDropZone");
const noteBrowseBtn    = $("#noteBrowseBtn");
const notePreview      = $("#notePreview");
const notePreviewWrap  = $("#notePreviewWrap");

// Actions
const predictBtn = $("#predictBtn");
const resetBtn   = $("#resetBtn");
const sampleBtn  = $("#sampleBtn");

// Loading
const loadingOverlay = $("#loadingOverlay");
const loadingText    = $("#loadingText");

// Result sections
const resultsSection          = $("#resultsSection");
const noteResultsSection      = $("#noteResultsSection");
const multidigitResultsSection = $("#multidigitResultsSection");

// Character result refs
const lowConfWarn    = $("#lowConfWarn");
const lowConfText    = $("#lowConfText");

// Stats
const statTotal = $("#statTotal");
const statAvg   = $("#statAvg");
const statHigh  = $("#statHigh");
const statLast  = $("#statLast");

// History
const historyList   = $("#historyList");
const historyEmpty  = $("#historyEmpty");
const exportCsvBtn  = $("#exportCsvBtn");
const clearHistBtn  = $("#clearHistBtn");

// Voice
const voiceBtn           = $("#voiceBtn");
const feedbackCorrectBtn = $("#feedbackCorrectBtn");
const feedbackWrongBtn   = $("#feedbackWrongBtn");
const noteVoiceBtn       = $("#noteVoiceBtn");
const noteReplayBtn      = $("#noteReplayBtn");
const voiceSelector      = $("#voiceSelector");

// ── State ─────────────────────────────────────────────────────
let activeTab   = "upload";
let drawMode    = "single";   // "single" | "multi"  (only relevant inside draw tab)
let uploadMode  = "single";   // "single" | "multi"  (upload tab)
let cameraMode  = "single";   // "single" | "multi"  (camera tab)
let cameraStream = null;
let capturedImageData = null;
let history = [];
let drawing = false;
let drawingMulti = false;
let lastPredictionData = null;
let lastNotePredictionData = null;
let noteVoiceMode = "english";
let lastSpokenText = "";
let lastSpokenLang = "en-US";
let selectedVoice = null;
let availableVoices = [];
let confidenceHistory = [];

// ══════════════════════════════════════════════════════════════
//  THEME
// ══════════════════════════════════════════════════════════════
function initTheme() {
  const saved = localStorage.getItem("dcr-theme");
  if (saved) {
    document.documentElement.setAttribute("data-theme", saved);
  } else {
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.setAttribute("data-theme", prefersDark ? "dark" : "light");
  }
}
function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme");
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("dcr-theme", next);
}
initTheme();
themeToggle.addEventListener("click", toggleTheme);

// ══════════════════════════════════════════════════════════════
//  TOAST
// ══════════════════════════════════════════════════════════════
function showToast(msg, type = "info", duration = 3000) {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  toastContainer.appendChild(el);
  setTimeout(() => {
    el.classList.add("out");
    setTimeout(() => el.remove(), 300);
  }, duration);
}

// ══════════════════════════════════════════════════════════════
//  VOICE SELECTOR
// ══════════════════════════════════════════════════════════════
function populateVoices() {
  if (!voiceSelector) return;
  availableVoices = window.speechSynthesis.getVoices();
  voiceSelector.innerHTML = '<option value="">Default Voice</option>';
  availableVoices.forEach((voice, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `${voice.name} (${voice.lang})`;
    voiceSelector.appendChild(opt);
  });
}
if ("speechSynthesis" in window) {
  populateVoices();
  window.speechSynthesis.onvoiceschanged = populateVoices;
}
if (voiceSelector) {
  voiceSelector.addEventListener("change", () => {
    const idx = voiceSelector.value;
    selectedVoice = idx !== "" ? availableVoices[parseInt(idx)] : null;
  });
}

// ══════════════════════════════════════════════════════════════
//  TABS
// ══════════════════════════════════════════════════════════════
tabPills.forEach((pill) => {
  pill.addEventListener("click", () => {
    const tab = pill.dataset.tab;
    const prevTab = activeTab;
    activeTab = tab;
    tabPills.forEach((p) => p.classList.remove("active"));
    pill.classList.add("active");
    tabPanels.forEach((panel) => {
      panel.classList.toggle("active", panel.id === `panel-${tab}`);
    });

    // Hide all result sections on tab switch
    resultsSection.classList.add("hidden");
    noteResultsSection.classList.add("hidden");
    multidigitResultsSection.classList.add("hidden");

    if (tab !== prevTab) {
      if (lowConfWarn) lowConfWarn.classList.add("hidden");
      const gradcamSection = $("#gradcamSection");
      if (gradcamSection) gradcamSection.style.display = "none";
    }
  });
});

// ══════════════════════════════════════════════════════════════
//  UPLOAD
// ══════════════════════════════════════════════════════════════
browseBtn.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("click", (e) => {
  if (e.target === browseBtn || browseBtn.contains(e.target)) return;
  fileInput.click();
});
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  previewFile(file);
});
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (!file) return;
  fileInput.files = e.dataTransfer.files;
  previewFile(file);
});
function previewFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    uploadPreview.src = e.target.result;
    uploadPreviewWrap.classList.add("show");
  };
  reader.readAsDataURL(file);
  showToast("Image loaded", "success");
}

// ══════════════════════════════════════════════════════════════
//  NOTE UPLOAD
// ══════════════════════════════════════════════════════════════
if (noteBrowseBtn) {
  noteBrowseBtn.addEventListener("click", () => noteFileInput.click());
}
if (noteDropZone) {
  noteDropZone.addEventListener("click", (e) => {
    if (e.target === noteBrowseBtn || (noteBrowseBtn && noteBrowseBtn.contains(e.target))) return;
    noteFileInput.click();
  });
}
if (noteFileInput) {
  noteFileInput.addEventListener("change", () => {
    const file = noteFileInput.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      notePreview.src = e.target.result;
      notePreviewWrap.classList.add("show");
    };
    reader.readAsDataURL(file);
    showToast("Note image loaded", "success");
  });
}
if (noteDropZone) {
  noteDropZone.addEventListener("dragover", (e) => { e.preventDefault(); noteDropZone.classList.add("dragover"); });
  noteDropZone.addEventListener("dragleave", () => noteDropZone.classList.remove("dragover"));
  noteDropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    noteDropZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (!file) return;
    noteFileInput.files = e.dataTransfer.files;
    const reader = new FileReader();
    reader.onload = (ev) => {
      notePreview.src = ev.target.result;
      notePreviewWrap.classList.add("show");
    };
    reader.readAsDataURL(file);
    showToast("Note image loaded", "success");
  });
}

// ══════════════════════════════════════════════════════════════
//  DRAWING CANVAS — single character (280×280)
// ══════════════════════════════════════════════════════════════
function initCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  ctx.lineWidth = parseInt(strokeSlider.value);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#000000";
}
initCanvas();

// ── Multi-digit canvas (560×200) — lazily initialized ──────────
function initMultiCanvas() {
  if (!drawCanvasMulti) return;
  ctxMulti = drawCanvasMulti.getContext("2d");
  ctxMulti.fillStyle = "#ffffff";
  ctxMulti.fillRect(0, 0, drawCanvasMulti.width, drawCanvasMulti.height);
  ctxMulti.lineWidth = parseInt(strokeSlider.value);
  ctxMulti.lineCap = "round";
  ctxMulti.lineJoin = "round";
  ctxMulti.strokeStyle = "#000000";
}

// Stroke slider — updates whichever canvas context is active
strokeSlider.addEventListener("input", () => {
  const w = parseInt(strokeSlider.value);
  ctx.lineWidth = w;
  if (ctxMulti) ctxMulti.lineWidth = w;
});

// Generic canvas position helper (works for any canvas element)
function getCanvasPosFor(e, canvasEl) {
  const rect = canvasEl.getBoundingClientRect();
  const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
  const y = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;
  return {
    x: x * (canvasEl.width / rect.width),
    y: y * (canvasEl.height / rect.height),
  };
}
function getCanvasPos(e) { return getCanvasPosFor(e, drawCanvas); }

// Single-character canvas events
drawCanvas.addEventListener("mousedown", (e) => { drawing = true; const p = getCanvasPos(e); ctx.beginPath(); ctx.moveTo(p.x, p.y); });
drawCanvas.addEventListener("mousemove", (e) => { if (!drawing) return; const p = getCanvasPos(e); ctx.lineTo(p.x, p.y); ctx.stroke(); });
drawCanvas.addEventListener("mouseup", () => { drawing = false; });
drawCanvas.addEventListener("mouseleave", () => { drawing = false; });
drawCanvas.addEventListener("touchstart", (e) => { e.preventDefault(); drawing = true; const p = getCanvasPos(e); ctx.beginPath(); ctx.moveTo(p.x, p.y); });
drawCanvas.addEventListener("touchmove", (e) => { e.preventDefault(); if (!drawing) return; const p = getCanvasPos(e); ctx.lineTo(p.x, p.y); ctx.stroke(); });
drawCanvas.addEventListener("touchend", () => { drawing = false; });

// Multi-digit canvas events
if (drawCanvasMulti) {
  drawCanvasMulti.addEventListener("mousedown", (e) => {
    if (!ctxMulti) initMultiCanvas();
    drawingMulti = true;
    const p = getCanvasPosFor(e, drawCanvasMulti);
    ctxMulti.beginPath(); ctxMulti.moveTo(p.x, p.y);
  });
  drawCanvasMulti.addEventListener("mousemove", (e) => {
    if (!drawingMulti || !ctxMulti) return;
    const p = getCanvasPosFor(e, drawCanvasMulti);
    ctxMulti.lineTo(p.x, p.y); ctxMulti.stroke();
  });
  drawCanvasMulti.addEventListener("mouseup", () => { drawingMulti = false; });
  drawCanvasMulti.addEventListener("mouseleave", () => { drawingMulti = false; });
  drawCanvasMulti.addEventListener("touchstart", (e) => {
    e.preventDefault();
    if (!ctxMulti) initMultiCanvas();
    drawingMulti = true;
    const p = getCanvasPosFor(e, drawCanvasMulti);
    ctxMulti.beginPath(); ctxMulti.moveTo(p.x, p.y);
  });
  drawCanvasMulti.addEventListener("touchmove", (e) => {
    e.preventDefault();
    if (!drawingMulti || !ctxMulti) return;
    const p = getCanvasPosFor(e, drawCanvasMulti);
    ctxMulti.lineTo(p.x, p.y); ctxMulti.stroke();
  });
  drawCanvasMulti.addEventListener("touchend", () => { drawingMulti = false; });
}

// Clear button — clears whichever canvas is currently active
clearCanvasBtn.addEventListener("click", () => {
  if (drawMode === "multi" && ctxMulti) {
    ctxMulti.fillStyle = "#ffffff";
    ctxMulti.fillRect(0, 0, drawCanvasMulti.width, drawCanvasMulti.height);
  } else {
    initCanvas();
  }
  showToast("Canvas cleared", "info");
});

// ══════════════════════════════════════════════════════════════
//  DRAW MODE SELECTOR (inside Draw tab only)
//  Switching modes shows/hides the appropriate canvas.
//  Multi-digit is NOT available in upload or camera tabs.
// ══════════════════════════════════════════════════════════════
$$(".draw-mode-pill").forEach((pill) => {
  pill.addEventListener("click", () => {
    if (!pill.dataset.dmode) return;  // only handle draw-tab pills here
    const mode = pill.dataset.dmode;
    if (mode === drawMode) return;

    drawMode = mode;
    $$("[data-dmode]").forEach((p) => p.classList.remove("active"));
    pill.classList.add("active");

    if (mode === "single") {
      singleCanvasWrap.style.display = "";
      multiCanvasWrap.style.display = "none";
    } else {
      singleCanvasWrap.style.display = "none";
      multiCanvasWrap.style.display = "";
      if (!ctxMulti) initMultiCanvas();
    }

    // Hide stale results when switching draw modes
    resultsSection.classList.add("hidden");
    multidigitResultsSection.classList.add("hidden");
  });
});

// ══════════════════════════════════════════════════════════════
//  UPLOAD MODE SELECTOR  (Single Character / Multi-Digit Number)
// ══════════════════════════════════════════════════════════════
$$("[data-umode]").forEach((pill) => {
  pill.addEventListener("click", () => {
    const mode = pill.dataset.umode;
    if (mode === uploadMode) return;
    uploadMode = mode;
    $$("[data-umode]").forEach((p) => p.classList.remove("active"));
    pill.classList.add("active");
    const hint = $("#uploadMultiHint");
    if (hint) hint.classList.toggle("hidden", mode !== "multi");
    resultsSection.classList.add("hidden");
    multidigitResultsSection.classList.add("hidden");
  });
});

// ══════════════════════════════════════════════════════════════
//  CAMERA MODE SELECTOR  (Single Character / Multi-Digit Number)
// ══════════════════════════════════════════════════════════════
$$("[data-cmode]").forEach((pill) => {
  pill.addEventListener("click", () => {
    const mode = pill.dataset.cmode;
    if (mode === cameraMode) return;
    cameraMode = mode;
    $$("[data-cmode]").forEach((p) => p.classList.remove("active"));
    pill.classList.add("active");
    const hint = $("#cameraMultiHint");
    if (hint) hint.classList.toggle("hidden", mode !== "multi");
    resultsSection.classList.add("hidden");
    multidigitResultsSection.classList.add("hidden");
    // Guide frame only makes sense for single-character mode
    if (cameraGuideFrame) {
      cameraGuideFrame.classList.toggle("hidden", mode !== "single" || !cameraStream);
    }
  });
});

// ══════════════════════════════════════════════════════════════
//  CAMERA
// ══════════════════════════════════════════════════════════════
startCameraBtn.addEventListener("click", async () => {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } }
    });
    cameraVideo.srcObject = cameraStream;
    await cameraVideo.play();
    captureBtn.disabled = false;
    stopCameraBtn.disabled = false;
    startCameraBtn.disabled = true;
    if (cameraGuideFrame) cameraGuideFrame.classList.remove("hidden");
    showToast("Camera started", "success");
  } catch (err) {
    showToast("Camera access denied: " + err.message, "error");
    console.error("Camera error:", err);
  }
});

captureBtn.addEventListener("click", () => { doCameraCapture(); });

function doCameraCapture() {
  if (!cameraVideo.srcObject) return;
  const snapCtx = cameraSnap.getContext("2d");
  const vw = cameraVideo.videoWidth || 640;
  const vh = cameraVideo.videoHeight || 480;
  if (cameraMode === "single") {
    // Crop the centre 58% square — exactly matches the guide frame overlay.
    // Sending only this region to the backend eliminates desk/hand background
    // and makes the preprocessing inversion and Otsu threshold more reliable.
    const cropSize = Math.round(Math.min(vw, vh) * 0.58);
    const cropX = Math.round((vw - cropSize) / 2);
    const cropY = Math.round((vh - cropSize) / 2);
    cameraSnap.width = cropSize;
    cameraSnap.height = cropSize;
    snapCtx.drawImage(cameraVideo, cropX, cropY, cropSize, cropSize, 0, 0, cropSize, cropSize);
  } else {
    // Multi-digit: send the full frame so segmentation has room for all digits
    cameraSnap.width = vw;
    cameraSnap.height = vh;
    snapCtx.drawImage(cameraVideo, 0, 0, vw, vh);
  }
  capturedImageData = cameraSnap.toDataURL("image/png");
  cameraPreview.src = capturedImageData;
  cameraPreviewWrap.classList.add("show");
  showToast("Image captured", "success");
}

stopCameraBtn.addEventListener("click", () => { stopCamera(); });

function stopCamera() {
  if (cameraStream) { cameraStream.getTracks().forEach((t) => t.stop()); cameraStream = null; }
  cameraVideo.srcObject = null;
  captureBtn.disabled = true;
  stopCameraBtn.disabled = true;
  startCameraBtn.disabled = false;
  if (cameraGuideFrame) cameraGuideFrame.classList.add("hidden");
}

// ══════════════════════════════════════════════════════════════
//  PREDICT — routes to the correct endpoint based on tab + draw mode
// ══════════════════════════════════════════════════════════════
predictBtn.addEventListener("click", () => runPrediction());

async function runPrediction() {
  if (activeTab === "note") return runNotePrediction();

  let body, url, inputType;

  if (activeTab === "upload") {
    const file = fileInput.files[0];
    if (!file) { showToast("Please select an image first", "error"); return; }
    const formData = new FormData();
    formData.append("file", file);
    if (uploadMode === "multi") {
      url = `${API_BASE}/predict-multidigit`;
      inputType = "multidigit-upload";
    } else {
      url = `${API_BASE}/predict`;
      inputType = "upload";
    }
    body = formData;

  } else if (activeTab === "draw") {
    if (drawMode === "multi") {
      // Multi-digit: use the wider canvas and the multidigit endpoint
      if (!ctxMulti) { showToast("Please draw something first", "error"); return; }
      const dataUrl = drawCanvasMulti.toDataURL("image/png");
      url = `${API_BASE}/predict-multidigit-base64`;
      body = JSON.stringify({ image: dataUrl, input_type: "canvas" });
      inputType = "multidigit-canvas";
    } else {
      // Single character
      const dataUrl = drawCanvas.toDataURL("image/png");
      url = `${API_BASE}/predict-base64`;
      body = JSON.stringify({ image: dataUrl, input_type: "canvas" });
      inputType = "canvas";
    }

  } else if (activeTab === "camera") {
    if (!capturedImageData) { showToast("Please capture an image first", "error"); return; }
    if (cameraMode === "multi") {
      url = `${API_BASE}/predict-multidigit-base64`;
      body = JSON.stringify({ image: capturedImageData, input_type: "camera" });
      inputType = "multidigit-camera";
    } else {
      url = `${API_BASE}/predict-base64`;
      body = JSON.stringify({ image: capturedImageData, input_type: "camera" });
      inputType = "camera";
    }
  }

  loadingText.textContent = inputType.startsWith("multidigit-") ? "Recognizing digits…" : "Analyzing character…";
  loadingOverlay.classList.remove("hidden");
  resultsSection.classList.add("hidden");
  noteResultsSection.classList.add("hidden");
  multidigitResultsSection.classList.add("hidden");

  try {
    const headers = body instanceof FormData ? {} : { "Content-Type": "application/json" };
    const res = await fetch(url, { method: "POST", body, headers });
    const data = await res.json();
    loadingOverlay.classList.add("hidden");

    if (data.success) {
      if (inputType.startsWith("multidigit-")) {
        displayMultidigitResults(data);
        addMultidigitToHistory(data, inputType);
        recordConfidence(data.overall_confidence, data.full_number || "multi", "multi-digit");
        const numStr = data.full_number && data.full_number_arabic
          ? `${data.full_number} (${data.full_number_arabic})`
          : (data.full_number || data.full_number_arabic || "–");
        showToast(`Recognized: ${numStr}`, "success");
      } else {
        lastPredictionData = data;
        displayResults(data, inputType);
        addToHistory(data, inputType);
        recordConfidence(data.prediction.confidence, data.prediction.char, inputType);
        refreshStats();
        showToast("Prediction complete!", "success");
      }
    } else {
      if (inputType.startsWith("multidigit-")) {
        multidigitResultsSection.classList.remove("hidden");
        const warnBar = $("#multidigitWarnBar");
        const warnText = $("#multidigitWarnText");
        if (warnBar) warnBar.classList.remove("hidden");
        if (warnText) warnText.textContent = data.error || "Could not recognize digits. Try drawing more clearly.";
        multidigitResultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
      }
      showToast(data.error || "Prediction failed", "error");
    }
  } catch (err) {
    loadingOverlay.classList.add("hidden");
    showToast("Could not connect to backend", "error");
    console.error(err);
  }
}

// ══════════════════════════════════════════════════════════════
//  NOTE PREDICTION
// ══════════════════════════════════════════════════════════════
async function runNotePrediction() {
  const file = noteFileInput.files[0];
  if (!file) { showToast("Please select a note image first", "error"); return; }

  const formData = new FormData();
  formData.append("file", file);
  loadingText.textContent = "Analyzing currency note…";
  loadingOverlay.classList.remove("hidden");
  resultsSection.classList.add("hidden");
  noteResultsSection.classList.add("hidden");
  multidigitResultsSection.classList.add("hidden");

  try {
    const res = await fetch(`${API_BASE}/predict-note`, { method: "POST", body: formData });
    const data = await res.json();
    loadingOverlay.classList.add("hidden");

    if (data.success) {
      lastNotePredictionData = data;
      displayNoteResults(data);
      const isRecognized = data.recognized !== false;
      addToHistory({
        prediction: {
          char: isRecognized ? `₨${data.prediction.value}` : "?",
          roman: data.prediction.english_name,
          nepali_name: data.prediction.nepali_name,
          type: isRecognized ? "note" : "note (unrecognized)",
          confidence: data.prediction.confidence,
          confidence_level: data.prediction.confidence_level,
          raw_label: data.prediction.denomination,
        },
        analysis: null,
      }, "note");
      if (isRecognized) recordConfidence(data.prediction.confidence, data.prediction.denomination || "note", "note");
      if (isRecognized) showToast("Note prediction complete!", "success");
      else showToast("Note not recognized — image may be outside the trained dataset", "error");
    } else {
      showToast(data.error || "Note prediction failed", "error");
    }
  } catch (err) {
    loadingOverlay.classList.add("hidden");
    showToast("Could not connect to backend", "error");
    console.error(err);
  }
}

// ══════════════════════════════════════════════════════════════
//  DISPLAY RESULTS (Character — single char or camera)
// ══════════════════════════════════════════════════════════════
function displayResults(data, inputType) {
  const pred = data.prediction;
  resultsSection.classList.remove("hidden");
  noteResultsSection.classList.add("hidden");
  multidigitResultsSection.classList.add("hidden");

  const resultHero = $("#resultHero");
  if (pred.confidence_level === "low") {
    lowConfWarn.classList.remove("hidden");
    let warnMsg = `Low confidence (${pred.confidence}%) – check the top 3 alternatives below.`;
    if (inputType === "camera") {
      warnMsg += " Tips: ensure bright even lighting, place the character squarely inside the guide frame, use plain white paper.";
    } else if (inputType === "canvas") {
      warnMsg += " Tip: use bold strokes and fill more of the drawing area.";
    } else {
      warnMsg += " Try a clearer, higher-contrast image.";
    }
    lowConfText.textContent = warnMsg;
    if (resultHero) resultHero.classList.add("low-conf");
  } else {
    lowConfWarn.classList.add("hidden");
    if (resultHero) resultHero.classList.remove("low-conf");
  }

  $("#resChar").textContent = pred.char;
  $("#resRoman").textContent = pred.roman;
  $("#resNepali").textContent = pred.nepali_name;
  $("#resType").textContent = pred.type;
  $("#resConf").textContent = `${pred.confidence}% (${pred.confidence_level})`;
  $("#resConfBar").style.width = `${pred.confidence}%`;
  $("#resWord").textContent = pred.example_word || "–";
  $("#resNote").textContent = pred.note || "";

  const bar = $("#resConfBar");
  if (pred.confidence_level === "high") bar.style.background = "linear-gradient(90deg, var(--success), #2d6a34)";
  else if (pred.confidence_level === "medium") bar.style.background = "linear-gradient(90deg, var(--warn), #d97706)";
  else bar.style.background = "linear-gradient(90deg, var(--danger), #dc2626)";

  if (data.analysis) {
    $("#resUncertainty").textContent = data.analysis.uncertainty;
    $("#resEntropy").textContent = data.analysis.normalised_entropy;
    $("#resInverted").textContent = data.analysis.was_inverted ? "Yes" : "No";
    const analysisBar = $("#analysisBar");
    if (analysisBar) analysisBar.style.display = "flex";
  }

  const confBox = $("#confusionBox");
  const confChips = $("#confusionChips");
  if (data.confusions && data.confusions.length > 0 && pred.confidence < 85) {
    confChips.innerHTML = data.confusions
      .map((c) => `<span class="confusion-chip">${c.char} (${c.roman})</span>`).join("");
    confBox.classList.remove("hidden");
  } else { confBox.classList.add("hidden"); }

  const top3Grid = $("#top3Grid");
  top3Grid.innerHTML = data.top3
    .map((item, i) => `
      <div class="top3-card ${i === 0 ? 'rank-1' : ''}">
        <div class="top3-rank">#${i + 1}</div>
        <div class="top3-char">${item.char}</div>
        <div class="top3-roman">${item.roman} · ${item.type}</div>
        <div class="top3-conf-bar"><div class="top3-conf-fill" style="width:${item.confidence}%"></div></div>
        <div class="top3-conf-text">${item.confidence}%</div>
      </div>`).join("");

  if (data.top5) {
    const top5Grid = $("#top5Grid");
    top5Grid.innerHTML = data.top5
      .map((item, i) => `
        <div class="top3-card ${i === 0 ? 'rank-1' : ''}">
          <div class="top3-rank">#${i + 1}</div>
          <div class="top3-char">${item.char}</div>
          <div class="top3-roman">${item.roman} · ${item.type}</div>
          <div class="top3-conf-bar"><div class="top3-conf-fill" style="width:${item.confidence}%"></div></div>
          <div class="top3-conf-text">${item.confidence}%</div>
        </div>`).join("");
  }

  const gradcamSection = $("#gradcamSection");
  if (data.gradcam) {
    $("#gradcamImg").src = data.gradcam;
    $("#ppModel2").src = data.preprocessing.model_input;
    gradcamSection.style.display = "block";
  } else { gradcamSection.style.display = "none"; }

  if (data.preprocessing) {
    $("#ppOriginal").src = data.preprocessing.original;
    $("#ppGray").src = data.preprocessing.grayscale;
    $("#ppProcessed").src = data.preprocessing.processed;
    $("#ppModel").src = data.preprocessing.model_input;
  }

  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ══════════════════════════════════════════════════════════════
//  DISPLAY NOTE RESULTS
// ══════════════════════════════════════════════════════════════
function displayNoteResults(data) {
  const pred = data.prediction;
  const isRecognized = data.recognized !== false;
  noteResultsSection.classList.remove("hidden");
  resultsSection.classList.add("hidden");
  multidigitResultsSection.classList.add("hidden");

  const warnEl = $("#noteNotRecognizedWarn"), warnText = $("#noteNotRecognizedText");
  if (!isRecognized) {
    if (warnEl) { warnEl.classList.remove("hidden"); warnText.textContent = `Not Recognized — ${data.reject_reason || "Low confidence or outside trained dataset."}`; }
  } else { if (warnEl) warnEl.classList.add("hidden"); }

  const noteResValue = $("#noteResValue");
  noteResValue.innerHTML = "";
  if (isRecognized) {
    const valSpan = document.createElement("span"); valSpan.textContent = `₨ ${pred.value}`; noteResValue.appendChild(valSpan);
    const smallEl = document.createElement("small"); smallEl.textContent = "Nepali Rupees"; noteResValue.appendChild(smallEl);
  } else {
    const valSpan = document.createElement("span"); valSpan.textContent = "?"; valSpan.style.color = "var(--danger)"; noteResValue.appendChild(valSpan);
    const smallEl = document.createElement("small"); smallEl.textContent = "Not Recognized"; smallEl.style.color = "var(--danger)"; noteResValue.appendChild(smallEl);
  }

  $("#noteResEnglish").textContent = pred.english_name || "–";
  $("#noteResNepali").textContent = pred.nepali_name || "–";
  $("#noteResNepaliNum").textContent = pred.nepali_numeral || "";
  $("#noteResConf").textContent = `${pred.confidence}% (${pred.confidence_level})`;
  $("#noteResConfBar").style.width = `${pred.confidence}%`;
  $("#noteResColor").textContent = pred.color_hint || "–";

  const methodBadge = $("#noteMethodBadge");
  if (methodBadge) {
    if (!isRecognized) { methodBadge.textContent = "Not Recognized"; methodBadge.className = "result-val tag tag-warn"; }
    else if (data.method === "deep_learning") { methodBadge.textContent = "Deep Learning"; methodBadge.className = "result-val tag"; }
    else { methodBadge.textContent = "Color Analysis (train model for better accuracy)"; methodBadge.className = "result-val tag tag-warn"; }
  }

  const bar = $("#noteResConfBar");
  if (!isRecognized) bar.style.background = "linear-gradient(90deg, var(--danger), #dc2626)";
  else if (pred.confidence_level === "high") bar.style.background = "linear-gradient(90deg, var(--success), #2d6a34)";
  else if (pred.confidence_level === "medium") bar.style.background = "linear-gradient(90deg, var(--warn), #d97706)";
  else bar.style.background = "linear-gradient(90deg, var(--danger), #dc2626)";

  const noteHero = $("#noteResultHero");
  if (noteHero) noteHero.style.borderColor = isRecognized ? "" : "var(--danger)";

  const noteTop3 = $("#noteTop3Grid");
  if (noteTop3 && data.top3) {
    const top3Label = isRecognized ? "" : ' <span style="color:var(--danger);font-size:.7rem">(low confidence)</span>';
    noteTop3.innerHTML = data.top3
      .map((item, i) => `
        <div class="top3-card ${i === 0 && isRecognized ? 'rank-1' : ''}" ${!isRecognized ? 'style="opacity:.65"' : ''}>
          <div class="top3-rank">#${i + 1}${!isRecognized && i === 0 ? top3Label : ''}</div>
          <div class="top3-char">₨${item.value}</div>
          <div class="top3-roman">${item.english_name}</div>
          <div class="top3-conf-bar"><div class="top3-conf-fill" style="width:${item.confidence}%"></div></div>
          <div class="top3-conf-text">${item.confidence}%</div>
        </div>`).join("");
  }

  if (data.voice) {
    const voiceText = data.voice[noteVoiceMode] || data.voice.english;
    const lang = noteVoiceMode === "nepali" ? "hi-IN" : "en-US";
    speak(voiceText, lang);
  }

  noteResultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ══════════════════════════════════════════════════════════════
//  DISPLAY MULTI-DIGIT RESULTS
// ══════════════════════════════════════════════════════════════
function displayMultidigitResults(data) {
  multidigitResultsSection.classList.remove("hidden");
  resultsSection.classList.add("hidden");
  noteResultsSection.classList.add("hidden");

  // Confidence warning
  const warnBar = $("#multidigitWarnBar"), warnText = $("#multidigitWarnText");
  if (warnBar && warnText) {
    if (data.overall_confidence < 50) {
      warnBar.classList.remove("hidden");
      warnText.textContent = `Low overall confidence (${data.overall_confidence}%) – try drawing with bolder, clearer strokes and leave small gaps between digits.`;
    } else { warnBar.classList.add("hidden"); }
  }

  // Main number
  const resDeva = $("#multidigitResDeva"), resArabic = $("#multidigitResArabic");
  if (resDeva) resDeva.textContent = data.full_number || "–";
  if (resArabic) resArabic.textContent = data.full_number_arabic ? `(${data.full_number_arabic})` : "";

  const resCount = $("#multidigitResCount");
  if (resCount) resCount.textContent = `${data.digit_count || 0} digit${data.digit_count !== 1 ? "s" : ""}`;

  const resConf = $("#multidigitResConf");
  if (resConf) resConf.textContent = `${data.overall_confidence}% (${data.overall_confidence_level})`;

  const confBar = $("#multidigitResConfBar");
  if (confBar) {
    confBar.style.width = `${data.overall_confidence}%`;
    const lvl = data.overall_confidence_level;
    if (lvl === "high") confBar.style.background = "linear-gradient(90deg, var(--success), #2d6a34)";
    else if (lvl === "medium") confBar.style.background = "linear-gradient(90deg, var(--warn), #d97706)";
    else confBar.style.background = "linear-gradient(90deg, var(--danger), #dc2626)";
  }

  // Per-segment warnings
  const warningsBox = $("#multidigitWarningsBox");
  if (warningsBox) {
    if (data.warnings && data.warnings.length > 0) {
      warningsBox.classList.remove("hidden");
      warningsBox.innerHTML = data.warnings
        .map(w => `<div class="multidigit-warning-item">&#9888; ${w}</div>`).join("");
    } else { warningsBox.classList.add("hidden"); }
  }

  // Segmentation image
  const segImg = $("#multidigitSegImg");
  if (segImg) {
    if (data.segmentation_image) { segImg.src = data.segmentation_image; segImg.style.display = "block"; }
    else { segImg.style.display = "none"; }
  }

  // Per-digit breakdown
  const digitsGrid = $("#multidigitDigitsGrid");
  if (digitsGrid) {
    if (data.segments && data.segments.length > 0) {
      digitsGrid.innerHTML = data.segments.map((seg, i) => {
        const confColor = seg.confidence >= 85 ? "var(--success)"
                        : seg.confidence >= 50 ? "var(--warn)"
                        : "var(--danger)";
        const top3html = (seg.top3 || []).map((t, ti) =>
          `<span class="digit-top3-item ${ti === 0 ? 'digit-top3-best' : ''}">${t.char} <small>${t.confidence}%</small></span>`
        ).join("");
        return `
          <div class="multidigit-digit-card">
            <div class="digit-card-pos">Digit ${i + 1}</div>
            ${seg.model_input_preview ? `<img class="digit-model-preview" src="${seg.model_input_preview}" alt="32×32 model input" title="What the model received (32×32, upscaled)" />` : ''}
            <div class="digit-card-char">${seg.char}</div>
            <div class="digit-card-arabic">${seg.roman}</div>
            <div class="top3-conf-bar" style="margin:.4rem 0">
              <div class="top3-conf-fill" style="width:${seg.confidence}%;background:${confColor}"></div>
            </div>
            <div class="digit-card-conf">${seg.confidence}%</div>
            <div class="digit-top3">${top3html}</div>
          </div>`;
      }).join("");
    } else {
      digitsGrid.innerHTML = "<p style='color:var(--text-muted);font-size:.85rem'>No digit data available.</p>";
    }
  }

  multidigitResultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ══════════════════════════════════════════════════════════════
//  VOICE OUTPUT (Browser TTS)
// ══════════════════════════════════════════════════════════════
if (voiceBtn) {
  voiceBtn.addEventListener("click", async () => {
    if (!lastPredictionData || !lastPredictionData.prediction) { showToast("No prediction to read", "info"); return; }
    const pred = lastPredictionData.prediction;
    try {
      const res = await fetch(`${API_BASE}/voice-data/${pred.raw_label}`);
      if (res.ok) { const data = await res.json(); speak(data.speech_text, "en-US"); return; }
    } catch (e) {}
    let text = `The character is ${pred.char}, romanised as ${pred.roman}.`;
    if (pred.nepali_name) text += ` In Nepali, it is called ${pred.nepali_name}.`;
    if (pred.example_word) text += ` An example word is ${pred.example_word}.`;
    speak(text, "en-US");
  });
}

if (noteVoiceBtn) {
  noteVoiceBtn.addEventListener("click", () => {
    if (!lastNotePredictionData || !lastNotePredictionData.voice) { showToast("No note prediction to read", "info"); return; }
    const voiceData = lastNotePredictionData.voice;
    speak(voiceData[noteVoiceMode] || voiceData.english, noteVoiceMode === "nepali" ? "hi-IN" : "en-US");
  });
}

if (noteReplayBtn) { noteReplayBtn.addEventListener("click", () => replayLastVoice()); }

function replayLastVoice() {
  if (lastSpokenText) speak(lastSpokenText, lastSpokenLang || "en-US");
  else showToast("Nothing to replay", "info");
}

function speak(text, lang) {
  if (!("speechSynthesis" in window)) { showToast("Text-to-speech not supported in this browser", "error"); return; }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.9; utterance.pitch = 1;
  if (lang) utterance.lang = lang;
  if (selectedVoice) { utterance.voice = selectedVoice; }
  else if (lang === "hi-IN" || lang === "ne-NP") {
    const voices = window.speechSynthesis.getVoices();
    const nepaliVoice = voices.find(v => v.lang.startsWith("ne") || v.lang.startsWith("hi"));
    if (nepaliVoice) utterance.voice = nepaliVoice;
  }
  window.speechSynthesis.speak(utterance);
  lastSpokenText = text; lastSpokenLang = lang;
  showToast("Speaking...", "info", 2000);
}

$$(".voice-pill").forEach((pill) => {
  pill.addEventListener("click", () => {
    $$(".voice-pill").forEach((p) => p.classList.remove("active"));
    pill.classList.add("active");
    noteVoiceMode = pill.dataset.vmode;
  });
});

// ══════════════════════════════════════════════════════════════
//  FEEDBACK
// ══════════════════════════════════════════════════════════════
if (feedbackCorrectBtn) feedbackCorrectBtn.addEventListener("click", () => submitFeedback(true));
if (feedbackWrongBtn) feedbackWrongBtn.addEventListener("click", () => submitFeedback(false));

async function submitFeedback(isCorrect) {
  if (!lastPredictionData) return;
  const pred = lastPredictionData.prediction;
  try {
    await fetch(`${API_BASE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ predicted_label: pred.raw_label, is_correct: isCorrect, confidence: pred.confidence, input_type: activeTab }),
    });
    showToast(isCorrect ? "Marked as correct!" : "Marked as incorrect. Thanks for feedback!", "success");
  } catch (e) { showToast("Feedback saved locally", "info"); }
}

// ══════════════════════════════════════════════════════════════
//  HISTORY
// ══════════════════════════════════════════════════════════════
function addToHistory(data, inputType) {
  const pred = data.prediction;
  history.unshift({
    timestamp: new Date().toISOString(), inputType,
    char: pred.char, roman: pred.roman, nepali_name: pred.nepali_name,
    type: pred.type, confidence: pred.confidence,
    uncertainty: data.analysis ? data.analysis.uncertainty : "–",
  });
  renderHistory();
}

function addMultidigitToHistory(data, inputType) {
  history.unshift({
    timestamp: new Date().toISOString(), inputType,
    char: data.full_number || "–", roman: data.full_number_arabic || "–",
    nepali_name: `${data.digit_count || 0} digits`, type: "multi-digit",
    confidence: data.overall_confidence || 0,
    uncertainty: data.overall_confidence_level || "–",
  });
  renderHistory();
}

function renderHistory() {
  if (history.length === 0) {
    historyEmpty.style.display = "block";
    historyList.querySelectorAll(".history-item").forEach((el) => el.remove());
    return;
  }
  historyEmpty.style.display = "none";
  historyList.querySelectorAll(".history-item").forEach((el) => el.remove());
  history.forEach((item) => {
    const el = document.createElement("div");
    el.className = "history-item";
    const time = new Date(item.timestamp).toLocaleTimeString();
    el.innerHTML = `
      <div class="history-thumb">${item.char}</div>
      <div class="history-info">
        <div class="history-main">${item.char} – ${item.roman}</div>
        <div class="history-meta">${item.type} · ${item.inputType} · ${time}</div>
      </div>
      <div class="history-conf">${item.confidence}%</div>`;
    historyList.appendChild(el);
  });
}

clearHistBtn.addEventListener("click", () => { history = []; renderHistory(); showToast("History cleared", "info"); });

exportCsvBtn.addEventListener("click", () => {
  if (history.length === 0) { showToast("No history to export", "info"); return; }
  const headers = ["timestamp", "input_type", "character", "transliteration", "nepali_name", "type", "confidence", "uncertainty"];
  const rows = history.map((h) => [h.timestamp, h.inputType, h.char, h.roman, h.nepali_name, h.type, h.confidence, h.uncertainty]);
  const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
  const blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = `devanagari_predictions_${Date.now()}.csv`; a.click();
  URL.revokeObjectURL(url);
  showToast("CSV exported", "success");
});

// ══════════════════════════════════════════════════════════════
//  RESET
// ══════════════════════════════════════════════════════════════
resetBtn.addEventListener("click", () => {
  fileInput.value = "";
  uploadPreview.src = "";
  uploadPreviewWrap.classList.remove("show");

  initCanvas();
  if (ctxMulti) { ctxMulti.fillStyle = "#ffffff"; ctxMulti.fillRect(0, 0, drawCanvasMulti.width, drawCanvasMulti.height); }

  capturedImageData = null;
  cameraPreview.src = ""; cameraPreviewWrap.classList.remove("show");
  if (noteFileInput) noteFileInput.value = "";
  if (notePreview) notePreview.src = "";
  if (notePreviewWrap) notePreviewWrap.classList.remove("show");

  resultsSection.classList.add("hidden");
  noteResultsSection.classList.add("hidden");
  multidigitResultsSection.classList.add("hidden");
  loadingOverlay.classList.add("hidden");

  lastPredictionData = null;
  lastNotePredictionData = null;

  // Reset upload mode to single character
  uploadMode = "single";
  $$("[data-umode]").forEach((p) => p.classList.toggle("active", p.dataset.umode === "single"));
  const uploadHint = $("#uploadMultiHint");
  if (uploadHint) uploadHint.classList.add("hidden");

  // Reset camera mode to single character
  cameraMode = "single";
  $$("[data-cmode]").forEach((p) => p.classList.toggle("active", p.dataset.cmode === "single"));
  const cameraHint = $("#cameraMultiHint");
  if (cameraHint) cameraHint.classList.add("hidden");

  showToast("Reset", "info");
});

// ══════════════════════════════════════════════════════════════
//  SAMPLE IMAGE
// ══════════════════════════════════════════════════════════════
sampleBtn.addEventListener("click", () => {
  const offCanvas = document.createElement("canvas");
  offCanvas.width = 128; offCanvas.height = 128;
  const offCtx = offCanvas.getContext("2d");
  offCtx.fillStyle = "#ffffff"; offCtx.fillRect(0, 0, 128, 128);
  offCtx.fillStyle = "#111111"; offCtx.font = "bold 80px serif";
  offCtx.textAlign = "center"; offCtx.textBaseline = "middle";
  const chars = "अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह";
  const pick = chars[Math.floor(Math.random() * chars.length)];
  offCtx.fillText(pick, 64, 70);
  const dataUrl = offCanvas.toDataURL("image/png");

  tabPills.forEach((p) => p.classList.remove("active"));
  tabPanels.forEach((p) => p.classList.remove("active"));
  document.querySelector('[data-tab="upload"]').classList.add("active");
  $("#panel-upload").classList.add("active");
  activeTab = "upload";

  resultsSection.classList.add("hidden");
  noteResultsSection.classList.add("hidden");
  multidigitResultsSection.classList.add("hidden");

  uploadPreview.src = dataUrl;
  uploadPreviewWrap.classList.add("show");

  fetch(dataUrl).then(r => r.blob()).then(blob => {
    const file = new File([blob], "sample.png", { type: "image/png" });
    const dt = new DataTransfer(); dt.items.add(file); fileInput.files = dt.files;
  });

  showToast(`Sample "${pick}" loaded – click Predict!`, "info");
});

// ══════════════════════════════════════════════════════════════
//  DASHBOARD STATS
// ══════════════════════════════════════════════════════════════
async function refreshStats() {
  try {
    const res = await fetch(`${API_BASE}/session-stats`);
    const data = await res.json();
    statTotal.textContent = data.total_predictions;
    statAvg.textContent = data.average_confidence > 0 ? `${data.average_confidence}%` : "–";
    statHigh.textContent = data.highest_confidence > 0 ? `${data.highest_confidence}% ${data.highest_confidence_char}` : "–";
    statLast.textContent = data.last_prediction ? `${data.last_prediction.char} (${data.last_prediction.confidence}%)` : "–";
  } catch (e) {}
}

// ══════════════════════════════════════════════════════════════
//  CONFIDENCE TREND CHART
// ══════════════════════════════════════════════════════════════
function drawConfidenceTrend(dataPoints) {
  const canvas = $("#confTrendCanvas");
  if (!canvas) return;
  const c = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const pad = { top: 20, right: 20, bottom: 30, left: 50 };
  const plotW = W - pad.left - pad.right, plotH = H - pad.top - pad.bottom;

  const style = getComputedStyle(document.documentElement);
  const textColor = style.getPropertyValue("--text-primary").trim() || "#333";
  const lineColor = style.getPropertyValue("--accent").trim() || "#2563eb";
  const gridColor = style.getPropertyValue("--border").trim() || "#e5e7eb";

  c.clearRect(0, 0, W, H);
  c.strokeStyle = gridColor; c.lineWidth = 0.5;
  for (let y = 0; y <= 100; y += 25) {
    const py = pad.top + plotH - (y / 100) * plotH;
    c.beginPath(); c.moveTo(pad.left, py); c.lineTo(pad.left + plotW, py); c.stroke();
    c.fillStyle = textColor; c.font = "11px DM Sans, sans-serif"; c.textAlign = "right";
    c.fillText(`${y}%`, pad.left - 8, py + 4);
  }
  c.fillStyle = textColor; c.font = "11px DM Sans, sans-serif"; c.textAlign = "center";
  c.fillText("Prediction #", pad.left + plotW / 2, H - 5);

  if (!dataPoints || dataPoints.length < 2) {
    c.fillStyle = textColor;
    c.font = "13px DM Sans, sans-serif";
    c.textAlign = "center";
    c.fillText("No prediction history yet", W / 2, H / 2);
    return;
  }
  const stepX = plotW / (dataPoints.length - 1);
  c.strokeStyle = lineColor; c.lineWidth = 2;
  c.beginPath();
  dataPoints.forEach((val, i) => {
    const x = pad.left + i * stepX, y = pad.top + plotH - (val / 100) * plotH;
    if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
  });
  c.stroke();

  const dotCount = Math.min(dataPoints.length, 20);
  for (let i = dataPoints.length - dotCount; i < dataPoints.length; i++) {
    const x = pad.left + i * stepX, y = pad.top + plotH - (dataPoints[i] / 100) * plotH;
    c.beginPath(); c.arc(x, y, 3, 0, Math.PI * 2); c.fillStyle = lineColor; c.fill();
  }
}

function recordConfidence(conf, label, source) {
  const v = parseFloat(conf);
  if (isNaN(v)) return;
  confidenceHistory.push({ conf: v, label: label || "–", source: source || "unknown" });
  if (confidenceHistory.length > 100) confidenceHistory = confidenceHistory.slice(-100);
  drawConfidenceTrend(confidenceHistory.map((e) => e.conf));
}

// ══════════════════════════════════════════════════════════════
//  EVALUATION DASHBOARD
// ══════════════════════════════════════════════════════════════
async function loadEvaluation() {
  try {
    const res = await fetch(`${API_BASE}/evaluation-summary`);
    if (!res.ok) { setEvalFallback(); return; }
    const data = await res.json();
    if (data && data.overall_accuracy > 0) setEvalCards(data);
    else setEvalFallback();
  } catch (e) { setEvalFallback(); }
}

function setEvalCards(data) {
  if (data.overall_accuracy > 0) $("#evalAccuracy").textContent = `${(data.overall_accuracy * 100).toFixed(2)}%`;
  if (data.macro_precision > 0) $("#evalPrecision").textContent = `${(data.macro_precision * 100).toFixed(2)}%`;
  if (data.macro_recall > 0) $("#evalRecall").textContent = `${(data.macro_recall * 100).toFixed(2)}%`;
  if (data.macro_f1 > 0) $("#evalF1").textContent = `${(data.macro_f1 * 100).toFixed(2)}%`;
  if (data.source === "estimated_from_training_history") {
    const note = document.createElement("p"); note.className = "eval-note";
    note.textContent = "Approximate values from training history. Run ml/evaluate.py for precise metrics.";
    const evalCards = $("#evalCards");
    if (evalCards && !evalCards.querySelector(".eval-note")) evalCards.appendChild(note);
  }
}

function setEvalFallback() {
  ["#evalAccuracy", "#evalPrecision", "#evalRecall", "#evalF1"].forEach(s => {
    const el = $(s); if (el) el.textContent = "Not available";
  });
}

// ══════════════════════════════════════════════════════════════
//  MODEL INFO
// ══════════════════════════════════════════════════════════════
async function loadModelInfo() {
  try {
    const res = await fetch(`${API_BASE}/model-info`);
    if (!res.ok) return;
    const data = await res.json();
    const badge = $("#modelParamsBadge");
    if (badge && data.total_parameters) badge.textContent = `${(data.total_parameters / 1000).toFixed(0)}K params`;
    const noteBadge = $("#noteModelBadge");
    if (noteBadge) {
      if (data.note_reader_available)
        noteBadge.textContent = data.note_model_available ? "Note Reader: Ready (DL)" : "Note Reader: Ready (Fallback)";
      else noteBadge.textContent = "Note Reader: Not available";
    }
  } catch (e) {}
}

// ══════════════════════════════════════════════════════════════
//  API STATUS CHECK
// ══════════════════════════════════════════════════════════════
async function checkAPI() {
  const badge = $("#apiStatusBadge");
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      badge.innerHTML = '<span class="badge-dot"></span> API Online'; badge.classList.add("online");
      const data = await res.json();
      const noteBadge = $("#noteModelBadge");
      if (noteBadge) {
        if (data.note_reader_available)
          noteBadge.textContent = data.note_model_loaded ? "Note Reader: Ready (DL)" : "Note Reader: Ready (Fallback)";
        else noteBadge.textContent = "Note Reader: Not available";
      }
    } else { badge.innerHTML = '<span class="badge-dot"></span> API Error'; badge.classList.add("offline"); }
  } catch { badge.innerHTML = '<span class="badge-dot"></span> API Offline'; badge.classList.add("offline"); }
}

// ══════════════════════════════════════════════════════════════
//  METRICS IMAGES
// ══════════════════════════════════════════════════════════════
async function loadMetrics() {
  try {
    const res = await fetch(`${API_BASE}/metrics`);
    if (!res.ok) { hideAllMetricImages(); return; }
    const data = await res.json();
    setMetricImage("#trainingPlotImg", data.training_plot);
    setMetricImage("#confMatImg", data.confusion_matrix);
    setMetricImage("#perClassAccImg", data.per_class_accuracy);
    setMetricImage("#lrScheduleImg", data.lr_schedule);
  } catch (e) { hideAllMetricImages(); }
}

function setMetricImage(selector, path) {
  const img = $(selector); if (!img) return;
  if (path) {
    img.src = `${API_BASE}${path}`; img.style.display = "";
    img.onerror = () => { img.style.display = "none"; showMetricPlaceholder(img); };
    const existingPh = img.parentElement.querySelector(".metric-placeholder");
    if (existingPh) existingPh.remove();
    img.parentElement.style.display = "";
  } else { img.style.display = "none"; showMetricPlaceholder(img); }
}

function hideAllMetricImages() {
  ["#trainingPlotImg", "#confMatImg", "#perClassAccImg", "#lrScheduleImg"].forEach(s => {
    const img = $(s); if (!img) return; img.style.display = "none"; showMetricPlaceholder(img);
  });
}

function showMetricPlaceholder(imgEl) {
  if (imgEl.parentElement.querySelector(".metric-placeholder")) return;
  const placeholder = document.createElement("div");
  placeholder.className = "metric-placeholder";
  placeholder.textContent = "Not available yet — run ml/evaluate.py to generate";
  imgEl.parentElement.appendChild(placeholder);
}

// ══════════════════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════════════════
checkAPI();
loadMetrics();
loadEvaluation();
loadModelInfo();
refreshStats();
drawConfidenceTrend([]);
