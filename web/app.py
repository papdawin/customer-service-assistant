import os
import time
import base64
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

# --------------------------------------------------------------------
# Service endpoints (wired via docker-compose environment variables)
# --------------------------------------------------------------------
STT_URL = os.getenv("STT_URL", "http://stt:5001/transcribe")
RAG_URL = os.getenv("RAG_URL", "http://rag-sonrisa:8080/query")
TTS_URL = os.getenv("TTS_URL", "http://tts:5002/speak")
COMPANY = os.getenv("COMPANY", "Sonrisa")

app = FastAPI(title=f"Voice assistant for {COMPANY}")

# --------------------------------------------------------------------
# Simple front-end: manual Record/Stop + Auto (pause to send) mode
# --------------------------------------------------------------------
INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Voice assistant for</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; max-width: 720px; margin: 40px auto; }
    button { padding: 10px 16px; margin-right: 8px; }
    #status { margin-left: 8px; color: #555; }
    .card { padding: 16px; border: 1px solid #ddd; border-radius: 12px; margin-top: 20px; }
    .label { font-weight: 600; color: #333; margin-bottom: 6px; }
    #answer, #transcript { white-space: pre-wrap; }
    .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  </style>
</head>
<body>
  <h1>{Voice assistant}</h1>
  <p>Use <b>Record/Stop</b> for one-shot, or <b>Auto (pause to send)</b> to detect pauses and submit automatically.</p>

  <div class="row">
    <button id="recordBtn">Record</button>
    <button id="stopBtn" disabled>Stop</button>
    <button id="autoBtn">Auto (pause to send)</button>
    <span id="status"></span>
  </div>

  <div class="card">
    <div class="label">Transcript</div>
    <div id="transcript">—</div>
  </div>

  <div class="card">
    <div class="label">Answer</div>
    <div id="answer">—</div>
  </div>

  <div class="card">
    <div class="label">Audio</div>
    <audio id="ttsAudio" controls></audio>
  </div>

  <pre id="timing" class="card" style="display:none"></pre>

<script>
let stream;                 // shared mic stream
let mediaRecorder;          // manual mode recorder
let chunks = [];

let autoMode = false;       // auto (pause-to-send) toggle
let vadTimer = null;
let audioCtx, analyser, sourceNode;
let capturing = false;      // are we currently recording an utterance in auto mode?
let segRecorder = null;     // per-utterance recorder (auto mode)
let segmentChunks = [];

let activeController = null; // AbortController to cancel in-flight /voice requests

const recordBtn   = document.getElementById('recordBtn');
const stopBtn     = document.getElementById('stopBtn');
const autoBtn     = document.getElementById('autoBtn');
const statusEl    = document.getElementById('status');
const transcriptEl= document.getElementById('transcript');
const answerEl    = document.getElementById('answer');
const ttsAudio    = document.getElementById('ttsAudio');
const timingEl    = document.getElementById('timing');

function stopPlayback() {
  try {
    if (!ttsAudio.paused) {
      ttsAudio.pause();
      ttsAudio.currentTime = 0;
    }
  } catch {}
}

async function startMicIfNeeded() {
  if (stream) return;
  statusEl.textContent = "Requesting microphone…";
  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
}

function showTimings(data) {
  const t = data.timings || {};
  const lines = [
    `STT HTTP: ${t.stt_http_ms ?? "—"} ms (inner: ${t.stt_inner_ms ?? "—"} ms)`,
    `RAG HTTP: ${t.rag_http_ms ?? "—"} ms (retrieval: ${t.rag_retrieval_ms ?? "—"} ms, llm: ${t.rag_llm_ms ?? "—"} ms, inner total: ${t.rag_inner_total_ms ?? "—"} ms)`,
    `TTS HTTP: ${t.tts_http_ms ?? "—"} ms (inner: ${t.tts_inner_ms ?? "—"} ms)`,
    `End-to-end: ${t.end_to_end_ms ?? "—"} ms`
  ].join("\\n");
  timingEl.style.display = "block";
  timingEl.textContent = "Timings\\n" + lines;
}

async function sendClip(blob) {
  // Abort any previous /voice call (e.g., user starts speaking again)
  if (activeController) activeController.abort();
  activeController = new AbortController();

  const form = new FormData();
  form.append('audio', blob, 'clip.webm');

  try {
    const res = await fetch('/voice', { method: 'POST', body: form, signal: activeController.signal });
    const data = await res.json();

    transcriptEl.textContent = data.transcript || "(empty)";
    answerEl.textContent = data.answer || "(no answer)";

    // Play TTS if present
    if (data.audio_b64 && data.audio_mime) {
      ttsAudio.src = `data:${data.audio_mime};base64,${data.audio_b64}`;
      try { await ttsAudio.play(); } catch (_) {}
    }
    showTimings(data);
  } catch (err) {
    if (err.name === "AbortError") {
      // Request was cancelled due to new utterance starting — this is expected.
      statusEl.textContent = "aborted (new speech)";
      return;
    }
    statusEl.textContent = "Error: " + err;
  }
}

/* ---------------- Manual mode ----------------- */
recordBtn.onclick = async () => {
  try {
    await startMicIfNeeded();
    stopPlayback();
    if (activeController) { activeController.abort(); activeController = null; }

    chunks = [];
    transcriptEl.textContent = "—";
    answerEl.textContent = "—";
    ttsAudio.src = "";
    const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus' : 'audio/webm';
    mediaRecorder = new MediaRecorder(stream, { mimeType: mime });
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    mediaRecorder.onstop = async () => {
      const blob = new Blob(chunks, { type: 'audio/webm' });
      statusEl.textContent = "Processing…";
      await sendClip(blob);
      statusEl.textContent = "Done.";
    };
    mediaRecorder.start();
    statusEl.textContent = "Recording…";
    recordBtn.disabled = true;
    stopBtn.disabled   = false;
    autoBtn.disabled   = true;
  } catch (e) {
    statusEl.textContent = "Mic error: " + e;
  }
};

stopBtn.onclick = () => {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  recordBtn.disabled = false;
  stopBtn.disabled   = true;
  autoBtn.disabled   = false;
};

/* ---------------- Auto (pause-to-send) mode ----------------- */
autoBtn.onclick = async () => {
  if (!autoMode) {
    await startAuto();
  } else {
    await stopAuto();
  }
};

async function startAuto() {
  await startMicIfNeeded();

  // WebAudio analyser for simple RMS-based VAD
  audioCtx   = new (window.AudioContext || window.webkitAudioContext)();
  sourceNode = audioCtx.createMediaStreamSource(stream);
  analyser   = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  sourceNode.connect(analyser);

  transcriptEl.textContent = "—";
  answerEl.textContent = "—";
  ttsAudio.src = "";
  autoMode = true;
  autoBtn.textContent = "Auto (listening…)";
  statusEl.textContent = "auto: idle";
  recordBtn.disabled = true;
  stopBtn.disabled   = true;

  const data = new Uint8Array(analyser.fftSize);
  const SILENCE_THRESHOLD = 0.010;  // adjust for room/mic
  const SILENCE_MS        = 600;    // pause length to end utterance
  const TALK_ARM_MS       = 20;    // small debounce before starting
  // gyorsabb vizsgálat
  const VAD_PERIOD_MS   = 20;

  // Float time-domain
  const data = new Float32Array(analyser.fftSize);

  // 300–400 ms előpuffer (~20ms/frame esetén 15–20 frame)
  const PRE_FRAMES = Math.ceil(400 / VAD_PERIOD_MS);
  let preBuffer = []; // ide Float32Array frame-ek mennek

  let silenceStart = performance.now();
  let voicedStart  = null;

  vadTimer = setInterval(() => {
    analyser.getByteTimeDomainData(data);

    // compute RMS around 128
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      const v = (data[i] - 128) / 128.0;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / data.length);
    const now = performance.now();

        // körkörös előpuffer karbantartása
    preBuffer.push(Float32Array.from(data));
    if (preBuffer.length > PRE_FRAMES) preBuffer.shift();

    if (rms > SILENCE_THRESHOLD) {
      // voice present
      if (!capturing) {
        if (voicedStart === null) voicedStart = now;
        if (now - voicedStart > TALK_ARM_MS) {
          onVoiceStart();
          capturing = true;
        }
      }
      silenceStart = now;
    } else {
      // silence
      voicedStart = null;
      if (capturing && (now - silenceStart) > SILENCE_MS) {
        onVoiceEnd();
        capturing = false;
      }
    }
  }, VAD_PERIOD_MS);
}

async function stopAuto() {
  autoMode = false;
  autoBtn.textContent = "Auto (pause to send)";
  statusEl.textContent = "idle";
  try {
    if (vadTimer) { clearInterval(vadTimer); vadTimer = null; }
    if (segRecorder && segRecorder.state !== 'inactive') segRecorder.stop();
    segRecorder = null;
    segmentChunks = [];
    if (audioCtx) await audioCtx.close();
  } catch {}
  recordBtn.disabled = false;
  stopBtn.disabled   = true;
}

function onVoiceStart() {
  stopPlayback();                 // stop any TTS audio immediately
  if (activeController) { activeController.abort(); activeController = null; }

  segmentChunks = [];
  const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    ? 'audio/webm;codecs=opus' : 'audio/webm';
  segRecorder = new MediaRecorder(stream, { mimeType: mime });

  segRecorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) segmentChunks.push(e.data); };

  segRecorder.onstop = async () => {
    if (segmentChunks.length === 0) return;
    const blob = new Blob(segmentChunks, { type: 'audio/webm' });
    segmentChunks = [];
    statusEl.textContent = "auto: processing…";
    await sendClip(blob);
    statusEl.textContent = "auto: idle";
  };

  segRecorder.start();            // single full blob per utterance
  statusEl.textContent = "auto: recording…";
}

function onVoiceEnd() {
  if (segRecorder && segRecorder.state !== 'inactive') segRecorder.stop();
  segRecorder = null;
}
</script>
</body>
</html>
"""

# --------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML

@app.get("/health")
def health():
    return {"ok": True, "stt_url": STT_URL, "rag_url": RAG_URL, "tts_url": TTS_URL}

@app.post("/voice")
async def voice(audio: UploadFile = File(...)):
    """
    1) Send audio to STT
    2) Send transcript to RAG
    3) Send answer to TTS
    4) Return transcript + answer + audio (base64) + timings
    """
    T0 = time.perf_counter()

    # ---- 1) STT ----
    s0 = time.perf_counter()
    stt_form = {"language": (None, "hu"), "beam_size": (None, "5")}
    files = {
        "audio": (audio.filename or "clip.webm",
                  await audio.read(),
                  audio.content_type or "audio/webm")
    }
    stt_resp = requests.post(STT_URL, files=files, data=stt_form, timeout=120)
    s1 = time.perf_counter()
    stt_resp.raise_for_status()
    stt = stt_resp.json()
    transcript = (stt.get("text") or "").strip()
    stt_http_ms = round((s1 - s0) * 1000, 1)
    # accept either "total_ms" or "transcribe_ms", depending on STT server
    stt_timings = stt.get("timings") or {}
    stt_inner_ms = stt_timings.get("total_ms") or stt_timings.get("transcribe_ms")

    # ---- 2) RAG ----
    answer = ""
    rag_http_ms = rag_retrieval_ms = rag_llm_ms = rag_total_ms = None
    if transcript:
        r0 = time.perf_counter()
        rag_resp = requests.post(RAG_URL, json={"question": transcript}, timeout=120)
        r1 = time.perf_counter()
        rag_resp.raise_for_status()
        rag = rag_resp.json()
        answer = rag.get("answer") or ""
        rag_http_ms = round((r1 - r0) * 1000, 1)
        inner = rag.get("timings") or {}
        rag_retrieval_ms = inner.get("retrieval_ms")
        rag_llm_ms = inner.get("llm_ms")
        rag_total_ms = inner.get("total_ms")

    # ---- 3) TTS ----
    audio_b64 = None
    audio_mime = None
    tts_http_ms = tts_inner_ms = None
    if answer:
        t0 = time.perf_counter()
        # Your TTS service returns JSON with {"audio_b64": "...", "audio_mime": "audio/wav", "timings": {...}}
        tts_resp = requests.post(TTS_URL, json={"text": answer, "format": "json"}, timeout=120)
        t1 = time.perf_counter()
        tts_resp.raise_for_status()
        tts = tts_resp.json()
        audio_b64 = tts.get("audio_b64")
        audio_mime = tts.get("audio_mime", "audio/wav")
        tts_http_ms = round((t1 - t0) * 1000, 1)
        tts_inner_ms = (tts.get("timings") or {}).get("total_ms")

    total_ms = round((time.perf_counter() - T0) * 1000, 1)

    return JSONResponse({
        "transcript": transcript,
        "answer": answer,
        "audio_b64": audio_b64,
        "audio_mime": audio_mime,
        "timings": {
            "stt_http_ms": stt_http_ms,
            "stt_inner_ms": stt_inner_ms,
            "rag_http_ms": rag_http_ms,
            "rag_retrieval_ms": rag_retrieval_ms,
            "rag_llm_ms": rag_llm_ms,
            "rag_inner_total_ms": rag_total_ms,
            "tts_http_ms": tts_http_ms,
            "tts_inner_ms": tts_inner_ms,
            "end_to_end_ms": total_ms
        }
    })
