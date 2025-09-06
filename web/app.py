import os
import time
import base64
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import json

STT_URL = os.getenv("STT_URL", "http://stt:5001/transcribe")
RAG_URL = os.getenv("RAG_URL", "http://rag-sonrisa:8080/query")
TTS_URL = os.getenv("TTS_URL", "http://tts:5002/speak")  # NEW

app = FastAPI(title="Voice → RAG (Sonrisa)")

INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Voice → RAG (Sonrisa)</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; max-width: 640px; margin: 40px auto; }
    button { padding: 10px 16px; margin-right: 8px; }
    #status { margin-top: 10px; color: #555; }
    #result { margin-top: 16px; white-space: pre-wrap; }
    .card { padding: 16px; border: 1px solid #ddd; border-radius: 12px; margin-top: 20px; }
    .label { font-weight: 600; color: #333; }
    textarea { width: 100%; height: 120px; }
  </style>
</head>
<body>
  <h1>🎙️ Voice → RAG (Sonrisa)</h1>
  <p>Click <b>Record</b>, speak Hungarian, then <b>Stop</b>. We’ll transcribe, ask RAG, then play the answer.</p>
  <div>
    <button id="recordBtn">Record</button>
    <button id="stopBtn" disabled>Stop</button>
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

<script>
let mediaRecorder;
let chunks = [];
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const statusEl = document.getElementById('status');
const transcriptEl = document.getElementById('transcript');
const answerEl = document.getElementById('answer');
const ttsAudio = document.getElementById('ttsAudio');

recordBtn.onclick = async () => {
  chunks = [];
  transcriptEl.textContent = "—";
  answerEl.textContent = "—";
  ttsAudio.src = "";
  statusEl.textContent = "Requesting microphone…";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm';
    mediaRecorder = new MediaRecorder(stream, { mimeType: mime });
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    mediaRecorder.onstop = onStop;
    mediaRecorder.start();
    statusEl.textContent = "Recording…";
    recordBtn.disabled = true;
    stopBtn.disabled = false;
  } catch (e) {
    statusEl.textContent = "Mic error: " + e;
  }
};

stopBtn.onclick = () => {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    statusEl.textContent = "Processing audio…";
  }
  recordBtn.disabled = false;
  stopBtn.disabled = true;
};

async function onStop() {
  const blob = new Blob(chunks, { type: 'audio/webm' });
  const form = new FormData();
  form.append('audio', blob, 'clip.webm');

  try {
    const res = await fetch('/voice', { method: 'POST', body: form });
    const data = await res.json();
    transcriptEl.textContent = data.transcript || "(empty)";
    answerEl.textContent = data.answer || "(no answer)";
    statusEl.textContent = "Done.";

    // Audio (if present)
    if (data.audio_b64 && data.audio_mime) {
      ttsAudio.src = `data:${data.audio_mime};base64,${data.audio_b64}`;
      try { await ttsAudio.play(); } catch (_) {}
    }

    // Show timings
    const t = data.timings || {};
    const timings = [
      `STT HTTP: ${t.stt_http_ms ?? "—"} ms (inner: ${t.stt_inner_ms ?? "—"} ms)`,
      `RAG HTTP: ${t.rag_http_ms ?? "—"} ms (retrieval: ${t.rag_retrieval_ms ?? "—"} ms, llm: ${t.rag_llm_ms ?? "—"} ms, inner total: ${t.rag_inner_total_ms ?? "—"} ms)`,
      `TTS HTTP: ${t.tts_http_ms ?? "—"} ms (inner: ${t.tts_inner_ms ?? "—"} ms)`,
      `End-to-end: ${t.end_to_end_ms ?? "—"} ms`
    ].join("\\n");

    let timingDiv = document.getElementById('timing');
    if (!timingDiv) {
      timingDiv = document.createElement('pre');
      timingDiv.id = 'timing';
      timingDiv.className = 'card';
      document.body.appendChild(timingDiv);
    }
    timingDiv.textContent = "Timings\\n" + timings;

  } catch (err) {
    statusEl.textContent = "Error: " + err;
  }
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML

@app.post("/voice")
async def voice(audio: UploadFile = File(...)):
    T0 = time.perf_counter()

    # 1) STT
    s0 = time.perf_counter()
    stt_form = {"language": (None, "hu"), "beam_size": (None, "5")}
    files = {"audio": (audio.filename, await audio.read(), audio.content_type or "audio/webm")}
    stt_resp = requests.post(STT_URL, files=files, data=stt_form, timeout=120)
    s1 = time.perf_counter()
    stt_resp.raise_for_status()
    stt = stt_resp.json()
    transcript = (stt.get("text") or "").strip()
    stt_http_ms = round((s1 - s0) * 1000, 1)
    stt_inner_ms = (stt.get("timings") or {}).get("total_ms")

    # 2) RAG
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

    # 3) TTS
    audio_b64 = None
    audio_mime = None
    tts_http_ms = tts_inner_ms = None
    if answer:
        t0 = time.perf_counter()
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
