import os
import tempfile
import time
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

STT_MODEL_ID = os.getenv("STT_MODEL_ID", "sarpba/whisper-hu-large-v3-turbo-finetuned")
STT_DEVICE_ENV = os.getenv("STT_DEVICE", "auto").lower()
STT_COMPUTE_TYPE_ENV = os.getenv("STT_COMPUTE_TYPE", "auto").lower()

def pick_device():
    if STT_DEVICE_ENV in ("cpu", "cuda"):
        return STT_DEVICE_ENV
    # auto: prefer CUDA if available
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def pick_compute_type(device: str):
    if STT_COMPUTE_TYPE_ENV != "auto":
        return STT_COMPUTE_TYPE_ENV
    # sensible defaults
    return "float16" if device == "cuda" else "int8"

DEVICE = pick_device()
COMPUTE_TYPE = pick_compute_type(DEVICE)

# Load model once
model = WhisperModel(STT_MODEL_ID, device=DEVICE, compute_type=COMPUTE_TYPE)

app = FastAPI(title="STT – Whisper HU")

@app.get("/health")
def health():
    return {"ok": True, "model": STT_MODEL_ID, "device": DEVICE, "compute_type": COMPUTE_TYPE}

@app.post("/transcribe")
async def transcribe(response: Response,
                     audio: UploadFile = File(...),
                     language: Optional[str] = Form("hu"),
                     beam_size: Optional[int] = Form(5)):
    t0 = time.perf_counter()

    suffix = os.path.splitext(audio.filename or "")[-1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        data = await audio.read()
        tmp.write(data)
        tmp_path = tmp.name

    try:
        t1 = time.perf_counter()
        segments, info = model.transcribe(
            tmp_path, language=language, beam_size=beam_size, vad_filter=True, task="transcribe"
        )
        t2 = time.perf_counter()

        text = "".join(seg.text for seg in segments).strip()

        transcribe_ms = round((t2 - t1) * 1000, 1)
        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        # Server-Timing header (visible with curl -i / browser devtools)
        response.headers["Server-Timing"] = f"transcribe;dur={transcribe_ms},total;dur={total_ms}"

        return {
            "text": text,
            "lang": info.language,
            "duration": info.duration,
            "timings": {"transcribe_ms": transcribe_ms, "total_ms": total_ms},
        }
    finally:
        try: os.remove(tmp_path)
        except Exception: pass