import os
import tempfile
import time
from typing import Optional
import torch
import whisper
from fastapi import FastAPI, UploadFile, File, Form, Response

MODEL_ID = os.getenv("STT_MODEL_ID", "large-v3")
DEFAULT_LANGUAGE = os.getenv("STT_LANGUAGE", "hu")


MODEL_NAME = MODEL_ID
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32

print(f"Loading Whisper model {MODEL_NAME} on {DEVICE} (dtype={DTYPE})", flush=True)
model = whisper.load_model(MODEL_NAME, device=DEVICE)
model.eval()

app = FastAPI(title="STT – Whisper HU")


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME, "device": DEVICE, "dtype": str(DTYPE), "default_language": DEFAULT_LANGUAGE}


@app.post("/transcribe")
async def transcribe(
    response: Response,
    audio: UploadFile = File(...),
    language: Optional[str] = Form(DEFAULT_LANGUAGE),
    beam_size: Optional[int] = Form(5),
):
    t0 = time.perf_counter()

    suffix = os.path.splitext(audio.filename or "")[-1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        t1 = time.perf_counter()
        result = model.transcribe(
            tmp_path,
            language=language or DEFAULT_LANGUAGE,
            beam_size=beam_size or 5,
            task="transcribe",
            fp16=DEVICE.startswith("cuda") and DTYPE == torch.float16,
            verbose=False,
        )
        transcribe_ms = round((time.perf_counter() - t1) * 1000, 1)
        total_ms = round((time.perf_counter() - t0) * 1000, 1)

        response.headers["Server-Timing"] = f"transcribe;dur={transcribe_ms},total;dur={total_ms}"

        # Filter out hallucinations using no_speech_prob
        segments = result.get("segments") or []
        NO_SPEECH_THRESHOLD = 0.5
        avg_no_speech_prob = sum(s.get("no_speech_prob", 0) for s in segments) / len(segments) if segments else 1.0

        text = (result.get("text") or "").strip()
        if avg_no_speech_prob > NO_SPEECH_THRESHOLD:
            text = ""  # Likely not speech, discard transcription

        return {
            "text": text,
            "lang": result.get("language") or language or DEFAULT_LANGUAGE,
            "duration": segments[-1]["end"] if segments else None,
            "timings": {"transcribe_ms": transcribe_ms, "total_ms": total_ms},
            "no_speech_prob": round(avg_no_speech_prob, 3),
        }
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
