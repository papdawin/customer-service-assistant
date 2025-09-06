import io
import os
import time
import base64
import wave
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from piper import PiperVoice  # provided by piper-tts

VOICE_PATH = os.getenv("PIPER_VOICE", "/voices/hu_HU-berta-medium.onnx")
VOICE_JSON = os.getenv("PIPER_VOICE_JSON", "/voices/hu_HU-berta-medium.onnx.json")
USE_CUDA = os.getenv("TTS_DEVICE", "cpu").lower() == "cuda"

app = FastAPI(title="TTS (Piper)")

@app.on_event("startup")
def _load_voice():
    app.state.voice = PiperVoice.load(VOICE_PATH, config_path=VOICE_JSON, use_cuda=USE_CUDA)

class SpeakRequest(BaseModel):
    text: str
    format: str | None = "json"  # "json" (base64) or "wav"

@app.get("/health")
def health():
    return {"status": "ok", "voice": os.path.basename(VOICE_PATH), "device": "cuda" if USE_CUDA else "cpu"}

@app.post("/speak")
def speak(req: SpeakRequest):
    text = (req.text or "").strip()
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)

    t0 = time.perf_counter()
    buf = io.BytesIO()

    # ✅ Write a proper WAV into the buffer
    with wave.open(buf, "wb") as wav_file:
        app.state.voice.synthesize_wav(text, wav_file)

    wav_bytes = buf.getvalue()
    dur_ms = round((time.perf_counter() - t0) * 1000, 1)

    if not wav_bytes:
        # Helpful diagnostics if something still goes wrong
        return JSONResponse({"error": "tts produced zero bytes"}, status_code=500)

    if (req.format or "json") == "json":   # NOTE: case-sensitive
        return JSONResponse({
            "audio_mime": "audio/wav",
            "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
            "timings": {"total_ms": dur_ms}
        })

    headers = {"Server-Timing": f"tts;dur={dur_ms}"}
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)
