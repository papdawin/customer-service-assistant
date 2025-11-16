import os
import tempfile
import time
from typing import Optional

import torch
import whisper
from fastapi import FastAPI, UploadFile, File, Form, Response

STT_MODEL_ID = os.getenv("STT_MODEL_ID", "openai/whisper-large-v3")
DEFAULT_LANGUAGE = os.getenv("STT_LANGUAGE", "hu")
STT_DEVICE_ENV = os.getenv("STT_DEVICE", "auto").lower()
STT_DEVICE_ID = os.getenv("STT_DEVICE_ID")
CUDA_VISIBLE_ENV = os.getenv("CUDA_VISIBLE_DEVICES")
STT_COMPUTE_TYPE_ENV = os.getenv("STT_COMPUTE_TYPE", "auto").lower()

def _normalize_model_name(model_id: str) -> str:
    """Allow passing names like 'openai/whisper-large-v3' while loading via whisper lib."""
    if not model_id:
        return "large-v3"
    name = model_id.split("/")[-1]
    if name.startswith("whisper-"):
        name = name[len("whisper-") :]
    return name

MODEL_NAME = _normalize_model_name(STT_MODEL_ID)

def _normalized_cuda_device_id() -> Optional[str]:
    """Map requested GPU id onto the container-visible CUDA ordinals."""
    ident = (STT_DEVICE_ID or "").strip()
    if not ident:
        return None
    if ident.startswith("cuda:"):
        ident = ident.split(":", 1)[1]
    visible = (CUDA_VISIBLE_ENV or "").strip()
    if visible:
        mapping = [token.strip() for token in visible.split(",") if token.strip()]
        if mapping and ident in mapping:
            return str(mapping.index(ident))
    return ident

NORMALIZED_CUDA_ID = _normalized_cuda_device_id()

def pick_device():
    if STT_DEVICE_ENV == "cpu":
        return "cpu"
    if STT_DEVICE_ENV == "cuda":
        if NORMALIZED_CUDA_ID:
            return f"cuda:{NORMALIZED_CUDA_ID}"
        return "cuda"
    # auto: prefer CUDA if available
    try:
        import torch
        if torch.cuda.is_available():
            if NORMALIZED_CUDA_ID:
                return f"cuda:{NORMALIZED_CUDA_ID}"
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"

def pick_dtype(device: str):
    if not device.startswith("cuda"):
        return torch.float32
    if STT_COMPUTE_TYPE_ENV in {"float16", "fp16"}:
        return torch.float16
    if STT_COMPUTE_TYPE_ENV in {"float32", "fp32"}:
        return torch.float32
    # default to fp16 on CUDA
    return torch.float16

DEVICE = pick_device()
DTYPE = pick_dtype(DEVICE)

# Load model once; keep weights in their native dtype so Whisper can manage casting internally
model = whisper.load_model(MODEL_NAME, device=DEVICE)

app = FastAPI(title="STT – Whisper HU")

@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL_NAME,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "default_language": DEFAULT_LANGUAGE,
    }

@app.post("/transcribe")
async def transcribe(response: Response,
                     audio: UploadFile = File(...),
                     language: Optional[str] = Form(DEFAULT_LANGUAGE),
                     beam_size: Optional[int] = Form(5)):
    t0 = time.perf_counter()

    suffix = os.path.splitext(audio.filename or "")[-1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        data = await audio.read()
        tmp.write(data)
        tmp_path = tmp.name

    try:
        t1 = time.perf_counter()
        result = model.transcribe(
            tmp_path,
            language=(language or DEFAULT_LANGUAGE),
            beam_size=beam_size or 5,
            task="transcribe",
            fp16=(DEVICE.startswith("cuda") and DTYPE == torch.float16),
            verbose=False,
        )
        t2 = time.perf_counter()

        text = (result.get("text") or "").strip()
        segments = result.get("segments") or []
        duration = segments[-1]["end"] if segments else None
        detected_lang = result.get("language") or (language or DEFAULT_LANGUAGE)

        transcribe_ms = round((t2 - t1) * 1000, 1)
        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        # Server-Timing header (visible with curl -i / browser devtools)
        response.headers["Server-Timing"] = f"transcribe;dur={transcribe_ms},total;dur={total_ms}"

        return {
            "text": text,
            "lang": detected_lang,
            "duration": duration,
            "timings": {"transcribe_ms": transcribe_ms, "total_ms": total_ms},
        }
    finally:
        try: os.remove(tmp_path)
        except Exception: pass
