import audioop
import base64
import json
import os
import time
import uuid
import wave
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import httpx
from fastapi import (
    FastAPI,
    File,
    Form,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# --------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------
STT_URL = os.getenv("STT_URL", "http://stt:5001/transcribe")
RAG_URL = os.getenv("RAG_URL", "http://rag-sonrisa:8080/query")
TTS_URL = os.getenv("TTS_URL", "http://tts:5002/speak")
VAD_URL = os.getenv("VAD_URL", "http://vad:9001")
COMPANY = os.getenv("COMPANY", "Sonrisa")
HTTPX_TIMEOUT = float(os.getenv("HTTPX_TIMEOUT", "120"))

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(title=f"Voice assistant for {COMPANY}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static frontend
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _resample_pcm16(pcm: bytes, src_rate: int, target_rate: int) -> bytes:
    """
    Resample signed 16-bit PCM using the stdlib audioop so we can feed
    VAD + STT with consistent audio without pulling in heavy deps.
    """
    if src_rate == target_rate:
        return pcm
    converted, _ = audioop.ratecv(pcm, 2, 1, src_rate, target_rate, None)
    return converted


def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


async def _call_stt(
    client: httpx.AsyncClient,
    audio_bytes: bytes,
    language: str,
    beam_size: int,
    mime_type: str,
) -> Dict[str, Any]:
    files = {
        "audio": ("clip", audio_bytes, mime_type),
    }
    data = {
        "language": language,
        "beam_size": str(beam_size),
    }
    resp = await client.post(STT_URL, files=files, data=data)
    resp.raise_for_status()
    return resp.json()


async def _call_rag(client: httpx.AsyncClient, question: str) -> Dict[str, Any]:
    resp = await client.post(RAG_URL, json={"question": question})
    resp.raise_for_status()
    return resp.json()


async def _call_tts(client: httpx.AsyncClient, text: str) -> Dict[str, Any]:
    resp = await client.post(TTS_URL, json={"text": text, "format": "json"})
    resp.raise_for_status()
    return resp.json()


class VadClient:
    def __init__(self, base_url: str, client: httpx.AsyncClient):
        self.base_url = base_url.rstrip("/")
        self.client = client

    async def analyze_clip(
        self,
        pcm: bytes,
        sample_rate: int,
        aggressiveness: int = 2,
        frame_ms: int = 30,
        silence_ms: int = 600,
    ) -> Dict[str, Any]:
        payload = {
            "audio_b64": base64.b64encode(pcm).decode(),
            "sample_rate": sample_rate,
            "frame_ms": frame_ms,
            "aggressiveness": aggressiveness,
            "silence_ms": silence_ms,
        }
        resp = await self.client.post(f"{self.base_url}/analyze", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def push_frame(
        self,
        session_id: str,
        pcm: bytes,
        sample_rate: int,
        aggressiveness: int,
        frame_ms: int,
        silence_ms: int,
    ) -> Dict[str, Any]:
        payload = {
            "session_id": session_id,
            "audio_b64": base64.b64encode(pcm).decode(),
            "sample_rate": sample_rate,
            "aggressiveness": aggressiveness,
            "frame_ms": frame_ms,
            "silence_ms": silence_ms,
        }
        resp = await self.client.post(f"{self.base_url}/frame", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def close_session(self, session_id: str) -> None:
        try:
            await self.client.post(
                f"{self.base_url}/frame",
                json={"session_id": session_id, "audio_b64": "", "close": True},
            )
        except Exception:
            # The VAD side is best-effort; failures should not break the UX.
            pass


async def run_pipeline(
    audio_bytes: bytes,
    mime_type: str,
    language: str,
    beam_size: int,
    client: httpx.AsyncClient,
) -> Dict[str, Any]:
    """
    Execute STT -> RAG -> TTS on an audio payload and return transcript,
    answer, audio_b64, audio_mime, timings.
    """
    T0 = time.perf_counter()

    # ---- STT ----
    s0 = time.perf_counter()
    stt = await _call_stt(
        client,
        audio_bytes,
        language=language,
        beam_size=beam_size,
        mime_type=mime_type,
    )
    s1 = time.perf_counter()
    transcript = (stt.get("text") or "").strip()
    stt_timings = stt.get("timings") or {}

    # ---- RAG ----
    answer = ""
    rag_http_ms = rag_retrieval_ms = rag_llm_ms = rag_total_ms = None
    if transcript:
        r0 = time.perf_counter()
        rag_resp = await _call_rag(client, question=transcript)
        r1 = time.perf_counter()
        answer = rag_resp.get("answer") or ""
        rag_http_ms = round((r1 - r0) * 1000, 1)
        inner = rag_resp.get("timings") or {}
        rag_retrieval_ms = inner.get("retrieval_ms")
        rag_llm_ms = inner.get("llm_ms")
        rag_total_ms = inner.get("total_ms")

    # ---- TTS ----
    audio_b64 = None
    audio_mime = None
    tts_http_ms = tts_inner_ms = None
    if answer:
        t0 = time.perf_counter()
        tts = await _call_tts(client, text=answer)
        t1 = time.perf_counter()
        audio_b64 = tts.get("audio_b64")
        audio_mime = tts.get("audio_mime", "audio/wav")
        tts_http_ms = round((t1 - t0) * 1000, 1)
        tts_inner_ms = (tts.get("timings") or {}).get("total_ms")

    total_ms = round((time.perf_counter() - T0) * 1000, 1)

    return {
        "transcript": transcript,
        "answer": answer,
        "audio_b64": audio_b64,
        "audio_mime": audio_mime,
        "timings": {
            "stt_http_ms": round((s1 - s0) * 1000, 1),
            "stt_inner_ms": stt_timings.get("total_ms")
            or stt_timings.get("transcribe_ms"),
            "rag_http_ms": rag_http_ms,
            "rag_retrieval_ms": rag_retrieval_ms,
            "rag_llm_ms": rag_llm_ms,
            "rag_inner_total_ms": rag_total_ms,
            "tts_http_ms": tts_http_ms,
            "tts_inner_ms": tts_inner_ms,
            "end_to_end_ms": total_ms,
        },
    }


# --------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "stt_url": STT_URL,
        "rag_url": RAG_URL,
        "tts_url": TTS_URL,
        "vad_url": VAD_URL,
    }


@app.get("/api/config")
async def config():
    return {
        "company": COMPANY,
        "stt_url": STT_URL,
        "rag_url": RAG_URL,
        "tts_url": TTS_URL,
        "vad_url": VAD_URL,
    }


@app.post("/api/voice")
async def voice(
    audio: UploadFile = File(...),
    language: str = Form("hu"),
    beam_size: int = Form(5),
    use_vad: bool = Form(False),
    vad_aggressiveness: int = Form(3),
    vad_silence_ms: int = Form(250),
    pcm_sample_rate: int = Form(16000),
    vad_frame_ms: int = Form(10),
):
    """
    One-shot processing path. Optionally runs VAD to trim silence, then
    sends audio through STT -> RAG -> TTS.
    """
    raw_bytes = await audio.read()
    pcm = raw_bytes
    sample_rate = pcm_sample_rate
    mime_type = audio.content_type or "application/octet-stream"
    payload = raw_bytes
    use_pcm_path = mime_type in {"audio/pcm", "application/octet-stream"}
    vad_warning = None

    # If VAD is requested, delegate trimming to the VAD microservice.
    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
        vad_client = VadClient(VAD_URL, client)
        if use_vad and use_pcm_path:
            try:
                vad_result = await vad_client.analyze_clip(
                    pcm=pcm,
                    sample_rate=sample_rate,
                    aggressiveness=vad_aggressiveness,
                    silence_ms=vad_silence_ms,
                    frame_ms=vad_frame_ms,
                )
                trimmed_b64 = vad_result.get("trimmed_audio_b64")
                if trimmed_b64:
                    pcm = base64.b64decode(trimmed_b64)
                payload = pcm16_to_wav_bytes(pcm, sample_rate)
                mime_type = "audio/wav"
            except Exception as exc:
                vad_warning = f"VAD unavailable: {exc}"
        elif use_pcm_path:
            payload = pcm16_to_wav_bytes(pcm, sample_rate)
            mime_type = "audio/wav"

        result = await run_pipeline(
            audio_bytes=payload,
            mime_type=mime_type,
            language=language,
            beam_size=beam_size,
            client=client,
        )
        if vad_warning:
            result["warning"] = vad_warning
        return JSONResponse(result)


@app.websocket("/api/stream")
async def stream(ws: WebSocket):
    """
    Bi-directional streaming endpoint.
    Frontend sends raw Int16 PCM frames. Backend forwards frames to VAD
    service, buffers speech segments, and runs STT/RAG/TTS when a stop
    event is detected by VAD.
    """
    await ws.accept()
    try:
        settings_msg = await ws.receive_text()
        settings = json.loads(settings_msg)
    except Exception:
        await ws.close(code=4001)
        return

    language = settings.get("language") or "hu"
    beam_size = int(settings.get("beam_size") or 5)
    sample_rate = int(settings.get("sample_rate") or 16000)
    greeting_text = (settings.get("greeting_text") or "").strip()
    vad_opts = settings.get("vad") or {}
    vad_aggressiveness = int(vad_opts.get("aggressiveness") or 3)
    vad_frame_ms = int(vad_opts.get("frame_ms") or 10)
    vad_silence_ms = int(vad_opts.get("silence_ms") or 250)
    session_id = settings.get("session_id") or str(uuid.uuid4())

    await ws.send_json({"type": "ready", "session_id": session_id})

    # Play greeting message if provided
    if greeting_text:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as greeting_client:
            try:
                tts_resp = await _call_tts(greeting_client, greeting_text)
                if tts_resp.get("audio_b64"):
                    await ws.send_json({
                        "type": "greeting",
                        "audio_b64": tts_resp["audio_b64"],
                        "audio_mime": tts_resp.get("audio_mime", "audio/wav"),
                    })
            except Exception as exc:
                await ws.send_json({
                    "type": "error",
                    "message": f"greeting tts failed: {exc}",
                })
    buffer = bytearray()
    capturing = False
    last_voice_ts = None

    async def finalize_segment():
        nonlocal buffer, capturing, last_voice_ts
        if not buffer:
            return
        await ws.send_json({"type": "status", "message": "speech-stop"})
        wav_audio = pcm16_to_wav_bytes(bytes(buffer), 16000)
        result = await run_pipeline(
            audio_bytes=wav_audio,
            mime_type="audio/wav",
            language=language,
            beam_size=beam_size,
            client=client,
        )
        await ws.send_json({"type": "result", **result})
        buffer.clear()
        capturing = False
        last_voice_ts = None

    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
        vad_client = VadClient(VAD_URL, client)
        try:
            while True:
                msg = await ws.receive()
                if "bytes" in msg:
                    raw_pcm = msg["bytes"]
                    pcm_16k = _resample_pcm16(raw_pcm, sample_rate, 16000)
                    try:
                        vad_resp = await vad_client.push_frame(
                            session_id=session_id,
                            pcm=pcm_16k,
                            sample_rate=16000,
                            aggressiveness=vad_aggressiveness,
                            frame_ms=vad_frame_ms,
                            silence_ms=vad_silence_ms,
                        )
                    except Exception as exc:
                        await ws.send_json(
                            {
                                "type": "error",
                                "message": f"vad unavailable: {exc}",
                            }
                        )
                        break
                    trigger = vad_resp.get("trigger")
                    in_speech = vad_resp.get("in_speech")

                    if in_speech:
                        buffer.extend(pcm_16k)
                        if not capturing:
                            capturing = True
                            await ws.send_json(
                                {"type": "status", "message": "speech-start"}
                            )
                        last_voice_ts = time.perf_counter()
                    else:
                        # Fallback: if we've been silent for the configured window, stop.
                        if capturing and last_voice_ts:
                            if (time.perf_counter() - last_voice_ts) * 1000 >= vad_silence_ms:
                                await finalize_segment()

                    if trigger == "stop" and buffer:
                        await finalize_segment()
                elif "text" in msg:
                    text = msg["text"]
                    if text == "stop":
                        if buffer:
                            await finalize_segment()
                        break
        except WebSocketDisconnect:
            pass
        finally:
            await vad_client.close_session(session_id)
            await ws.close()
