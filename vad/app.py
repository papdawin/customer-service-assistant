import asyncio
import base64
import logging
import struct
import time
from typing import Dict, List, Tuple

import webrtcvad
from fastapi import FastAPI
from pydantic import BaseModel, Field


def frame_energy(frame: bytes) -> float:
    """Calculate RMS energy of a 16-bit PCM frame."""
    if len(frame) < 2:
        return 0.0
    samples = struct.unpack(f"<{len(frame)//2}h", frame)
    if not samples:
        return 0.0
    sum_sq = sum(s * s for s in samples)
    return (sum_sq / len(samples)) ** 0.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vad")

app = FastAPI(title="Streaming VAD service")

sessions: Dict[str, "VadSession"] = {}
sessions_lock = asyncio.Lock()


class AnalyzeRequest(BaseModel):
    audio_b64: str
    sample_rate: int = 16000
    frame_ms: int = Field(10, description="frame size in ms (10/20/30)")
    aggressiveness: int = Field(3, ge=0, le=3)
    silence_ms: int = Field(250, description="silence threshold before stop")


class FrameRequest(BaseModel):
    session_id: str
    audio_b64: str
    sample_rate: int = 16000
    frame_ms: int = 10
    aggressiveness: int = 3
    silence_ms: int = 250
    min_speech_ms: int = 150  # Minimum speech duration before triggering start
    energy_threshold: float = 500.0  # Minimum RMS energy to consider as speech
    max_speech_ms: int = 30000  # Maximum speech duration before forcing stop
    close: bool = False


def chunk_frames(pcm: bytes, sample_rate: int, frame_ms: int) -> List[bytes]:
    frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # 16-bit mono
    return [
        pcm[i : i + frame_bytes]
        for i in range(0, len(pcm) - frame_bytes + 1, frame_bytes)
    ]


def run_offline_vad(
    pcm: bytes, sample_rate: int, frame_ms: int, aggressiveness: int, silence_ms: int
) -> Tuple[List[Dict[str, int]], bytes]:
    vad = webrtcvad.Vad(aggressiveness)
    frames = chunk_frames(pcm, sample_rate, frame_ms)

    segments: List[Dict[str, int]] = []
    current = None
    silence_acc = 0
    cursor = 0
    collected = bytearray()

    for frame in frames:
        is_speech = vad.is_speech(frame, sample_rate)
        if is_speech:
            collected.extend(frame)
            silence_acc = 0
            if current is None:
                current = {"start_ms": cursor}
        else:
            if current:
                silence_acc += frame_ms
                if silence_acc >= silence_ms:
                    current["end_ms"] = cursor
                    segments.append(current)
                    current = None
                    silence_acc = 0
        cursor += frame_ms

    if current:
        current["end_ms"] = cursor
        segments.append(current)

    return segments, bytes(collected)


class VadSession:
    def __init__(self, aggressiveness: int, frame_ms: int, silence_ms: int, min_speech_ms: int = 150, energy_threshold: float = 500.0, max_speech_ms: int = 30000):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_ms = frame_ms
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms  # Minimum speech duration before confirming
        self.energy_threshold = energy_threshold  # Minimum RMS energy to consider as potential speech
        self.max_speech_ms = max_speech_ms  # Maximum speech duration before forcing stop (30s default)
        self.in_speech = False
        self.pending_speech = False  # Speech detected but not yet confirmed
        self.speech_acc = 0  # Accumulated speech duration
        self.silence_acc = 0
        self.buffer_ms = 0
        self.created_at = time.time()

    def process(self, pcm: bytes, sample_rate: int) -> str:
        """
        Process the incoming PCM chunk and return the last trigger:
        "start" -> speech started, "speech" -> continuing, "stop" -> speech ended,
        "silence" -> outside speech.
        """
        frames = chunk_frames(pcm, sample_rate, self.frame_ms)
        last_trigger = "silence"
        for frame in frames:
            energy = frame_energy(frame)
            vad_says_speech = self.vad.is_speech(frame, sample_rate) if energy >= self.energy_threshold else False
            speech = energy >= self.energy_threshold and vad_says_speech

            # Debug logging
            if self.in_speech:
                logger.debug(
                    "energy=%.0f threshold=%.0f vad=%s silence_acc=%d/%d",
                    energy, self.energy_threshold, vad_says_speech, self.silence_acc, self.silence_ms
                )

            # Force stop if speech has been going on too long
            if self.in_speech and self.buffer_ms >= self.max_speech_ms:
                logger.info("Force stop: max speech duration reached (%d ms)", self.buffer_ms)
                self.in_speech = False
                self.buffer_ms = 0
                self.silence_acc = 0
                self.speech_acc = 0
                return "stop"

            if speech:
                if not self.in_speech:
                    # Accumulate speech frames before confirming start
                    self.speech_acc += self.frame_ms
                    self.pending_speech = True
                    if self.speech_acc >= self.min_speech_ms:
                        self.in_speech = True
                        self.pending_speech = False
                        self.silence_acc = 0
                        last_trigger = "start"
                else:
                    self.silence_acc = 0  # Reset silence counter when speech continues
                    last_trigger = "speech"
                self.buffer_ms += self.frame_ms
            else:
                if self.pending_speech:
                    # Reset pending speech if silence interrupts before confirmation
                    self.pending_speech = False
                    self.speech_acc = 0
                    self.buffer_ms = 0
                if self.in_speech:
                    self.silence_acc += self.frame_ms
                    last_trigger = "silence"
                    logger.debug("Silence accumulating: %d/%d ms", self.silence_acc, self.silence_ms)
                    if self.silence_acc >= self.silence_ms:
                        logger.info("Stop triggered: silence_acc=%d >= silence_ms=%d", self.silence_acc, self.silence_ms)
                        self.in_speech = False
                        self.buffer_ms = 0
                        self.silence_acc = 0
                        self.speech_acc = 0
                        last_trigger = "stop"
                else:
                    last_trigger = "silence"
        return last_trigger


@app.get("/health")
async def health():
    return {"ok": True, "active_sessions": len(sessions)}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    pcm = base64.b64decode(req.audio_b64)
    segments, trimmed = run_offline_vad(
        pcm=pcm,
        sample_rate=req.sample_rate,
        frame_ms=req.frame_ms,
        aggressiveness=req.aggressiveness,
        silence_ms=req.silence_ms,
    )
    trimmed_b64 = base64.b64encode(trimmed).decode() if trimmed else None
    if segments:
        logger.info(
            "offline vad detected %d segments (len=%d bytes, sr=%d)",
            len(segments),
            len(pcm),
            req.sample_rate,
        )
    return {
        "speech_segments": segments,
        "has_speech": bool(segments),
        "trimmed_audio_b64": trimmed_b64,
        "sample_rate": req.sample_rate,
    }


@app.post("/frame")
async def frame(req: FrameRequest):
    if req.close:
        async with sessions_lock:
            sessions.pop(req.session_id, None)
        return {"closed": True}

    pcm = base64.b64decode(req.audio_b64) if req.audio_b64 else b""

    async with sessions_lock:
        if req.session_id not in sessions:
            sessions[req.session_id] = VadSession(
                aggressiveness=req.aggressiveness,
                frame_ms=req.frame_ms,
                silence_ms=req.silence_ms,
                min_speech_ms=req.min_speech_ms,
                energy_threshold=req.energy_threshold,
                max_speech_ms=req.max_speech_ms,
            )
        session = sessions[req.session_id]

    trigger = session.process(pcm, sample_rate=req.sample_rate) if pcm else "silence"

    if trigger in {"start", "stop"}:
        logger.info(
            "session %s trigger=%s in_speech=%s buffer_ms=%d",
            req.session_id,
            trigger,
            session.in_speech,
            session.buffer_ms,
        )

    return {
        "session_id": req.session_id,
        "trigger": trigger,
        "in_speech": session.in_speech,
        "buffer_ms": session.buffer_ms,
    }
