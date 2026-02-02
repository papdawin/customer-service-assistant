---
name: tts-service
description: Shared text-to-speech microservice for the multi-tenant voice assistant platform.
---

# TTS Service (Piper)

## Purpose
Provide text-to-speech for a multi-tenant voice assistant deployed at different companies. This service is shared across tenants.

## Interfaces
- HTTP GET `/health` for voice and device info.
- HTTP POST `/speak` for speech synthesis.

## Models
- Piper voice ONNX from `PIPER_VOICE` with config `PIPER_VOICE_JSON`.

## Libraries
- `fastapi`
- `piper` (PiperVoice)
- `pydantic`
- stdlib: `io`, `wave`, `base64`, `time`, `os`

## Runtime Config
- `PIPER_VOICE` (voice path)
- `PIPER_VOICE_JSON` (voice config)
- `TTS_DEVICE` (cpu or cuda)

## Main Components
- Voice load on startup.
- `SpeakRequest` validation.
- WAV serialization and JSON or binary response.
