---
name: stt-service
description: Shared speech-to-text microservice for the multi-tenant voice assistant platform.
---

# STT Service (Whisper)

## Purpose
Provide speech-to-text for a multi-tenant voice assistant deployed at different companies. This service is shared across tenants.

## Interfaces
- HTTP GET `/health` for readiness and model/device info.
- HTTP POST `/transcribe` for audio transcription with optional language and beam size.

## Models
- Whisper model from `STT_MODEL_ID` (default `large-v3`).

## Libraries
- `fastapi`
- `torch`
- `whisper`
- stdlib: `os`, `tempfile`, `time`

## Runtime Config
- `STT_MODEL_ID` (model name)
- `STT_LANGUAGE` (default language)

## Main Components
- Model load on startup with GPU/CPU selection.
- Temp file handling for uploads.
- Transcription timing and `Server-Timing` header.
