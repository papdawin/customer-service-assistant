---
name: api-gateway
description: Multi-tenant backend gateway for the voice assistant platform.
---

# Backend API Gateway

## Purpose
Serve a callable voice assistant deployed at different companies. It orchestrates shared STT, VAD, and TTS services, and routes tenant-specific queries to the correct RAG service.

## Interfaces
- HTTP GET `/api/health` and `/api/config` for service config.
- HTTP POST `/api/voice` for one-shot audio processing.
- WebSocket `/api/stream` for streaming VAD + pipeline execution.

## Models
- None locally; delegates to shared STT/TTS/VAD and tenant-specific RAG.

## Libraries
- `fastapi`
- `httpx`
- `starlette`
- stdlib: `audioop`, `base64`, `json`, `wave`, `uuid`, `pathlib`, `time`

## Runtime Config
- `STT_URL`, `RAG_URL`, `TTS_URL`, `VAD_URL`
- `COMPANY`, `HTTPX_TIMEOUT`

## Main Components
- Pipeline runner combining STT, RAG, and TTS calls.
- `VadClient` helper for VAD service calls.
- PCM resampling and WAV conversion utilities.
- Static frontend mount when present.
