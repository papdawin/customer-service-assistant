---
name: vad-service
description: Shared voice activity detection for the multi-tenant voice assistant platform.
---

# VAD Service (WebRTC VAD)

## Purpose
Detect speech segments for a multi-tenant voice assistant deployed at different companies. This service is shared across tenants.

## Interfaces
- HTTP GET `/health` for session counts.
- HTTP POST `/analyze` for offline segment detection and trimmed audio.
- HTTP POST `/frame` for streaming VAD sessions and triggers.

## Models
- WebRTC VAD engine `webrtcvad.Vad` with aggressiveness 0-3.

## Libraries
- `fastapi`
- `pydantic`
- `webrtcvad`
- stdlib: `asyncio`, `base64`, `struct`, `time`, `logging`

## Runtime Config
- None (all tuning is request parameters).

## Main Components
- `VadSession` state machine with energy gating.
- Offline segmentation and trimmed-audio output.
- Session tracking with async lock.
