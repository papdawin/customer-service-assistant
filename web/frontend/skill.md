---
name: frontend-server
description: UI server for the voice assistant platform.
---

# Frontend Server

## Purpose
Serve the UI for a callable voice assistant deployed at different companies.

## Interfaces
- HTTP GET `/` renders `index.html` via Jinja2.
- Static files mounted at `/static`.

## Models
- None.

## Libraries
- `fastapi`
- `jinja2`
- `starlette`
- stdlib: `os`, `pathlib`

## Runtime Config
- `API_BASE` (backend base URL)
- `TITLE` (page title)

## Main Components
- Template loader and render context.
- Static asset mount.
