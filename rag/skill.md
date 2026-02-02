---
name: rag-service
description: Tenant-specific RAG service for company knowledge in the voice assistant platform.
---

# RAG Service

## Purpose
Enable each company to supply its own information (e.g., location, opening hours, contact and reachability details) so the voice assistant can search and answer accurately. Each tenant has its own index; the LLM is shared across tenants.

## Interfaces
- HTTP GET `/health` for index and model status.
- HTTP POST `/query` for question answering and timing data.

## Models
- Embedding model from `EMBED_MODEL`.
- Shared LLM client `llm` from `llm` module.

## Libraries
- `fastapi`
- stdlib: `time`, `contextlib`

## Runtime Config
- `TENANT_ID`, `EMBED_MODEL`, `TOP_K`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CONTEXT_TOKENS_BUDGET`

## Main Components
- Lifespan startup loads FAISS index and retriever for the tenant.
- Retrieval via `retrieve_documents` and context formatting.
- LLM prompt assembly and invocation with timing.
