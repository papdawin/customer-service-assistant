import os

"""
Environment-driven configuration values shared across the RAG service.
"""

TENANT_ID = os.getenv("TENANT_ID", "tenant_unknown")
DATA_DIR = os.getenv("DATA_DIR", "/app/data")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

TOP_K = int(os.getenv("TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "320"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")

CONTEXT_TOKENS_BUDGET = int(os.getenv("CONTEXT_TOKENS_BUDGET", "1800"))
APPROX_CHARS_PER_TOKEN = int(os.getenv("APPROX_CHARS_PER_TOKEN", "4"))
PER_DOC_TOKEN_CAP = int(os.getenv("PER_DOC_TOKEN_CAP", "400"))
PER_DOC_CHAR_CAP = int(os.getenv("PER_DOC_CHAR_CAP", "1200"))

TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", LLM_MODEL)

__all__ = [
    "TENANT_ID",
    "DATA_DIR",
    "VLLM_BASE_URL",
    "OPENAI_API_KEY",
    "LLM_MODEL",
    "TOP_K",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "EMBED_MODEL",
    "CONTEXT_TOKENS_BUDGET",
    "APPROX_CHARS_PER_TOKEN",
    "PER_DOC_TOKEN_CAP",
    "PER_DOC_CHAR_CAP",
    "TOKENIZER_NAME",
]
