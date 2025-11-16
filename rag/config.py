import os

"""
Environment-driven configuration values shared across the RAG service.
"""

TENANT_ID = os.getenv("TENANT_ID", "tenant_unknown")
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
INDEX_BASE_DIR = os.getenv("INDEX_DIR", "/app/indices")
INDEX_DIR = os.path.join(INDEX_BASE_DIR, TENANT_ID)

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-4B")

TOP_K = int(os.getenv("TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "320"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
ALLOW_DANGEROUS_DESER = os.getenv("ALLOW_DANGEROUS_DESERIALIZATION", "true").lower()

CONTEXT_TOKENS_BUDGET = int(os.getenv("CONTEXT_TOKENS_BUDGET", "1800"))
APPROX_CHARS_PER_TOKEN = int(os.getenv("APPROX_CHARS_PER_TOKEN", "4"))
PER_DOC_TOKEN_CAP = int(os.getenv("PER_DOC_TOKEN_CAP", "400"))
PER_DOC_CHAR_CAP = int(os.getenv("PER_DOC_CHAR_CAP", "1200"))

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "2"))
USE_RERANK = os.getenv("USE_RERANK", "true").lower() in {"1", "true", "yes"}

TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", LLM_MODEL)

__all__ = [
    "TENANT_ID",
    "DATA_DIR",
    "INDEX_BASE_DIR",
    "INDEX_DIR",
    "VLLM_BASE_URL",
    "OPENAI_API_KEY",
    "LLM_MODEL",
    "TOP_K",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "EMBED_MODEL",
    "ALLOW_DANGEROUS_DESER",
    "CONTEXT_TOKENS_BUDGET",
    "APPROX_CHARS_PER_TOKEN",
    "PER_DOC_TOKEN_CAP",
    "PER_DOC_CHAR_CAP",
    "RERANK_MODEL",
    "RERANK_TOP_N",
    "USE_RERANK",
    "TOKENIZER_NAME",
]
