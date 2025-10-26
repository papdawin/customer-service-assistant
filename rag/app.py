import os
import glob
import time
from typing import List, Optional, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, Body, Response
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Small LCEL bits for the (optional) pipeline; we mostly call LLM directly
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------
# Configuration (env-driven)
# ----------------------------
TENANT_ID = os.getenv("TENANT_ID", "tenant_unknown")
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
INDEX_BASE_DIR = os.getenv("INDEX_DIR", "/app/indices")
INDEX_DIR = os.path.join(INDEX_BASE_DIR, TENANT_ID)  # per-tenant isolation

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-4B")  # good multilingual baseline

# Retrieval & ingestion
TOP_K = int(os.getenv("TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "320"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))

# Embeddings (e5 multilingual is strong for HU; use query/passage prefixes)
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
# Normalize => cosine via L2 is OK in FAISS
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)

ALLOW_DANGEROUS_DESER = os.getenv("ALLOW_DANGEROUS_DESERIALIZATION", "true").lower() in {"1", "true", "yes"}

# Packing budget (modest; avoids lost-in-the-middle)
CONTEXT_TOKENS_BUDGET = int(os.getenv("CONTEXT_TOKENS_BUDGET", "1800"))
APPROX_CHARS_PER_TOKEN = int(os.getenv("APPROX_CHARS_PER_TOKEN", "4"))
PER_DOC_TOKEN_CAP = int(os.getenv("PER_DOC_TOKEN_CAP", "400"))
PER_DOC_CHAR_CAP = int(os.getenv("PER_DOC_CHAR_CAP", "1200"))

# Optional reranker
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "2"))
USE_RERANK = os.getenv("USE_RERANK", "true").lower() in {"1", "true", "yes"}

# ----------------------------
# Optional tokenizer-aware packing
# ----------------------------
_tokenizer = None
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", LLM_MODEL)
try:
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
except Exception:
    _tokenizer = None  # fall back to char-based packing

def _truncate_chars(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"

def _pack_docs_token_budget(docs: List[Document], max_tokens: int) -> str:
    """Pack docs into a strict token budget. Falls back to char-based packing if tokenizer is unavailable."""
    if _tokenizer is None:
        # Char-based fallback using a strict global budget.
        pieces = []
        remaining = max_tokens * APPROX_CHARS_PER_TOKEN
        for d in docs:
            if remaining <= 0:
                break
            src = d.metadata.get("source", "unknown")
            snippet = _truncate_chars(d.page_content, min(PER_DOC_CHAR_CAP, max(0, remaining)))
            piece = f"[{src}] {snippet}"
            pieces.append(piece)
            remaining -= len(piece)
        return "\n\n".join(pieces)

    used = 0
    pieces = []
    for d in docs:
        if used >= max_tokens:
            break
        src = d.metadata.get("source", "unknown")

        tokens = _tokenizer.encode(d.page_content, add_special_tokens=False)
        if not tokens:
            continue
        tokens = tokens[: max(0, min(PER_DOC_TOKEN_CAP, max_tokens - used))]
        snippet = _tokenizer.decode(tokens)

        piece = f"[{src}] {snippet}"
        piece_tok_len = len(_tokenizer.encode(piece, add_special_tokens=False))
        if used + piece_tok_len > max_tokens:
            room = max_tokens - used - len(_tokenizer.encode(f"[{src}] ", add_special_tokens=False))
            if room <= 0:
                break
            tokens = tokens[: max(0, room)]
            snippet = _tokenizer.decode(tokens)
            piece = f"[{src}] {snippet}"
            piece_tok_len = len(_tokenizer.encode(piece, add_special_tokens=False))
            if piece_tok_len <= 0:
                break

        used += piece_tok_len
        pieces.append(piece)
    return "\n\n".join(pieces)

def format_docs(docs: List[Document]) -> str:
    return _pack_docs_token_budget(docs, CONTEXT_TOKENS_BUDGET)

# ----------------------------
# Index build / load
# ----------------------------
def _retrieve(retriever, query: str):
    # Compat with older/newer LangChain
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    return retriever.invoke(query)

def _to_passage(text: str) -> str:
    # e5 expects "passage: ..." for docs
    return f"passage: {text.strip()}"

def load_or_build_index(force_rebuild: bool = False) -> FAISS:
    """Build a FAISS index from DATA_DIR/*.txt or load an existing one (per-tenant path)."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss_path = os.path.join(INDEX_DIR, "faiss_index")

    if (not force_rebuild) and os.path.exists(faiss_path):
        return FAISS.load_local(
            faiss_path,
            embeddings,
            allow_dangerous_deserialization=ALLOW_DANGEROUS_DESER,
        )

    files = sorted(glob.glob(os.path.join(DATA_DIR, "**/*.txt"), recursive=True))
    docs: List[Document] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                txt = f.read()
        except Exception:
            continue
        if not txt.strip():
            continue
        docs.append(
            Document(
                page_content=_to_passage(txt),
                metadata={"tenant": TENANT_ID, "source": os.path.basename(fp)},
            )
        )

    if not docs:
        vs = FAISS.from_texts(
            ["passage: "], embeddings, metadatas=[{"tenant": TENANT_ID, "source": "empty"}]
        )
        vs.save_local(faiss_path)
        return vs

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(faiss_path)
    return vs

def make_retriever(vs: FAISS):
    return vs.as_retriever(search_kwargs={"k": TOP_K})

# ----------------------------
# Optional: Reranker
# ----------------------------
_reranker = None
if USE_RERANK:
    try:
        # pip install FlagEmbedding
        from FlagEmbedding import FlagReranker  # type: ignore
        _reranker = FlagReranker(RERANK_MODEL, use_fp16=True)
    except Exception:
        _reranker = None  # silently disable if not available

def _maybe_rerank(query: str, docs: List[Document], top_n: int) -> List[Document]:
    if not _reranker:
        return docs[:top_n] if top_n and top_n < len(docs) else docs
    pairs = [[query, d.page_content] for d in docs]
    try:
        scores = _reranker.compute_score(pairs, normalize=True)  # returns List[float]
    except Exception:
        return docs[:top_n] if top_n and top_n < len(docs) else docs
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_n]]

# ----------------------------
# LLM + Prompt (Hungarian)
# ----------------------------
SYSTEM_PROMPT = (
    f"A {TENANT_ID} nevű vállalat asszisztense vagy, válaszolj röviden az ügyfél kérdéseire. Használd a kontextust, ha nem találod benne a megfelelő választ mondd, hogy nincs információd erről."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Kérdés: {question}\n\nKontextus:\n{context}"),
    ]
)

# No explicit max_tokens cap here; vLLM/model limits apply.
llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url=VLLM_BASE_URL,
    api_key=OPENAI_API_KEY,
    temperature=0.5,
    top_p=0.9,
    timeout=60,
    max_retries=2,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

# ----------------------------
# FastAPI app with lifespan
# ----------------------------
class Query(BaseModel):
    question: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vectorstore = load_or_build_index(force_rebuild=False)
    app.state.retriever = make_retriever(app.state.vectorstore)
    yield

app = FastAPI(title=f"RAG Service – {TENANT_ID}", version="1.3.0", lifespan=lifespan)

@app.get("/health")
def health():
    doc_count = None
    try:
        vs = app.state.vectorstore
        doc_count = getattr(vs.index, "ntotal", None)
    except Exception:
        pass
    return {
        "status": "ok",
        "tenant": TENANT_ID,
        "embed_model": EMBED_MODEL,
        "index_dir": INDEX_DIR,
        "doc_count": doc_count,
        "tokenizer_loaded": _tokenizer is not None,
        "reranker_loaded": _reranker is not None,
        "top_k": TOP_K,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "context_tokens_budget": CONTEXT_TOKENS_BUDGET,
    }

@app.post("/reindex")
def reindex():
    app.state.vectorstore = load_or_build_index(force_rebuild=True)
    app.state.retriever = make_retriever(app.state.vectorstore)
    return {"status": "reindexed", "tenant": TENANT_ID}

@app.post("/query")
def query(response: Response, payload: Query = Body(...)):
    t0 = time.perf_counter()

    # 1) Retrieval (+ optional rerank)
    tr0 = time.perf_counter()
    q = f"query: {payload.question.strip()}"  # e5 query prefix
    docs = _retrieve(app.state.retriever, q)
    docs = _maybe_rerank(q, docs, RERANK_TOP_N)
    tr1 = time.perf_counter()
    ctx = format_docs(docs)

    # 2) LLM call
    tl0 = time.perf_counter()
    messages = prompt.format_messages(question=payload.question, context=ctx)
    resp = llm.invoke(messages)
    tl1 = time.perf_counter()

    retrieval_ms = round((tr1 - tr0) * 1000, 1)
    llm_ms = round((tl1 - tl0) * 1000, 1)
    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["Server-Timing"] = (
        f"retrieval;dur={retrieval_ms},llm;dur={llm_ms},total;dur={total_ms}"
    )

    return {
        "answer": resp.content,
        "tenant": TENANT_ID,
        "timings": {"retrieval_ms": retrieval_ms, "llm_ms": llm_ms, "total_ms": total_ms},
        "sources": [d.metadata.get("source", "unknown") for d in docs],
    }
