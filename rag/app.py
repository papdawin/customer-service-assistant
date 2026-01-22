import time
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException, Response

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONTEXT_TOKENS_BUDGET,
    EMBED_MODEL,
    TENANT_ID,
    TOP_K,
)
from index import build_vector_store, log_data_dir_contents, make_retriever
from models import Query
from token_utils import tokenizer_loaded
from retrieval import retrieve_documents
from llm import llm, prompt
from token_utils import format_docs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the tenant vector index at startup and expose retriever state."""
    log_data_dir_contents()
    app.state.vectorstore = build_vector_store()
    app.state.retriever = make_retriever(app.state.vectorstore)
    try:
        doc_total = getattr(app.state.vectorstore.index, "ntotal", None)
        print(f"[RAG] Loaded FAISS index for tenant '{TENANT_ID}' (ntotal={doc_total})")
    except Exception as exc:
        print(f"[RAG] Failed to introspect FAISS index: {exc!r}")
    yield


app = FastAPI(title=f"RAG Service – {TENANT_ID}", version="1.3.0", lifespan=lifespan)


@app.get("/health")
def health():
    """Report service readiness plus index and model status flags."""
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
        "doc_count": doc_count,
        "tokenizer_loaded": tokenizer_loaded(),
        "top_k": TOP_K,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "context_tokens_budget": CONTEXT_TOKENS_BUDGET,
    }


@app.post("/query")
def query(response: Response, payload: Query = Body(...)):
    """Answer a user question by retrieving context, calling the LLM, and returning timings."""
    t0 = time.perf_counter()

    try:
        question = payload.prompt()
    except ValueError:
        raise HTTPException(status_code=400, detail="Empty question/text payload")
    print(f"[RAG] Incoming question: {question}")
    tr0 = time.perf_counter()
    docs = retrieve_documents(app.state.retriever, f"query: {question}")
    retrieval_ms = round((time.perf_counter() - tr0) * 1000, 1)
    ctx = format_docs(docs)

    sources = [doc.metadata.get("source", "unknown") for doc in docs]
    preview = ctx[:800]
    if len(ctx) > 800:
        preview += " …"
    print(f"[RAG] Retrieved {len(docs)} docs for tenant '{TENANT_ID}': {sources}")
    print(f"[RAG] Context preview:\n{preview}")

    tl0 = time.perf_counter()
    messages = prompt.format_messages(question=question, context=ctx)
    resp = llm.invoke(messages)
    llm_ms = round((time.perf_counter() - tl0) * 1000, 1)

    print(f"[RAG] LLM answer: {resp.content}")

    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["Server-Timing"] = (
        f"retrieval;dur={retrieval_ms},llm;dur={llm_ms},total;dur={total_ms}"
    )

    return {
        "answer": resp.content,
        "tenant": TENANT_ID,
        "timings": {
            "retrieval_ms": retrieval_ms,
            "llm_ms": llm_ms,
            "total_ms": total_ms,
        },
        "sources": sources,
    }
