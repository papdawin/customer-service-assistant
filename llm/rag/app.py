import os
import glob
from typing import List
from fastapi import FastAPI, Body
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# --------------------
# Environment / config
# --------------------
TENANT_ID = os.getenv("TENANT_ID", "tenant_unknown")
DATA_DIR = "/app/data"
INDEX_DIR = "/app/indices"
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")  # e.g., http://vllm:8000/v1
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy")

# Your vLLM server uses --max-model-len 256
MAX_MODEL_TOKENS = int(os.getenv("MAX_MODEL_TOKENS", "256"))
# Keep answers short so inputs have room
GEN_TOKENS = int(os.getenv("GEN_TOKENS", "64"))
# Approx token calc (rough but safe for packing)
APPROX_CHARS_PER_TOKEN = 4
# Reserve room for system + question
RESERVE_FOR_SYS_Q = int(os.getenv("RESERVE_FOR_SYS_Q", "110"))

# Context budget
CONTEXT_TOKENS_BUDGET = max(32, MAX_MODEL_TOKENS - GEN_TOKENS - RESERVE_FOR_SYS_Q)
CONTEXT_CHAR_BUDGET = CONTEXT_TOKENS_BUDGET * APPROX_CHARS_PER_TOKEN
# Prevent a single chunk from eating the whole budget
PER_DOC_CHAR_CAP = int(os.getenv("PER_DOC_CHAR_CAP", "160"))

# --------------------
# Embeddings (CPU)
# --------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# ------------------------------
# Load or build per-tenant FAISS
# ------------------------------
def load_or_build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss_path = os.path.join(INDEX_DIR, "faiss_index")
    if os.path.exists(faiss_path):
        return FAISS.load_local(
            faiss_path, embeddings, allow_dangerous_deserialization=True
        )

    # Ingest .txt files under /app/data (each file becomes a doc)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "**/*.txt"), recursive=True))
    docs: List[Document] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            txt = f.read()
        docs.append(
            Document(
                page_content=txt,
                metadata={"tenant": TENANT_ID, "source": os.path.basename(fp)},
            )
        )

    if not docs:
        # Build an empty index to avoid boot failure
        return FAISS.from_texts(
            [""], embeddings, metadatas=[{"tenant": TENANT_ID, "source": "empty"}]
        )

    # Use smaller chunks so retrieved context stays short
    splitter = RecursiveCharacterTextSplitter(chunk_size=220, chunk_overlap=40)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(faiss_path)
    return vs

vectorstore = load_or_build_index()

# Return fewer docs to the LLM to respect the 256-token limit
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# --------------------
# LLM (via vLLM/OpenAI)
# --------------------
llm = ChatOpenAI(
    model="Qwen/Qwen2.5-3B-Instruct-AWQ",  # forwarded to your vLLM server
    base_url=VLLM_BASE_URL,
    api_key=OPENAI_API_KEY,
    temperature=0.2,
    max_tokens=GEN_TOKENS,  # <= cap generation
)

SYSTEM_PROMPT = "If the answer is not in the context, say you don't know."

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer concisely:"),
    ]
)

# --------------------
# Formatting utilities
# --------------------
def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"

def format_docs(docs: List[Document]) -> str:
    """Pack docs into a strict character budget so we never exceed model limits."""
    pieces = []
    remaining = CONTEXT_CHAR_BUDGET
    for d in docs:
        if remaining <= 0:
            break
        src = d.metadata.get("source", "unknown")
        snippet = _truncate(d.page_content, min(PER_DOC_CHAR_CAP, max(0, remaining)))
        piece = f"[{src}] {snippet}"
        pieces.append(piece)
        remaining -= len(piece)
    return "\n\n".join(pieces)

# --------------------
# LCEL pipeline
# --------------------
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# --------------------
# FastAPI app
# --------------------
app = FastAPI(title=f"RAG Service – {TENANT_ID}", version="1.0.0")

class Query(BaseModel):
    question: str

@app.get("/health")
def health():
    return {
        "status": "ok",
        "tenant": TENANT_ID,
        "max_model_tokens": MAX_MODEL_TOKENS,
        "gen_tokens": GEN_TOKENS,
        "context_tokens_budget": CONTEXT_TOKENS_BUDGET,
    }

@app.post("/query")
def query(payload: Query = Body(...)):
    resp = chain.invoke(payload.question)
    return {"answer": resp.content, "tenant": TENANT_ID}
