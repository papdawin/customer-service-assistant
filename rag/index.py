import glob
import os
from typing import List

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, TENANT_ID, TOP_K
from embeddings import embeddings


def _to_passage(text: str) -> str:
    """Wrap raw document text so FAISS stores uniform 'passage' entries."""
    return f"passage: {text.strip()}"


def log_data_dir_contents() -> None:
    """Log discovered data files for the tenant, highlighting missing inputs."""
    print(f"[RAG] Tenant '{TENANT_ID}' using data dir: {DATA_DIR}")
    if not os.path.isdir(DATA_DIR):
        print(f"[RAG] Data dir missing for tenant '{TENANT_ID}'")
        return
    txt_files = sorted(glob.glob(os.path.join(DATA_DIR, "**/*.txt"), recursive=True))
    if not txt_files:
        print(f"[RAG] No .txt files found under {DATA_DIR}")
        return
    print(f"[RAG] Found {len(txt_files)} .txt files:")
    for fp in txt_files:
        rel = os.path.relpath(fp, DATA_DIR)
        size = os.path.getsize(fp)
        print(f"  - {rel} ({size} bytes)")


def build_vector_store() -> FAISS:
    """Create an in-memory FAISS index from tenant documents."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "**/*.txt"), recursive=True))

    docs: List[Document] = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as source_file:
                content = source_file.read()
        except Exception:
            continue
        if not content.strip():
            continue
        docs.append(
            Document(
                page_content=_to_passage(content),
                metadata={"tenant": TENANT_ID, "source": os.path.basename(path)},
            )
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs) if docs else []

    embedding_dim = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    if chunks:
        print(f"[RAG] Building FAISS index from {len(chunks)} document chunks (in-memory).")
        vector_store.add_documents(chunks)
    else:
        print("[RAG] No documents found; creating placeholder index.")
        vector_store.add_texts(
            ["passage: "], metadatas=[{"tenant": TENANT_ID, "source": "empty"}]
        )
    try:
        print(f"[RAG] Finished indexing; ntotal={getattr(vector_store.index, 'ntotal', 'unknown')}")
    except Exception:
        pass
    return vector_store


def make_retriever(vs: FAISS):
    """Produce a vector retriever tuned to the configured top-k setting."""
    return vs.as_retriever(search_kwargs={"k": TOP_K})


__all__ = ["build_vector_store", "make_retriever", "log_data_dir_contents"]
