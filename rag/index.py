import glob
import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    ALLOW_DANGEROUS_DESER,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    INDEX_DIR,
    TENANT_ID,
    TOP_K,
)
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
    txt_files = sorted(glob.glob(os.path.join(DATA_DIR, '**/*.txt'), recursive=True))
    if not txt_files:
        print(f"[RAG] No .txt files found under {DATA_DIR}")
        return
    print(f"[RAG] Found {len(txt_files)} .txt files:")
    for fp in txt_files:
        rel = os.path.relpath(fp, DATA_DIR)
        size = os.path.getsize(fp)
        print(f"  - {rel} ({size} bytes)")


def load_or_build_index(force_rebuild: bool = False) -> FAISS:
    """Load the tenant FAISS index from disk, rebuilding from source docs when needed."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss_path = os.path.join(INDEX_DIR, "faiss_index")

    files = sorted(glob.glob(os.path.join(DATA_DIR, "**/*.txt"), recursive=True))

    if (not force_rebuild) and os.path.exists(faiss_path):
        try:
            vs = FAISS.load_local(
                faiss_path,
                embeddings,
                allow_dangerous_deserialization=ALLOW_DANGEROUS_DESER,
            )
            doc_total = getattr(vs.index, "ntotal", None)
            sources = set()
            try:
                sources = {doc.metadata.get("source") for doc in vs.docstore._dict.values()}
            except Exception:
                pass

            placeholder_index = "empty" in sources or (doc_total == 0)
            if files and placeholder_index:
                print("[RAG] Existing FAISS index is placeholder; rebuilding from disk data.")
            else:
                return vs
        except Exception as exc:
            print(f"[RAG] Failed to load FAISS index (will rebuild): {exc!r}")

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

    if not docs:
        if files:
            print("[RAG] No non-empty documents found; creating placeholder index.")
        else:
            print("[RAG] No .txt files found; creating placeholder index.")
        vs = FAISS.from_texts(
            ["passage: "], embeddings, metadatas=[{"tenant": TENANT_ID, "source": "empty"}]
        )
        vs.save_local(faiss_path)
        return vs

    print(f"[RAG] Building FAISS index from {len(docs)} document chunks.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(faiss_path)
    try:
        print(f"[RAG] Finished indexing; ntotal={getattr(vs.index, 'ntotal', 'unknown')}")
    except Exception:
        pass
    return vs


def make_retriever(vs: FAISS):
    """Produce a vector retriever tuned to the configured top-k setting."""
    return vs.as_retriever(search_kwargs={"k": TOP_K})


__all__ = ["load_or_build_index", "make_retriever", "log_data_dir_contents"]
