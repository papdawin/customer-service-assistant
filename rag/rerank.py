from typing import List

from langchain_core.documents import Document

from config import RERANK_MODEL, RERANK_TOP_N, USE_RERANK

_reranker = None
if USE_RERANK:
    try:
        from FlagEmbedding import FlagReranker  # type: ignore

        _reranker = FlagReranker(RERANK_MODEL, use_fp16=True)
    except Exception:
        _reranker = None


def maybe_rerank(query: str, docs: List[Document], top_n: int = RERANK_TOP_N) -> List[Document]:
    """Return the top documents, reranking with the optional cross-encoder when available."""
    if not _reranker:
        return docs[:top_n] if top_n and top_n < len(docs) else docs
    pairs = [[query, doc.page_content] for doc in docs]
    try:
        scores = _reranker.compute_score(pairs, normalize=True)
    except Exception:
        return docs[:top_n] if top_n and top_n < len(docs) else docs
    ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]


def reranker_loaded() -> bool:
    """Signal whether the optional reranker model loaded successfully."""
    return _reranker is not None


__all__ = ["maybe_rerank", "reranker_loaded"]
