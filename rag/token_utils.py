from typing import List

from langchain_core.documents import Document

from config import (
    APPROX_CHARS_PER_TOKEN,
    CONTEXT_TOKENS_BUDGET,
    PER_DOC_CHAR_CAP,
    PER_DOC_TOKEN_CAP,
    TOKENIZER_NAME,
)

try:
    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
except Exception:
    _tokenizer = None


def _truncate_chars(text: str, max_chars: int) -> str:
    """Trim text to the allowed length, appending an ellipsis when clipped."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def _pack_docs_token_budget(docs: List[Document], max_tokens: int) -> str:
    """Assemble source-tagged snippets within the token budget, falling back to char limits."""
    if _tokenizer is None:
        pieces = []
        remaining = max_tokens * APPROX_CHARS_PER_TOKEN
        for doc in docs:
            if remaining <= 0:
                break
            src = doc.metadata.get("source", "unknown")
            snippet = _truncate_chars(
                doc.page_content, min(PER_DOC_CHAR_CAP, max(0, remaining))
            )
            piece = f"[{src}] {snippet}"
            pieces.append(piece)
            remaining -= len(piece)
        return "\n\n".join(pieces)

    used = 0
    pieces = []
    for doc in docs:
        if used >= max_tokens:
            break
        src = doc.metadata.get("source", "unknown")

        tokens = _tokenizer.encode(doc.page_content, add_special_tokens=False)
        if not tokens:
            continue
        tokens = tokens[: max(0, min(PER_DOC_TOKEN_CAP, max_tokens - used))]
        snippet = _tokenizer.decode(tokens)

        piece = f"[{src}] {snippet}"
        piece_tok_len = len(_tokenizer.encode(piece, add_special_tokens=False))
        if used + piece_tok_len > max_tokens:
            room = (
                max_tokens
                - used
                - len(_tokenizer.encode(f"[{src}] ", add_special_tokens=False))
            )
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
    """Condense retrieved docs into the configured context window."""
    return _pack_docs_token_budget(docs, CONTEXT_TOKENS_BUDGET)


def tokenizer_loaded() -> bool:
    """Report whether the configured tokenizer was successfully initialized."""
    return _tokenizer is not None


__all__ = ["format_docs", "tokenizer_loaded"]
