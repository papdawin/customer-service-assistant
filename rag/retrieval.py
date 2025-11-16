def retrieve_documents(retriever, query: str):
    """Query the retriever, preferring `get_relevant_documents` and falling back to `invoke`."""
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    return retriever.invoke(query)


__all__ = ["retrieve_documents"]
