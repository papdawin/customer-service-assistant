from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBED_MODEL

# Use normalized embeddings so cosine similarity works well with FAISS.
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)

__all__ = ["embeddings"]
