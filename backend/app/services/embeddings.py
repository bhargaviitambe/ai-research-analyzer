from sentence_transformers import SentenceTransformer
from typing import List

# This downloads a small, fast embedding model the first time it's used.
# It then gets cached locally so future runs are instant.
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Converts a list of text chunks into a list of embedding vectors.
    """
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def generate_single_embedding(text: str) -> List[float]:
    """
    Converts a single piece of text (like a user's question) into one embedding vector.
    """
    embedding = model.encode([text], show_progress_bar=False)
    return embedding[0].tolist()