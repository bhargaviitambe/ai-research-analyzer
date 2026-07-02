import chromadb
from chromadb.config import Settings
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# Creates a persistent client - data is saved to disk, not lost on restart
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_or_create_collection(paper_id: str):
    """
    Each uploaded paper gets its own 'collection' (like a separate table)
    so chunks from different papers never mix together.
    """
    collection = client.get_or_create_collection(name=f"paper_{paper_id}")
    return collection


def store_chunks(paper_id: str, chunks: List[str], embeddings: List[List[float]]):
    """
    Saves text chunks and their embeddings into the vector database.
    """
    collection = get_or_create_collection(paper_id)

    # ChromaDB requires a unique ID for each chunk
    ids = [f"{paper_id}_chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )


def query_similar_chunks(paper_id: str, query_embedding: List[float], top_k: int = 3) -> List[str]:
    """
    Finds the most relevant chunks for a given question's embedding.
    Returns the top_k most similar chunks of text.
    """
    collection = get_or_create_collection(paper_id)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # results['documents'] is a list of lists (one list per query) - we only sent 1 query
    return results['documents'][0]