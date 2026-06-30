from typing import List


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Splits a long string of text into overlapping chunks.

    chunk_size: number of characters per chunk
    overlap: number of characters repeated between consecutive chunks
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())

        # Move start forward, but back up by 'overlap' so chunks share context
        start = end - overlap

    # Remove any empty chunks that might have been created
    chunks = [c for c in chunks if c]

    return chunks