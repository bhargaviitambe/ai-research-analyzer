import fitz  # PyMuPDF
from typing import Dict


def extract_text_from_pdf(file_path: str) -> str:
    """
    Opens a PDF file and extracts all text content from every page.
    """
    doc = fitz.open(file_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text()

    doc.close()
    return full_text


def extract_metadata(file_path: str) -> Dict:
    """
    Extracts basic metadata like title and author if available
    in the PDF's internal metadata.
    """
    doc = fitz.open(file_path)
    metadata = doc.metadata
    doc.close()

    return {
        "title": metadata.get("title", "Unknown"),
        "author": metadata.get("author", "Unknown"),
    }