import re
from typing import IO, Optional

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings


def extract_pdf_text(pdf_file: IO[bytes]) -> str:
    """Read a PDF file, extract text from all pages, and combine the results."""
    try:
        reader = PdfReader(pdf_file)
    except Exception as exc:  # pragma: no cover - exercised via UI
        raise ValueError(f"Invalid PDF file: {exc}") from exc

    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    if not text_parts:
        return ""

    return "\n\n".join(text_parts)


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving meaningful resume content."""
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)
    cleaned = re.sub(r"\n +", "\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned


def create_chunks(text: str, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
    """Split a cleaned resume into smaller, retrievable chunks."""
    settings = get_settings()
    chunk_size = chunk_size or settings["chunk_size"]
    chunk_overlap = chunk_overlap or settings["chunk_overlap"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)
