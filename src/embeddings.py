import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import get_settings


def get_embeddings():
    """Return a Gemini embedding model configured through environment variables."""
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=settings["embedding_model"],
        google_api_key=settings["google_api_key"] or os.getenv("GOOGLE_API_KEY"),
    )
