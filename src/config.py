import os
from typing import Dict

from dotenv import load_dotenv


load_dotenv()


def get_settings() -> Dict[str, str | int]:
    return {
        "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "800")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "150")),
        "top_k": int(os.getenv("TOP_K", "4")),
    }
