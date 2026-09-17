import os

from langchain_chroma import Chroma


def build_vector_store(chunks, embeddings, collection_name: str = "resume_collection"):
    """Create a Chroma vector store from the resume chunks."""
    persist_directory = os.path.join(os.getcwd(), "chroma_db")
    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )


def get_retriever(vectorstore, k: int = 4):
    """Return a retriever that fetches the most relevant chunks."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
