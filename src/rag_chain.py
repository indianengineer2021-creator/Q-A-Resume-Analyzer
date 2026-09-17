from typing import Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_settings


def extract_response_text(response) -> str:
    """Convert Gemini's text or structured content into displayable text."""
    content = response.content if hasattr(response, "content") else response

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
        return "\n".join(text_parts).strip()

    return str(content).strip()


def get_llm():
    """Return the configured Gemini Flash model for grounded generation."""
    settings = get_settings()
    if not settings["google_api_key"]:
        raise ValueError("Missing GOOGLE_API_KEY environment variable.")

    return ChatGoogleGenerativeAI(
        model=settings["gemini_model"],
        temperature=0.2,
        google_api_key=settings["google_api_key"],
    )


def build_rag_prompt():
    """Create the system prompt used to ground answers in retrieved resume context."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI Resume Q&A Assistant.

Your task is to answer the user's question using ONLY the information contained in the provided resume context.

Rules:
- Do not invent information.
- Do not make assumptions about the candidate.
- Do not infer skills or experience that are not supported by the resume.
- If the requested information is not available in the provided context, say: "I could not find this information in the resume."
- Keep the response professional and concise.
- Base every answer on the retrieved resume context.
- When possible, mention the relevant evidence from the resume.

Resume Context:
{context}

Answer only from that context.""",
            ),
            ("human", "{question}"),
        ]
    )


def generate_answer(question: str, retriever, llm) -> Tuple[str, list]:
    """Retrieve relevant chunks and generate an answer grounded in them."""
    if not isinstance(question, str):
        question = str(question)
    question = question.strip()
    if not question:
        raise ValueError("Please provide a valid question.")

    documents = retriever.invoke(question)
    print("Question type:", type(question))
    print("Documents type:", type(documents))
    print("First document type:", type(documents[0]) if documents else None)
    print(
        "First page_content type:",
        type(documents[0].page_content) if documents else None,
    )

    if not documents:
        return "I could not find this information in the resume.", []

    context = "\n\n".join(
        doc.page_content
        for doc in documents
        if hasattr(doc, "page_content") and isinstance(doc.page_content, str)
    )
    print("Context type:", type(context))

    if not context:
        return "I could not find this information in the resume.", documents

    prompt = build_rag_prompt()
    prompt_value = prompt.invoke({"context": context, "question": question})
    print("Prompt value type:", type(prompt_value))
    response = llm.invoke(prompt_value)
    print("Response content type:", type(response.content))
    answer = extract_response_text(response)
    return answer, documents
