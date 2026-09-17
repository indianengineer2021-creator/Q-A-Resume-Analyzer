import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from src.config import get_settings
from src.embeddings import get_embeddings
from src.pdf_processor import clean_text, create_chunks, extract_pdf_text
from src.rag_chain import generate_answer, get_llm
from src.vector_store import build_vector_store, get_retriever

load_dotenv()
settings = get_settings()

st.set_page_config(page_title="Resume Q&A Assistant", page_icon="📄", layout="wide")


def normalize_question(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value).strip()
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def init_session_state() -> None:
    defaults = {
        "resume_text": "",
        "chunks": [],
        "vectorstore": None,
        "retriever": None,
        "resume_name": "",
        "chat_history": [],
        "question_input": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_session_state()


def process_uploaded_resume(uploaded_file) -> None:
    if uploaded_file is None:
        return

    file_name = uploaded_file.name or "resume.pdf"
    if not file_name.lower().endswith(".pdf"):
        st.error("Please upload a valid PDF file.")
        return

    try:
        raw_text = extract_pdf_text(uploaded_file)
    except Exception as exc:  # pragma: no cover - UI path
        st.error(f"Unable to read the PDF: {exc}")
        return

    if not raw_text or not raw_text.strip():
        st.error("The uploaded PDF does not contain any readable text.")
        return

    cleaned_text = clean_text(raw_text)
    chunks = create_chunks(cleaned_text)

    if not chunks:
        st.error("The resume could not be divided into searchable chunks.")
        return

    try:
        embeddings = get_embeddings()
        vectorstore = build_vector_store(chunks, embeddings)
        retriever = get_retriever(vectorstore, settings["top_k"])
    except Exception as exc:  # pragma: no cover - UI path
        st.error(f"The resume could not be indexed for search: {exc}")
        return

    st.session_state["resume_text"] = cleaned_text
    st.session_state["chunks"] = chunks
    st.session_state["vectorstore"] = vectorstore
    st.session_state["retriever"] = retriever
    st.session_state["resume_name"] = file_name

    st.success("✅ Resume processed successfully")
    st.info(f"📊 {len(chunks)} chunks created")
    st.caption("🗄️ ChromaDB knowledge base created")


def show_chat_history() -> None:
    if not st.session_state.get("chat_history"):
        return

    st.subheader("💬 Chat History")
    for item in st.session_state["chat_history"]:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])


def main() -> None:
    st.title("📄 Resume Q&A Assistant")
    st.caption("Ask grounded questions about an uploaded resume using Gemini and RAG.")

    uploaded_file = st.file_uploader(
        "Upload your Resume PDF",
        type=["pdf"],
        help="Upload a PDF resume to extract text and build the knowledge base.",
    )

    if uploaded_file is not None:
        process_uploaded_resume(uploaded_file)

    if st.session_state.get("retriever") is None:
        st.info("Upload a PDF resume to begin asking questions.")

    sample_questions = [
        "What are my technical skills?",
        "How many years of project management experience do I have?",
        "What companies have I worked for?",
        "What certifications do I have?",
        "What projects have I worked on?",
        "What Agile experience do I have?",
        "What experience do I have with Python?",
        "What experience do I have with AI or GenAI?",
        "What are my strongest skills?",
        "What are my educational qualifications?",
    ]

    st.subheader("Ask a question about your resume")
    sample_question = st.selectbox(
        "Example questions",
        ["Select a sample question..."] + sample_questions,
        index=0,
    )

    if sample_question != "Select a sample question...":
        st.session_state["question_input"] = normalize_question(sample_question)

    question = st.text_input(
        "Your question",
        key="question_input",
        value=st.session_state.get("question_input", ""),
        placeholder="For example: What projects have I worked on?",
    )

    if st.button("Ask Question"):
        user_question = normalize_question(question)

        if not st.session_state.get("retriever"):
            st.error("Please upload and process a resume before asking a question.")
        elif not user_question:
            st.error("Please enter a question before submitting.")
        else:
            try:
                llm = get_llm()
                answer, docs = generate_answer(user_question, st.session_state["retriever"], llm)
            except ValueError as exc:
                st.error(str(exc))
                return
            except Exception as exc:  # pragma: no cover - UI path
                st.error(f"The question could not be answered: {exc}")
                return

            st.subheader("🤖 Answer")
            st.write(answer)
            st.caption(f"Sources: Resume PDF | Retrieved chunks: {len(docs)}")

            st.markdown("---")
            with st.expander("🔎 Retrieved Resume Context", expanded=False):
                if docs:
                    for index, doc in enumerate(docs, start=1):
                        st.markdown(f"### Retrieved Chunk {index}")
                        st.write(doc.page_content)
                        st.markdown("---")
                else:
                    st.info("No relevant resume chunks were retrieved.")

            st.session_state["chat_history"].append({"role": "user", "content": user_question})
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})

    show_chat_history()


if __name__ == "__main__":
    main()
