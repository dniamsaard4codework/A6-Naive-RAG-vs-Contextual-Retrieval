"""
app.py — Streamlit chat interface for the Chapter 5 RAG chatbot.

Backend: Contextual Retrieval via app/rag.py
   - Embedding : text-embedding-3-small
   - Generator : gpt-4o-mini

Run locally:
    cd app && streamlit run app.py

Run via Docker Compose (from repo root):
    docker-compose up --build
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()   # pick up OPENAI_API_KEY from .env when running locally

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SLP3 Ch.5 RAG Chatbot",
    page_icon="📖",
    layout="centered",
)

# ── Load RAG backend (cached so FAISS index is only loaded once per session) ───
@st.cache_resource(show_spinner="Loading FAISS index …")
def load_rag():
    from rag import answer_with_sources   # import here so cache_resource sees it
    return answer_with_sources

answer_with_sources = load_rag()

# ── UI header ──────────────────────────────────────────────────────────────────
st.title("📖 Chapter 5: Vector Semantics and Embeddings")
st.caption(
    "RAG Chatbot · Contextual Retrieval · "
    "`text-embedding-3-small` · `gpt-4o-mini`"
)
st.markdown(
    "Ask any question about **Chapter 5: Vector Semantics and Embeddings** from the "
    "Stanford *Speech and Language Processing* textbook. "
    "Answers are formatted with **Markdown** and **LaTeX equations** where applicable. "
    "Each answer cites the source chunk it was drawn from."
)
st.divider()
st.markdown(
    "Developed by Dechathon Niamsa-ard [st126235]"
)
st.divider()

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 Source chunks used", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {i}**")
                    st.markdown(src[:600] + ("…" if len(src) > 600 else ""))
                    if i < len(msg["sources"]):
                        st.divider()

# ── Input box ──────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about Chapter 5 …"):
    # Record user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating …"):
            try:
                answer, sources = answer_with_sources(prompt, k=4)
            except Exception as e:
                answer  = f"Error: {e}"
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("📄 Source chunks used", expanded=False):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.markdown(src[:600] + ("…" if len(src) > 600 else ""))
                    if i < len(sources):
                        st.divider()

    # Record assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
