"""
rag.py — Contextual Retrieval backend shared by the Streamlit app.

Loads the pre-built contextual FAISS index produced by the notebook and
exposes a single public function: answer_with_sources().

The FAISS index path is resolved in this priority order:
  1. FAISS_INDEX_PATH environment variable  (used inside Docker)
  2. ../index/contextual_faiss relative to this file  (local dev)
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL   = "gpt-4o-mini"
INDEX_PATH  = Path(
    os.getenv(
        "FAISS_INDEX_PATH",
        str(Path(__file__).parent.parent / "index" / "contextual_faiss"),
    )
)

_db: FAISS | None = None   # module-level singleton — loaded once per process


def _get_db() -> FAISS:
    global _db
    if _db is None:
        embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        _db = FAISS.load_local(
            str(INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return _db


def answer_with_sources(question: str, k: int = 4) -> tuple[str, list[str]]:
    """
    Retrieve the top-k most relevant enriched chunks and generate an answer.

    Returns
    -------
    answer  : str          — the generated answer from gpt-4o-mini
    sources : list[str]    — the k retrieved chunk texts (for citation display)
    """
    db   = _get_db()
    docs = db.similarity_search(question, k=k)
    ctx  = [d.page_content for d in docs]

    context = "\n\n---\n\n".join(ctx)
    prompt = (
        "Answer the following question using ONLY the provided context.\n"
        "Format your response using Markdown:\n"
        "  - Use **bold** for key terms.\n"
        "  - Use bullet lists where appropriate.\n"
        "  - Write ALL mathematical expressions in LaTeX notation:\n"
        "      * Inline math  : $...$ (e.g., $\\cos(\\theta)$)\n"
        "      * Display math : $$...$$ on its own line (e.g., $$\\cos(v,w)=\\frac{v \\cdot w}{|v||w|}$$)\n"
        "Be concise (3–5 sentences). "
        "If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    client = OpenAI()
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip(), ctx
