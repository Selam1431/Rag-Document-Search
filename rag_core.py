"""
Shared RAG utilities used by all entry points (streamlit_app, embed_store, chat_rag).
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ollama
from pypdf import PdfReader

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL, LLM_MODEL

logger = logging.getLogger(__name__)


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text.replace("\n", " ") + " "
    text = " ".join(text.split())
    logger.debug("Extracted %d characters from %s", len(text), path)
    return text


def split_into_chunks(text: str) -> list:
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE]
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    logger.debug("Split text into %d chunks", len(chunks))
    return chunks


def embed_text(text: str) -> list:
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def embed_chunks_concurrent(chunks: list, max_workers: int = 4) -> list:
    """Embed all chunks in parallel using a thread pool. Returns embeddings in original order."""
    logger.info("Embedding %d chunks with %d workers", len(chunks), max_workers)
    t0 = time.time()
    results = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(embed_text, chunk): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()

    elapsed = time.time() - t0
    logger.info("Embedded %d chunks in %.2fs", len(chunks), elapsed)
    return results


def make_metadata(source: str, uploaded_at: str, chunk_index: int) -> dict:
    return {"source": source, "uploaded_at": uploaded_at, "chunk_index": chunk_index}


def build_prompt(context: str, query: str) -> str:
    return (
        "You are a helpful AI assistant.\n"
        "Answer the user's question using only the context below.\n"
        "Keep the answer clear, concise, and professional.\n"
        "If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}"
    )


def generate_answer(prompt: str) -> str:
    logger.info("Sending prompt to LLM (%s)", LLM_MODEL)
    t0 = time.time()
    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    elapsed = time.time() - t0
    logger.info("LLM responded in %.2fs", elapsed)
    return response["message"]["content"]
