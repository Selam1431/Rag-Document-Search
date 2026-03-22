import os
import re
import tempfile

import chromadb
import ollama
import requests
import streamlit as st
from pypdf import PdfReader

from config import (
    CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL, LLM_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, MAX_QUERY_LENGTH, MAX_FILE_SIZE_MB,
    OLLAMA_BASE_URL
)

PDF_MAGIC_BYTES = b"%PDF"


def check_ollama():
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        missing = [m for m in (EMBED_MODEL, LLM_MODEL) if not any(m in name for name in models)]
        return missing if missing else True
    except requests.exceptions.ConnectionError:
        return None

st.set_page_config(
    page_title="AI Document Chat",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f8fafc, #eef2ff);
    }

    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        text-align: center;
        color: #111827;
        margin-bottom: 0.3rem;
    }

    .main-subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 2rem;
    }

    .hero-box {
        background: linear-gradient(135deg, #111827, #1e3a8a);
        padding: 2rem;
        border-radius: 22px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin-bottom: 2rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .hero-text {
        font-size: 1rem;
        color: #dbeafe;
    }

    .card {
        background: white;
        padding: 1.3rem;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }

    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.5rem;
    }

    .card-text {
        color: #4b5563;
        font-size: 0.96rem;
        line-height: 1.6;
    }

    .answer-box {
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        padding: 1.3rem;
        border-radius: 18px;
        border-left: 6px solid #2563eb;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        color: #111827;
        font-size: 1rem;
        line-height: 1.7;
        margin-top: 0.8rem;
    }

    .metric-box {
        background: white;
        border-radius: 18px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        border: 1px solid #e5e7eb;
    }

    .metric-number {
        font-size: 1.6rem;
        font-weight: 800;
        color: #1d4ed8;
    }

    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
    }

    .section-label {
        font-size: 1.2rem;
        font-weight: 700;
        color: #111827;
        margin-top: 1rem;
        margin-bottom: 0.7rem;
    }

    .footer-note {
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🤖 AI Document Chat")
    st.write("A portfolio-ready RAG app for searching and chatting with PDF documents.")
    st.markdown("---")
    st.markdown("### Tech Stack")
    st.write("• Python")
    st.write("• Ollama")
    st.write("• ChromaDB")
    st.write("• Streamlit")
    st.write("• PyPDF")
    st.markdown("---")
    st.markdown("### Workflow")
    st.write("1. Upload PDF")
    st.write("2. Extract text")
    st.write("3. Chunk document")
    st.write("4. Create embeddings")
    st.write("5. Retrieve context")
    st.write("6. Generate answer")

ollama_status = check_ollama()
if ollama_status is None:
    st.error(
        f"Cannot connect to Ollama at `{OLLAMA_BASE_URL}`. "
        "Make sure Ollama is running: `ollama serve`"
    )
    st.stop()
elif isinstance(ollama_status, list):
    st.error(
        f"Required Ollama models not found: {', '.join(ollama_status)}. "
        f"Run: `ollama pull {' && ollama pull '.join(ollama_status)}`"
    )
    st.stop()

st.markdown('<div class="main-title">AI Document Chat</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Upload a PDF, retrieve relevant context with embeddings, and generate grounded answers with a local LLM.</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero-box">
    <div class="hero-title">Retrieval-Augmented Generation Demo</div>
    <div class="hero-text">
        This app processes uploaded PDF documents, stores semantic embeddings in a vector database,
        retrieves the most relevant chunks, and answers questions using a local language model.
    </div>
</div>
""", unsafe_allow_html=True)

top1, top2, top3 = st.columns(3)

with top1:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-number">PDF</div>
        <div class="metric-label">Document Input</div>
    </div>
    """, unsafe_allow_html=True)

with top2:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-number">RAG</div>
        <div class="metric-label">Semantic Retrieval</div>
    </div>
    """, unsafe_allow_html=True)

with top3:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-number">LLM</div>
        <div class="metric-label">Generated Answers</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-label">Upload Document</div>', unsafe_allow_html=True)

left, right = st.columns([1.2, 1])

with left:
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

with right:
    st.markdown("""
    <div class="card">
        <div class="card-title">How this app works</div>
        <div class="card-text">
            The uploaded PDF is converted into text, split into chunks, embedded using a local model,
            stored in ChromaDB, and searched semantically to generate grounded answers.
        </div>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file:
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB.")
        st.stop()

    if uploaded_file.read(4) != PDF_MAGIC_BYTES:
        st.error("Invalid file. Only real PDF files are accepted.")
        st.stop()
    uploaded_file.seek(0)

    with st.spinner("Processing document..."):
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            reader = PdfReader(temp_path)
            text = ""

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text.replace("\n", " ") + " "

            text = " ".join(text.split())

            if not text.strip():
                st.error("Could not extract text from this PDF. It may be scanned or password-protected.")
                st.stop()

            chunks = []
            start = 0
            while start < len(text):
                end = start + CHUNK_SIZE
                chunks.append(text[start:end])
                start += CHUNK_SIZE - CHUNK_OVERLAP

            client = chromadb.PersistentClient(path=CHROMA_PATH)

            try:
                client.delete_collection(name=COLLECTION_NAME)
            except chromadb.errors.NotFoundError:
                pass

            collection = client.create_collection(name=COLLECTION_NAME)

            for i, chunk in enumerate(chunks):
                embedding = ollama.embeddings(
                    model=EMBED_MODEL,
                    prompt=chunk
                )["embedding"]

                collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    ids=[str(i)]
                )
        except Exception as e:
            st.error(f"Failed to process document: {e}")
            st.stop()
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    st.success("Document processed successfully. Ask a question below.")

    q1, q2 = st.columns([2, 1])

    with q1:
        st.markdown('<div class="section-label">Ask a Question</div>', unsafe_allow_html=True)
        query = st.text_input("Type your question about the uploaded PDF")

    with q2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Suggested prompts</div>
            <div class="card-text">
                • Summarize this document<br>
                • What are the main ideas?<br>
                • What risks are mentioned?<br>
                • What recommendations are given?
            </div>
        </div>
        """, unsafe_allow_html=True)

    if query:
        query = query.strip()
        if len(query) > MAX_QUERY_LENGTH:
            st.error(f"Query too long. Maximum {MAX_QUERY_LENGTH} characters.")
        elif not re.search(r'\w', query):
            st.error("Query must contain at least one word.")
        else:
            with st.spinner("Searching and generating answer..."):
                try:
                    query_embedding = ollama.embeddings(
                        model=EMBED_MODEL,
                        prompt=query
                    )["embedding"]

                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=3
                    )

                    docs = results.get("documents", [[]])
                    if not docs or not docs[0]:
                        st.warning("No relevant context found for your query.")
                    else:
                        retrieved_chunks = docs[0]
                        context = "\n\n".join(retrieved_chunks)

                        prompt = (
                            "You are a helpful AI assistant.\n"
                            "Answer the user's question using only the context below.\n"
                            "Keep the answer clear, concise, and professional.\n"
                            "If the context does not contain enough information, say so.\n\n"
                            f"Context:\n{context}\n\n"
                            f"Question:\n{query}"
                        )

                        response = ollama.chat(
                            model=LLM_MODEL,
                            messages=[{"role": "user", "content": prompt}]
                        )

                        answer = response["message"]["content"]

                        st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                        with st.expander("View Retrieved Context"):
                            st.write(context)
                except Exception as e:
                    st.error(f"Failed to generate answer: {e}")

st.markdown('<div class="footer-note">Built with Streamlit, Ollama, and ChromaDB</div>', unsafe_allow_html=True)