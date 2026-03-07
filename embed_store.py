import streamlit as st
import ollama
import chromadb
from pypdf import PdfReader
import tempfile

st.set_page_config(
    page_title="AI Document Chat",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .title {
        font-size: 3rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
        text-align: center;
    }

    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #111827;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .answer-box {
        background-color: #f9fafb;
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        margin-top: 0.75rem;
        color: #111827;
    }

    .info-card {
        background-color: #f3f4f6;
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }

    .small-text {
        color: #6b7280;
        font-size: 0.95rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">AI Document Chat</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a PDF and ask questions using a Retrieval-Augmented Generation (RAG) system powered by local embeddings, ChromaDB, and a local LLM.</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="section-header">Upload Document</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-card"><div class="small-text">Upload a PDF file to extract text, create embeddings, store document chunks in a vector database, and generate answers grounded in the uploaded content.</div></div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

with col2:
    st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
            <div class="small-text">
            1. Upload a PDF<br>
            2. Extract and chunk the document text<br>
            3. Generate embeddings with Ollama<br>
            4. Store chunks in ChromaDB<br>
            5. Retrieve the most relevant context<br>
            6. Generate an answer with Gemma
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

if uploaded_file:
    with st.spinner("Processing document..."):
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

        chunk_size = 500
        overlap = 100
        chunks = []

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

        client = chromadb.Client()

        try:
            client.delete_collection(name="documents")
        except:
            pass

        collection = client.create_collection(name="documents")

        for i, chunk in enumerate(chunks):
            embedding = ollama.embeddings(
                model="nomic-embed-text",
                prompt=chunk
            )["embedding"]

            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[str(i)]
            )

    st.success("Document processed successfully. You can now ask questions.")

    st.markdown('<div class="section-header">Ask a Question</div>', unsafe_allow_html=True)
    query = st.text_input("Enter your question about the document")

    if query:
        with st.spinner("Searching and generating answer..."):
            query_embedding = ollama.embeddings(
                model="nomic-embed-text",
                prompt=query
            )["embedding"]

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )

            retrieved_chunks = results["documents"][0]
            context = "\n\n".join(retrieved_chunks)

            prompt = f"""
You are a helpful AI assistant.
Answer the user's question using only the context below.
Keep the answer clear, concise, and professional.

Context:
{context}

Question:
{query}
"""

            response = ollama.chat(
                model="gemma3:4b",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            answer = response["message"]["content"]

        st.markdown('<div class="section-header">Answer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        with st.expander("Retrieved Context"):
            st.write(context)