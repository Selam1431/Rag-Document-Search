# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

This project requires Ollama running locally with two models pulled:

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

Python dependencies must be installed in a virtual environment (system Python is externally managed on Debian/Ubuntu):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the App

**Web UI (primary entry point):**
```bash
streamlit run streamlit_app.py
```

**CLI pipeline** (loads files from `data/docs/`, then enters interactive Q&A loop):
```bash
python app.py
```

## Architecture

The project has two separate execution paths:

### Web UI path (`streamlit_app.py`)
Self-contained — does everything inline: reads an uploaded PDF, chunks it, embeds chunks via `ollama.embeddings(model="nomic-embed-text")`, stores them in an **in-memory** ChromaDB collection (ephemeral, reset on each upload), then answers queries using `ollama.chat(model="gemma3:4b")`. No state persists between sessions.

### CLI pipeline path (`app.py` → `load_documents.py` → `chat_rag.py`)
- `load_documents.py` — reads `.txt` and `.pdf` files from a folder, chunks text (500 chars, 100 char overlap), returns list of `{id, text}` dicts
- `chat_rag.py` — `ask_rag(collection)` runs an interactive terminal loop: embeds the query, retrieves top-3 chunks from ChromaDB, generates an answer with the LLM

### Key parameters
- Chunk size: 500 characters, overlap: 100 characters
- Embedding model: `nomic-embed-text` (via Ollama)
- LLM: `gemma3:4b` (via Ollama)
- Vector DB: ChromaDB (in-memory, collection named `"documents"`)
- Retrieval: top-3 nearest chunks by cosine similarity

### Note on `embed_store.py`
This file currently contains a full Streamlit app (a duplicate of an older version of `streamlit_app.py`), not a `store_embeddings` utility function. The CLI pipeline in `app.py` imports `store_embeddings` from it, which means `app.py` is currently broken. If restoring CLI functionality, `embed_store.py` needs to export a `store_embeddings(documents)` function that embeds and loads chunks into ChromaDB.
