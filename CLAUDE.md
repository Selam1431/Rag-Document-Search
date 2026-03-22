# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Backends

The app supports two backend modes selected by env vars:

| Backend | LLM | Embeddings | When |
|---|---|---|---|
| **Cloud** | Groq (`llama-3.1-8b-instant`) | Cohere (`embed-english-v3.0`) | `GROQ_API_KEY` + `COHERE_API_KEY` set |
| **Local** | Ollama (`gemma3:4b`) | Ollama (`nomic-embed-text`) | No API keys set |

Cloud mode is required for Render deployment. Local mode requires Ollama running.

## Setup

### Local (Ollama)

Requires Ollama running locally with two models:

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

Python dependencies via virtual environment (system Python is externally managed on Debian/Ubuntu):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Cloud (Groq + Cohere)

Set `GROQ_API_KEY` and `COHERE_API_KEY` in `.env` — no Ollama needed.

## Commands

```bash
# Web UI
streamlit run streamlit_app.py

# CLI (loads data/docs/, interactive Q&A with streaming)
python app.py

# Tests (no Ollama or ChromaDB needed)
python -m pytest tests/ -v
```

## Architecture

### Module responsibilities

| File | Role |
|---|---|
| `config.py` | Single source of truth — loads all settings from `.env` via `python-dotenv` |
| `rag_core.py` | Shared library: PDF extraction, sentence-aware chunking, concurrent embedding, prompt building, streaming and non-streaming LLM calls |
| `streamlit_app.py` | Web UI entry point — chat interface, upload, auth, rate limiting |
| `app.py` | CLI entry point — Ollama health check, loads `data/docs/`, calls embed_store + chat_rag |
| `embed_store.py` | `store_embeddings(documents, source)` — embeds and stores chunks with metadata, replaces existing chunks for same source |
| `load_documents.py` | Loads `.txt`, `.pdf`, `.docx` from a folder, returns list of `{id, text}` dicts |
| `chat_rag.py` | CLI Q&A loop — multi-turn with conversation history, streaming output to stdout |
| `logger.py` | `setup_logging()` — configures logging level from `LOG_LEVEL` env var |

### Two execution paths

**Web UI** (`streamlit_app.py`):
- Upload → validate (size + magic bytes) → extract text → sentence-chunk → concurrent embed → store in ChromaDB with metadata
- Query → embed (cached per session) → retrieve top-3 with distances → stream LLM answer → display with relevance scores

**CLI** (`app.py` → `load_documents` → `embed_store` → `chat_rag`):
- Loads all files from `data/docs/` at startup, embeds and stores them
- Interactive loop: embed query → retrieve → stream answer to stdout → maintain conversation history

### Key design decisions

- **Sentence-aware chunking**: `split_into_chunks()` splits on `.!?` boundaries, groups sentences up to `CHUNK_SIZE`, falls back to character chunking for sentences longer than the limit.
- **Persistent ChromaDB**: `PersistentClient(path=CHROMA_PATH)` — embeddings survive restarts. Per-document replacement: existing chunks for a `source` are deleted before re-embedding.
- **Metadata on every chunk**: `source` (filename), `uploaded_at` (UTC ISO), `chunk_index` stored in ChromaDB for attribution and per-document deletion.
- **Concurrent embedding**: `embed_chunks_concurrent()` uses `ThreadPoolExecutor(max_workers=4)` with an optional `progress_callback` for the UI progress bar.
- **Dual-backend streaming**: `stream_answer()` branches on `GROQ_API_KEY` — Groq SDK if set, `ollama.chat(stream=True)` otherwise. Web UI uses `st.write_stream()`, CLI writes to `sys.stdout` directly.
- **Dual-backend embedding**: `embed_text()` / `embed_chunks_concurrent()` use Cohere batch API (single HTTP call) if `COHERE_API_KEY` set, else concurrent Ollama via `ThreadPoolExecutor`.
- **Auth**: Optional — only active when `APP_PASSWORD` env var is non-empty. Uses `st.session_state.authenticated`.
- **Rate limiting**: Sliding 60-second window tracked in `st.session_state.request_times`.

### Configuration

All tunable values live in `.env` (gitignored) and are loaded once by `config.py`. `.env.example` is the committed template. Never hardcode model names, paths, or limits in module files — always import from `config`.

Key variables: `LLM_MODEL`, `EMBED_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MAX_FILE_SIZE_MB`, `MAX_QUERY_LENGTH`, `RATE_LIMIT_PER_MINUTE`, `MAX_HISTORY_TURNS`, `APP_PASSWORD`, `LOG_LEVEL`, `CHROMA_PATH`, `OLLAMA_BASE_URL`, `GROQ_API_KEY`, `GROQ_MODEL`, `COHERE_API_KEY`.

### Docker / Render

`Dockerfile` builds the app image (python:3.12-slim). `docker-compose.yml` runs both the app and an Ollama container together with named volumes for ChromaDB and Ollama model data. `.dockerignore` excludes `.env`, `venv/`, `chroma_db/`, and `.git/`.

`render.yaml` configures a free-tier Render web service using Docker. `GROQ_API_KEY` and `COHERE_API_KEY` are `sync: false` (must be set in the Render dashboard). `CHROMA_PATH` is `/tmp/chroma_db` (ephemeral on free tier).
