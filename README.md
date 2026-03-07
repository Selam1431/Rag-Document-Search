# AI Document Search System (RAG)

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to ask questions about documents.

The system retrieves relevant document sections using embeddings and a vector database, then generates answers using a local language model.

## Features

- Document ingestion from files
- Text chunking for better retrieval
- Local embeddings using Ollama
- Vector database using ChromaDB
- Semantic search
- AI-generated answers using a local LLM

## Tech Stack

- Python
- Ollama
- ChromaDB
- Local LLM (Gemma)
- Embedding model (nomic-embed-text)

## How It Works

1. Documents are loaded from the `data/docs` folder
2. Documents are split into chunks
3. Each chunk is converted into embeddings
4. Embeddings are stored in a vector database
5. User questions are converted into embeddings
6. The most relevant document chunk is retrieved
7. The language model generates a final answer using the retrieved context

## Run the Project

```bash
python app.py
