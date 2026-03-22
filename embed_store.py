import logging

import chromadb

from config import COLLECTION_NAME, CHROMA_PATH
from rag_core import embed_chunks_concurrent

logger = logging.getLogger(__name__)


def store_embeddings(documents: list):
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(name=COLLECTION_NAME)
        logger.info("Dropped existing collection: %s", COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    chunks = [doc["text"] for doc in documents]
    ids = [doc["id"] for doc in documents]

    embeddings = embed_chunks_concurrent(chunks)

    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    logger.info("Stored %d chunks in ChromaDB at %s", len(documents), CHROMA_PATH)
    return collection
