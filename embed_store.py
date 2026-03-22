import logging
from datetime import datetime, timezone

import chromadb

from config import COLLECTION_NAME, CHROMA_PATH
from rag_core import embed_chunks_concurrent, make_metadata

logger = logging.getLogger(__name__)


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        return client.create_collection(name=COLLECTION_NAME)


def store_embeddings(documents: list, source: str = "cli"):
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        existing = collection.get(where={"source": source})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            logger.info("Replaced %d existing chunks for source: %s", len(existing["ids"]), source)
    except chromadb.errors.NotFoundError:
        collection = client.create_collection(name=COLLECTION_NAME)

    uploaded_at = datetime.now(timezone.utc).isoformat()
    chunks = [doc["text"] for doc in documents]
    ids = [doc["id"] for doc in documents]
    metadatas = [make_metadata(source, uploaded_at, i) for i in range(len(documents))]

    embeddings = embed_chunks_concurrent(chunks)
    collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
    logger.info("Stored %d chunks for source '%s'", len(documents), source)
    return collection
