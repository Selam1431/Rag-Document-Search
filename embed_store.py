import ollama
import chromadb
from config import COLLECTION_NAME, EMBED_MODEL, CHROMA_PATH


def store_embeddings(documents):
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    for doc in documents:
        embedding = ollama.embeddings(
            model=EMBED_MODEL,
            prompt=doc["text"]
        )["embedding"]

        collection.add(
            documents=[doc["text"]],
            embeddings=[embedding],
            ids=[doc["id"]]
        )

    print(f"Stored {len(documents)} chunks in ChromaDB at {CHROMA_PATH}")
    return collection
