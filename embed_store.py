import chromadb
import ollama


def store_embeddings(documents):
    client = chromadb.PersistentClient(path="chroma_db")

    try:
        client.delete_collection(name="documents")
    except:
        pass

    collection = client.create_collection(name="documents")

    for doc in documents:
        text = doc["text"]
        doc_id = doc["id"]

        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=text
        )

        embedding = response["embedding"]

        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id]
        )

    print("Documents embedded and stored!")
    return collection