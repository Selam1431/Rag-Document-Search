from load_documents import load_documents
from embed_store import store_embeddings
from chat_rag import ask_rag

folder = "data/docs"

documents = load_documents(folder)
collection = store_embeddings(documents)
ask_rag(collection)