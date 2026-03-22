import ollama
from config import EMBED_MODEL, LLM_MODEL, MAX_QUERY_LENGTH


def sanitize_query(query):
    query = query.strip()
    if not query:
        return None, "Query cannot be empty."
    if len(query) > MAX_QUERY_LENGTH:
        return None, f"Query too long. Maximum {MAX_QUERY_LENGTH} characters allowed."
    return query, None


def ask_rag(collection):
    while True:
        raw_query = input("Ask a question (or type 'exit'): ")

        if raw_query.strip().lower() == "exit":
            print("Goodbye!")
            break

        query, error = sanitize_query(raw_query)
        if error:
            print(f"Error: {error}\n")
            continue

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
            print("No relevant context found for your query.\n")
            continue

        retrieved_chunks = docs[0]
        context = "\n\n".join(retrieved_chunks)

        print("\nBest Matching Context:\n")
        print(context[:600])

        prompt = (
            "You are a helpful AI assistant.\n"
            "Answer the user's question using only the context below.\n"
            "Keep the answer short, clear, and direct.\n"
            "If the context does not contain enough information, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}"
        )

        answer = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        print("\nFinal Answer:\n")
        print(answer["message"]["content"])
        print("\n" + "-" * 50 + "\n")
