import ollama


def ask_rag(collection):
    while True:
        query = input("Ask a question (or type 'exit'): ").strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

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

        print("\nBest Matching Context:\n")
        print(context[:600])  # only show part so terminal is cleaner

        prompt = f"""
You are a helpful AI assistant.
Answer the user's question using only the context below.
Keep the answer short, clear, and direct.

Context:
{context}

Question:
{query}
"""

        answer = ollama.chat(
            model="gemma3:4b",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        print("\nFinal Answer:\n")
        print(answer["message"]["content"])
        print("\n" + "-" * 50 + "\n")