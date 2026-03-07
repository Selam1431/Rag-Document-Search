import ollama

def ask_rag(collection):
    while True:
        query = input("Ask a question (or type 'exit'): ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        query_embedding = ollama.embeddings(
            model="nomic-embed-text",
            prompt=query
        )["embedding"]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )

        best_chunk = results["documents"][0][0]

        print("\nBest Matching Chunk:")
        print(best_chunk)

        prompt = f"""
You are a helpful AI assistant.
Answer the user's question using only the context below.

Context:
{best_chunk}

Question:
{query}
"""

        answer = ollama.chat(
            model="gemma3:4b",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        print("\nFinal Answer:")
        print(answer["message"]["content"])
        print("\n" + "-" * 50 + "\n")