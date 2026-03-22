import logging

from config import MAX_QUERY_LENGTH
from rag_core import embed_text, build_prompt, generate_answer

logger = logging.getLogger(__name__)


def sanitize_query(query: str):
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

        logger.info("Query received: %s", query)

        query_embedding = embed_text(query)

        results = collection.query(query_embeddings=[query_embedding], n_results=3)

        docs = results.get("documents", [[]])
        if not docs or not docs[0]:
            print("No relevant context found for your query.\n")
            logger.warning("No results for query: %s", query)
            continue

        context = "\n\n".join(docs[0])
        print("\nBest Matching Context:\n")
        print(context[:600])

        prompt = build_prompt(context, query)
        answer = generate_answer(prompt)

        print("\nFinal Answer:\n")
        print(answer)
        print("\n" + "-" * 50 + "\n")
