import os

def load_documents(folder):
    documents = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = text.split("\n\n")

        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()

            if chunk:
                documents.append({
                    "id": f"{file}_chunk_{i}",
                    "text": chunk
                })

    return documents