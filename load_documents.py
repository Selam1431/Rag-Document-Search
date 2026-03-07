import os
from pypdf import PdfReader


def split_into_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    text = " ".join(text.split())  # clean extra spaces/newlines

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def load_documents(folder):
    documents = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

        elif file.endswith(".pdf"):
            reader = PdfReader(path)
            text = ""

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    clean_text = page_text.replace("\n", " ")
                    text += clean_text + " "

        else:
            continue

        chunks = split_into_chunks(text, chunk_size=500, overlap=100)

        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()

            if chunk:
                documents.append({
                    "id": f"{file}_chunk_{i}",
                    "text": chunk
                })

    return documents