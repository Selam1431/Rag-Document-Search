import logging
import os

from rag_core import extract_text_from_pdf, split_into_chunks

logger = logging.getLogger(__name__)


def load_documents(folder: str) -> list:
    documents = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            logger.info("Loaded text file: %s", file)

        elif file.endswith(".pdf"):
            text = extract_text_from_pdf(path)
            logger.info("Loaded PDF: %s", file)

        else:
            logger.debug("Skipping unsupported file: %s", file)
            continue

        chunks = split_into_chunks(text)

        for i, chunk in enumerate(chunks):
            documents.append({"id": f"{file}_chunk_{i}", "text": chunk})

    logger.info("Loaded %d chunks from %s", len(documents), folder)
    return documents
