
from pathlib import Path


def load_documents(directory="data/knowledge"):
    documents = []

    for file_path in Path(directory).glob("*.md"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "source": str(file_path),
            "content": text
        })

    return documents

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks
