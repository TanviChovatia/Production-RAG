from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_document(path: str | Path) -> List[Document]:
    path = Path(path)

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    loader = PyPDFLoader(str(path))
    docs = loader.load()

    for idx, doc in enumerate(docs):
        doc.metadata.setdefault("source", str(path))
        doc.metadata.setdefault("page", idx)
        doc.metadata.setdefault("doc_id", f"{path.stem}-page-{idx + 1}")

    return docs


def load_documents_from_dir(folder: str | Path) -> List[Document]:
    folder = Path(folder)
    docs: List[Document] = []

    for path in sorted(folder.rglob("*.pdf")):
        if path.is_file():
            docs.extend(load_document(path))

    return docs


def normalize_source_name(source: str) -> str:
    return Path(source).name