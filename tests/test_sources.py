from langchain_core.documents import Document

from vector_store import dedupe_documents, tokenize


def test_tokenize_lowercases() -> None:
    assert tokenize("Hello WORLD") == ["hello", "world"]


def test_dedupe_documents() -> None:
    docs = [
        Document(page_content="same", metadata={}),
        Document(page_content="same", metadata={}),
        Document(page_content="different", metadata={}),
    ]
    deduped = dedupe_documents(docs)
    assert len(deduped) == 2
