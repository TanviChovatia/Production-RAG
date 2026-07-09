from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from config import settings


@dataclass(slots=True)
class IndexedCorpus:
    vector_store: Chroma
    bm25: BM25Okapi
    chunks: List[Document]
    corpus_tokens: List[List[str]]


class HybridIndexer:
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n### ", "\n\n## ", "\n\n", "\n", ". ", " "]
        )

    def chunk_documents(self, documents: Sequence[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(list(documents))
        for i, chunk in enumerate(chunks):
            chunk.metadata.setdefault("chunk_id", i)
            chunk.metadata.setdefault("source", chunk.metadata.get("source", "unknown"))
            chunk.metadata.setdefault("page", chunk.metadata.get("page", 0))
            chunk.metadata["content_hash"] = hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()
        return chunks

    def build(self, documents: Sequence[Document], collection_name: str = "rag_docs") -> IndexedCorpus:
        chunks = self.chunk_documents(documents)
        
        # Ensure we delete the collection if it already exists to prevent duplicate accumulation
        try:
            old_db = Chroma(
                persist_directory=str(settings.vector_db_dir),
                embedding_function=self.embeddings,
                collection_name=collection_name,
            )
            old_db.delete_collection()
        except Exception:
            pass

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(settings.vector_db_dir),
            collection_name=collection_name,
        )
        tokenized = [tokenize(doc.page_content) for doc in chunks]
        bm25 = BM25Okapi(tokenized)
        return IndexedCorpus(vector_store=vector_store, bm25=bm25, chunks=chunks, corpus_tokens=tokenized)


def tokenize(text: str) -> List[str]:
    return [token.strip().lower() for token in text.split() if token.strip()]


def bm25_search(index: IndexedCorpus, query: str, k: int) -> List[Document]:
    tokenized_query = tokenize(query)
    scores = index.bm25.get_scores(tokenized_query)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    return [index.chunks[i] for i, _ in ranked]


def dense_search(index: IndexedCorpus, query: str, k: int) -> List[Document]:
    return index.vector_store.similarity_search(query, k=k)


def dedupe_documents(documents: Sequence[Document]) -> List[Document]:
    seen: Dict[str, Document] = {}
    for doc in documents:
        key = doc.metadata.get("content_hash") or hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()
        if key not in seen:
            seen[key] = doc
    return list(seen.values())
