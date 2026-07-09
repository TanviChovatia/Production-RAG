from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import settings
from costs import TokenUsage, estimate_cost
from llm import build_llm
from metrics import MetricsStore, RequestMetric
from observability import LocalTraceCallbackHandler, get_callbacks
from vector_store import HybridIndexer, IndexedCorpus, bm25_search, dedupe_documents, dense_search


SYSTEM_PROMPT = """You are a production-grade RAG assistant.
Use ONLY the provided context.
Answer with the shortest possible factual response.
Answer in exactly 1 sentence whenever possible, maximum 2 sentences.
Prefer exact wording from the context.
Do not explain, elaborate, justify, summarize, or add extra details.
Do not infer anything that is not explicitly stated in the context.
Every sentence MUST include at least one inline citation like [S1].
If the answer is not explicitly stated in the context, say: I could not find that in the provided documents. [S1]
Return only the answer text.
"""


@dataclass(slots=True)
class RAGResponse:
    answer: str
    source_documents: List[Document]
    citations: List[str]
    metadata: Dict[str, Any]


class ProductionRAG:
    def __init__(self, corpus: Sequence[Document], collection_name: str = "rag_docs") -> None:
        self.indexer = HybridIndexer()
        self.index = self.indexer.build(corpus, collection_name=collection_name)
        self.reranker = CrossEncoder(settings.reranker_model)
        self.llm = build_llm()
        self.metrics_store = MetricsStore(settings.metrics_dir / "requests.jsonl")

    def retrieve(self, question: str) -> List[Document]:
        dense_docs = dense_search(self.index, question, settings.vector_top_k)
        sparse_docs = bm25_search(self.index, question, settings.bm25_top_k)
        merged = dedupe_documents([*dense_docs, *sparse_docs])
        return self.rerank(question, merged, settings.rerank_top_k)

    def rerank(self, question: str, docs: Sequence[Document], top_k: int) -> List[Document]:
        if not docs:
            return []
        pairs = [(question, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    def build_context(self, docs: Sequence[Document]) -> str:
        blocks: List[str] = []
        for i, doc in enumerate(docs[: settings.max_context_chunks], start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            if isinstance(page, int):
                page = page + 1
            blocks.append(f"[S{i}] Source={source} Page={page}\n{doc.page_content.strip()}")
        return "\n\n".join(blocks)

    def ask(self, question: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> RAGResponse:
        started = time.perf_counter()
        retrieval_start = time.perf_counter()
        
        # Clean query for retrieval to prevent instruction pollution from app.py
        search_query = question.split("\n\nInstruction:")[0].strip()
        docs = self.retrieve(search_query)
        retrieval_elapsed = time.perf_counter() - retrieval_start

        if not docs:
            return RAGResponse(
                answer="I could not find that in the provided documents.",
                source_documents=[],
                citations=[],
                metadata={"citation_coverage": 0.0},
            )

        context = self.build_context(docs)
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", f"Context:\n{context}\n\nQuestion: {question}\nAnswer with inline citations."),
        ]

        callbacks = get_callbacks(user_id=user_id, session_id=session_id)
        generation_start = time.perf_counter()
        result = self.llm.invoke(messages, config={"callbacks": callbacks, "tags": ["rag", settings.prompt_version]})
        generation_elapsed = time.perf_counter() - generation_start
        answer = getattr(result, "content", str(result)).strip()
        answer = enforce_citations(answer, docs)
        citations = extract_citations(answer)
        coverage = citation_coverage(answer)

        usage_meta = getattr(result, "usage_metadata", {}) or {}
        usage = TokenUsage(
            input_tokens=int(usage_meta.get("input_tokens", 0)),
            output_tokens=int(usage_meta.get("output_tokens", 0)),
        )
        cost = estimate_cost(usage, settings.input_cost_per_1m_tokens, settings.output_cost_per_1m_tokens)

        total_elapsed = time.perf_counter() - started
        self.metrics_store.append(
            RequestMetric(
                timestamp=time.time(),
                question=question,
                latency_seconds=total_elapsed,
                retrieval_latency_seconds=retrieval_elapsed,
                generation_latency_seconds=generation_elapsed,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_cost_usd=cost.total_cost_usd,
                retrieved_docs=len(docs),
                reranked_docs=min(len(docs), settings.rerank_top_k),
                citation_count=len(citations),
                citation_coverage=coverage,
            )
        )

        local_handlers = [cb for cb in callbacks if isinstance(cb, LocalTraceCallbackHandler)]
        trace_id = None
        for handler in local_handlers:
            trace_id = handler.flush(metadata={"question": question, "citation_count": len(citations)})

        return RAGResponse(
            answer=answer,
            source_documents=list(docs),
            citations=citations,
            metadata={
                "citation_coverage": coverage,
                "trace_id": trace_id,
                "usage": usage_meta,
                "cost_usd": cost.total_cost_usd,
                "latency_seconds": round(total_elapsed, 4),
                "retrieval_latency_seconds": round(retrieval_elapsed, 4),
                "generation_latency_seconds": round(generation_elapsed, 4),
            },
        )


SENTENCE_PATTERN = re.compile(r'[^.!?]+?[.!?](?:\s*\[S\d+\])*')


def split_sentences(text: str) -> List[str]:
    return [m.group(0).strip() for m in SENTENCE_PATTERN.finditer(text)]


def enforce_citations(answer: str, docs: Sequence[Document]) -> str:
    if not answer:
        return "I could not find that in the provided documents. [S1]"

    sentences = split_sentences(answer.strip())
    if not sentences:
        return "I could not find that in the provided documents. [S1]"

    fixed = []
    for sentence in sentences:
        if "[S" not in sentence:
            sentence = sentence.rstrip(" .") + " [S1]."
        fixed.append(sentence)

    return " ".join(fixed)


CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


def extract_citations(text: str) -> List[str]:
    return [f"S{m}" for m in CITATION_PATTERN.findall(text)]


def citation_coverage(text: str) -> float:
    sentences = split_sentences(text)
    if not sentences:
        return 0.0
    cited = sum(1 for sentence in sentences if CITATION_PATTERN.search(sentence))
    return round(cited / len(sentences), 4)


def build_rag_system(documents: Sequence[Document], collection_name: str = "rag_docs") -> ProductionRAG:
    return ProductionRAG(documents, collection_name=collection_name)
