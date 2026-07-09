from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Production RAG Application")
    docs_dir: Path = Path(os.getenv("DOCS_DIR", "data"))
    vector_db_dir: Path = Path(os.getenv("VECTOR_DB_DIR", "artifacts/chroma"))
    metrics_dir: Path = Path(os.getenv("METRICS_DIR", "artifacts/metrics"))
    traces_dir: Path = Path(os.getenv("TRACES_DIR", "artifacts/traces"))
    evals_dir: Path = Path(os.getenv("EVALS_DIR", "evals"))

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    openrouter_site_url: Optional[str] = os.getenv("OPENROUTER_SITE_URL")
    openrouter_app_name: Optional[str] = os.getenv("OPENROUTER_APP_NAME", "production-rag-app")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "180"))
    vector_top_k: int = int(os.getenv("VECTOR_TOP_K", "12"))
    bm25_top_k: int = int(os.getenv("BM25_TOP_K", "12"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "6"))
    max_context_chunks: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "6"))

    temperature: float = float(os.getenv("TEMPERATURE", "0"))
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "120"))
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "1200"))
    
    langfuse_public_key: Optional[str] = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[str] = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_base_url: Optional[str] = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
    enable_langsmith: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

    prompt_version: str = os.getenv("PROMPT_VERSION", "v1")

    input_cost_per_1m_tokens: float = float(os.getenv("INPUT_COST_PER_1M_TOKENS", "0"))
    output_cost_per_1m_tokens: float = float(os.getenv("OUTPUT_COST_PER_1M_TOKENS", "0"))

    eval_threshold_faithfulness: float = float(os.getenv("EVAL_THRESHOLD_FAITHFULNESS", "0.60"))
    eval_threshold_answer_relevance: float = float(os.getenv("EVAL_THRESHOLD_ANSWER_RELEVANCE", "0.60"))
    eval_threshold_context_precision: float = float(os.getenv("EVAL_THRESHOLD_CONTEXT_PRECISION", "0.60"))
    eval_threshold_citation_coverage: float = float(os.getenv("EVAL_THRESHOLD_CITATION_COVERAGE", "1.0"))

    extra_headers: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.extra_headers = {"HTTP-Referer": self.openrouter_site_url or "", "X-Title": self.openrouter_app_name or ""}


settings = Settings()
