from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

from config import settings

try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
except Exception:  # pragma: no cover
    LangfuseCallbackHandler = None


class LocalTraceCallbackHandler(BaseCallbackHandler):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.runs: List[Dict[str, Any]] = []

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        self.runs.append({"event": "chain_start", "serialized": serialized, "inputs": inputs})

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        self.runs.append({"event": "chain_end", "outputs": outputs})

    def on_retriever_end(self, documents: Any, **kwargs: Any) -> Any:
        self.runs.append({"event": "retriever_end", "document_count": len(documents) if documents else 0})

    def flush(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        trace_id = str(uuid.uuid4())
        payload = {"trace_id": trace_id, "metadata": metadata or {}, "events": self.runs}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.runs = []
        return trace_id


def get_callbacks(user_id: str | None = None, session_id: str | None = None) -> List[Any]:
    callbacks: List[Any] = [LocalTraceCallbackHandler(settings.traces_dir / "traces.jsonl")]

    print("LANGFUSE_PUBLIC_KEY loaded:", bool(settings.langfuse_public_key))
    print("LANGFUSE_SECRET_KEY loaded:", bool(settings.langfuse_secret_key))
    print("LANGFUSE_BASE_URL:", settings.langfuse_base_url)
    print("Langfuse import ok:", LangfuseCallbackHandler is not None)

    if settings.langfuse_public_key and settings.langfuse_secret_key and LangfuseCallbackHandler is not None:
        print("Langfuse callback enabled")
        callbacks.append(LangfuseCallbackHandler())

    return callbacks