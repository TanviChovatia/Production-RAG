# from __future__ import annotations

# import argparse
# from pathlib import Path

# from document_loader import load_document, load_documents_from_dir
# from rag_chain import build_rag_system


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Production RAG CLI")
#     parser.add_argument("--path", type=str, default="data", help="PDF, markdown, text file, or folder")
#     parser.add_argument("--question", type=str, required=True, help="Question to ask")
#     return parser.parse_args()


# def main() -> None:
#     args = parse_args()
#     path = Path(args.path)
#     docs = load_documents_from_dir(path) if path.is_dir() else load_document(path)
#     rag = build_rag_system(docs, collection_name=path.stem.replace(" ", "_") or "rag_docs")
#     response = rag.ask(args.question)
#     print("\nAnswer:\n")
#     print(response.answer)
#     print("\nSources:")
#     for doc in response.source_documents:
#         page = doc.metadata.get("page", "?")
#         if isinstance(page, int):
#             page += 1
#         print(f"- {doc.metadata.get('source')} (page {page})")
#     print("\nMetadata:")
#     print(response.metadata)


# if __name__ == "__main__":
#     main()


from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from document_loader import load_document, load_documents_from_dir
from rag_chain import build_rag_system
from config import settings
from metrics import MetricsStore


app = FastAPI(title=settings.app_name, version="1.0.0")

STATE: Dict[str, Any] = {
    "rag": None,
    "loaded_source": None,
}


class LoadPathRequest(BaseModel):
    path: str


class AskRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/load-path")
def load_path(payload: LoadPathRequest) -> Dict[str, Any]:
    path = Path(payload.path)

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    if path.is_dir():
        docs = load_documents_from_dir(path)
        if not docs:
            raise HTTPException(status_code=400, detail="No PDF files found in the folder.")
    else:
        if path.suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        docs = load_document(path)

    STATE["rag"] = build_rag_system(
        docs,
        collection_name=path.stem.replace(" ", "_") or "rag_docs"
    )
    STATE["loaded_source"] = str(path)

    return {
        "message": "Documents loaded successfully",
        "source": str(path),
        "document_count": len(docs),
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()

    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    docs = load_document(tmp_path)

    STATE["rag"] = build_rag_system(
        docs,
        collection_name=tmp_path.stem.replace(" ", "_") or "uploaded_pdf"
    )
    STATE["loaded_source"] = file.filename

    return {
        "message": "PDF uploaded successfully",
        "source": file.filename,
        "document_count": len(docs),
    }


@app.post("/ask")
def ask_question(payload: AskRequest) -> Dict[str, Any]:
    rag = STATE.get("rag")

    if rag is None:
        raise HTTPException(
            status_code=400,
            detail="No PDF loaded. Use /upload or /load-path first."
        )

    response = rag.ask(
        payload.question,
        user_id=payload.user_id,
        session_id=payload.session_id,
    )

    sources: List[Dict[str, Any]] = []
    for i, doc in enumerate(response.source_documents, start=1):
        page = doc.metadata.get("page", "?")
        if isinstance(page, int):
            page += 1

        sources.append(
            {
                "source_id": f"S{i}",
                "source": doc.metadata.get("source", "unknown"),
                "page": page,
                "chunk_id": doc.metadata.get("chunk_id"),
            }
        )

    return {
        "answer": response.answer,
        "citations": response.citations,
        "sources": sources,
        "metadata": response.metadata,
    }


@app.get("/metrics")
def get_metrics() -> Dict[str, Any]:
    store = MetricsStore(settings.metrics_dir / "requests.jsonl")
    return store.summary()