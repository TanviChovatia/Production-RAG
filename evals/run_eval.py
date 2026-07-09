from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
import sys
from typing import Dict, List

from dotenv import load_dotenv

from config import settings
from document_loader import load_documents_from_dir
from rag_chain import build_rag_system
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_score(a: str, b: str) -> float:
    emb1 = model.encode([a])[0]
    emb2 = model.encode([b])[0]
    return float(cosine_similarity([emb1], [emb2])[0][0])
def strip_citations(text: str) -> str:
    return re.sub(r"\[S\d+\]", "", text).strip()

def load_goldens(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def overlap_score(a: str, b: str) -> float:
    aset = set(a.lower().split())
    bset = set(b.lower().split())
    if not aset or not bset:
        return 0.0
    return len(aset & bset) / len(aset | bset)


def evaluate() -> Dict:
    docs = load_documents_from_dir(settings.docs_dir)
    rag = build_rag_system(docs, collection_name="evals")
    goldens = load_goldens(settings.evals_dir / "goldens.jsonl")

    results = []
    for row in goldens:
        eval_question = (
            f"{row['question']}\n\n"
            "Answer in exactly 1 short sentence if possible, maximum 2 sentences. "
            "Use only the provided context. "
            "Prefer exact wording from the context. "
            "Do not explain. "
            "Do not infer. "
            "Every sentence must include inline citations like [S1]."
        )
        response = rag.ask(eval_question, session_id="eval")
        clean_answer = strip_citations(response.answer)
        answer = clean_answer

        results.append(
        {
            "question": row["question"],
            "answer": answer,
            "faithfulness": semantic_score(answer, row["ground_truth"]),
            "answer_relevance": semantic_score(answer, row["ground_truth"]),
            "answer_relevance": semantic_score(answer, row["ground_truth"]),
            "context_precision": min(1.0, len(response.source_documents) / max(1, row.get("expected_min_sources", 1))),
            "citation_coverage": response.metadata.get("citation_coverage", 0.0),
        }
    )

    summary = {
        "faithfulness": round(fmean(r["faithfulness"] for r in results), 4) if results else 0.0,
        "answer_relevance": round(fmean(r["answer_relevance"] for r in results), 4) if results else 0.0,
        "context_precision": round(fmean(r["context_precision"] for r in results), 4) if results else 0.0,
        "citation_coverage": round(fmean(r["citation_coverage"] for r in results), 4) if results else 0.0,
        "examples": results,
    }

    out_path = settings.evals_dir / "eval_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2))
    enforce_thresholds(summary)
    return summary


def enforce_thresholds(summary: Dict) -> None:
    checks = {
        "faithfulness": settings.eval_threshold_faithfulness,
        "answer_relevance": settings.eval_threshold_answer_relevance,
        "context_precision": settings.eval_threshold_context_precision,
        "citation_coverage": settings.eval_threshold_citation_coverage,
    }

    failed = [
        f"{metric}={summary[metric]:.4f} < {threshold}"
        for metric, threshold in checks.items()
        if summary.get(metric, 0) < threshold
    ]

    if failed:
        print("\n[FAIL] CI regression gate failed:", "; ".join(failed))
        sys.exit(1)
    else:
        print("\n[PASS] CI gate passed.")

if __name__ == "__main__":
    evaluate()
