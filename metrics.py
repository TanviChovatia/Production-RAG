from __future__ import annotations

import json
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(slots=True)
class RequestMetric:
    timestamp: float
    question: str
    latency_seconds: float
    retrieval_latency_seconds: float
    generation_latency_seconds: float
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    retrieved_docs: int
    reranked_docs: int
    citation_count: int
    citation_coverage: float


class MetricsStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, metric: RequestMetric) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(metric), ensure_ascii=False) + "\n")

    def load_all(self) -> List[Dict]:
        if not self.path.exists():
            return []
        rows: List[Dict] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def summary(self) -> Dict[str, float]:
        rows = self.load_all()
        if not rows:
            return {"count": 0, "p50_latency": 0.0, "p95_latency": 0.0, "avg_cost": 0.0}
        latencies = [row["latency_seconds"] for row in rows]
        costs = [row.get("total_cost_usd", 0.0) for row in rows]
        return {
            "count": len(rows),
            "p50_latency": round(percentile(latencies, 50), 4),
            "p95_latency": round(percentile(latencies, 95), 4),
            "avg_cost": round(statistics.fmean(costs), 8) if costs else 0.0,
        }


@contextmanager
def timer() -> Iterable[callable]:
    start = time.perf_counter()
    box = {"elapsed": 0.0}

    def read() -> float:
        return box["elapsed"]

    try:
        yield read
    finally:
        box["elapsed"] = time.perf_counter() - start


def percentile(values: List[float], p: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)
