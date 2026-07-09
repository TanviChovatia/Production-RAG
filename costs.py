from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(slots=True)
class CostEstimate:
    input_cost_usd: float
    output_cost_usd: float

    @property
    def total_cost_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd


def estimate_cost(
    usage: Optional[TokenUsage],
    input_cost_per_1m_tokens: float,
    output_cost_per_1m_tokens: float,
) -> CostEstimate:
    usage = usage or TokenUsage()
    input_cost = (usage.input_tokens / 1_000_000) * input_cost_per_1m_tokens
    output_cost = (usage.output_tokens / 1_000_000) * output_cost_per_1m_tokens
    return CostEstimate(input_cost_usd=round(input_cost, 8), output_cost_usd=round(output_cost, 8))
