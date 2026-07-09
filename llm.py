from __future__ import annotations
from typing import Dict, Optional
from langchain_openai import ChatOpenAI
from config import settings


def build_llm(model: Optional[str] = None) -> ChatOpenAI:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to your .env file.")

    headers: Dict[str, str] = {}
    if settings.openrouter_site_url:
        headers["HTTP-Referer"] = settings.openrouter_site_url
    if settings.openrouter_app_name:
        headers["X-Title"] = settings.openrouter_app_name

    return ChatOpenAI(
        model=model or settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
        max_tokens=settings.max_output_tokens,
        default_headers=headers or None,
    )