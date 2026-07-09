from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


CHAT_DIR = Path("artifacts/chat_history")
CHAT_DIR.mkdir(parents=True, exist_ok=True)


def _chat_path(chat_id: str) -> Path:
    return CHAT_DIR / f"{chat_id}.json"


def create_chat(document_name: str | None = None) -> str:
    chat_id = str(uuid.uuid4())
    data = {
        "chat_id": chat_id,
        "title": "New Chat",
        "document_name": document_name,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "messages": [],
    }
    save_chat(data)
    return chat_id


def save_chat(chat_data: Dict[str, Any]) -> None:
    chat_data["updated_at"] = datetime.utcnow().isoformat()
    path = _chat_path(chat_data["chat_id"])
    with path.open("w", encoding="utf-8") as f:
        json.dump(chat_data, f, ensure_ascii=False, indent=2)


def load_chat(chat_id: str) -> Dict[str, Any]:
    path = _chat_path(chat_id)
    if not path.exists():
        raise FileNotFoundError(f"Chat not found: {chat_id}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_chats() -> List[Dict[str, Any]]:
    chats: List[Dict[str, Any]] = []
    for path in CHAT_DIR.glob("*.json"):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        messages = data.get("messages", [])
        if not messages:
            continue

        chats.append(
            {
                "chat_id": data["chat_id"],
                "title": data.get("title", "Untitled Chat"),
                "document_name": data.get("document_name"),
                "updated_at": data.get("updated_at"),
                "message_count": len(messages),
            }
        )

    chats.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return chats


def append_message(chat_id: str, role: str, content: str, **extra: Any) -> Dict[str, Any]:
    data = load_chat(chat_id)
    message = {
        "role": role,
        "content": content,
        **extra,
    }
    data["messages"].append(message)

    if role == "user" and data.get("title", "New Chat") == "New Chat":
        data["title"] = content[:60].strip() or "New Chat"

    save_chat(data)
    return data


def delete_chat(chat_id: str) -> None:
    path = _chat_path(chat_id)
    if path.exists():
        path.unlink()