"""Per-session metadata: title and lifecycle timestamps.

Lives next to the conversation messages and is owned by the same key namespace.
Each session has at most one meta record; the JSON shape is::

    {
        "title": "气井生产数据分析",
        "created_at": 1726300800.0,
        "updated_at": 1726390000.0
    }

Created the first time a user message is appended to a previously empty
session. Updated whenever a new message lands.
"""
import json
import time
from typing import Optional

from app.core.logging_config import get_logger
from app.memory.redis_store import get_redis_store

logger = get_logger(__name__)


def _key(session_id: str) -> str:
    return f"session:{session_id}:meta"


def _snippet(text: str, limit: int = 20) -> str:
    """Build a human-friendly session title out of the first user message.

    Strips whitespace, replaces internal whitespace with single spaces, then
    takes the first ``limit`` characters. A trailing ellipsis is appended when
    the original was longer.
    """
    compact = " ".join(text.split())
    if not compact:
        return "新对话"
    return compact[:limit] + ("…" if len(compact) > limit else "")


def get_meta(session_id: str) -> Optional[dict]:
    """Return the session meta as a dict, or ``None`` when not set."""
    raw = get_redis_store().get(_key(session_id))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        logger.warning("session_meta_corrupt", session_id=session_id)
        return None


def touch(session_id: str, first_user_message: Optional[str] = None) -> dict:
    """Refresh ``updated_at`` and seed ``title`` the first time.

    Idempotent: calling repeatedly only bumps ``updated_at`` (and seeds the
    title if it was still unset and we have a first message to name it from).
    """
    now = time.time()
    store = get_redis_store()
    current = get_meta(session_id)

    if current is None:
        meta = {
            "title": _snippet(first_user_message) if first_user_message else "新对话",
            "created_at": now,
            "updated_at": now,
        }
    else:
        meta = dict(current)
        meta["updated_at"] = now
        if first_user_message and (meta.get("title") in (None, "", "新对话")):
            meta["title"] = _snippet(first_user_message)

    store.set(_key(session_id), json.dumps(meta, ensure_ascii=False))
    return meta


def clear_meta(session_id: str) -> None:
    get_redis_store().delete(_key(session_id))