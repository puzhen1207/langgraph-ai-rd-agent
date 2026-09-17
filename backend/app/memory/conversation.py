"""
Conversation Memory - Window Memory + Summary Memory.
Backed by Redis with in-memory dict fallback.
"""
import json
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging_config import get_logger
from app.memory.redis_store import get_redis_store
from app.memory import session_meta

logger = get_logger(__name__)

WINDOW_SIZE = 10


class ConversationMemory:
    """
    Per-session conversation memory.
    - Window: keeps last N message turns
    - Summary: LLM-compressed summary of older turns
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.store = get_redis_store()
        self._messages_key = f"session:{session_id}:messages"
        self._summary_key = f"session:{session_id}:summary"

    def add_user_message(self, content: str):
        # Seed / refresh session metadata at the user-turn boundary. The first
        # user message names the session; subsequent ones only bump updated_at.
        session_meta.touch(self.session_id, first_user_message=content)
        self._append_message({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        session_meta.touch(self.session_id)
        self._append_message({"role": "assistant", "content": content})

    def _append_message(self, message: Dict[str, str]):
        messages = self._load_messages()
        messages.append(message)
        # Keep only recent window
        if len(messages) > WINDOW_SIZE:
            self._maybe_summarize(messages[:-WINDOW_SIZE])
            messages = messages[-WINDOW_SIZE:]
        self._save_messages(messages)

    def get_messages(self) -> List[Dict[str, str]]:
        return self._load_messages()

    def build_context(self, recent_turns: int = 6) -> str:
        """Build text context from summary + recent window."""
        summary = self._load_summary()
        messages = self._load_messages()
        recent = messages[-recent_turns:] if len(messages) > recent_turns else messages

        parts = []
        if summary:
            parts.append(f"【历史摘要】\n{summary}")
        if recent:
            lines = []
            for msg in recent:
                role = "用户" if msg["role"] == "user" else "助手"
                lines.append(f"{role}: {msg['content'][:400]}")
            parts.append("【最近对话】\n" + "\n".join(lines))
        return "\n\n".join(parts) if parts else ""

    def _maybe_summarize(self, old_messages: List[Dict[str, str]]):
        """Summarize older messages that fall out of the window."""
        if not old_messages:
            return
        try:
            history = "\n".join(
                f"{m['role']}: {m['content'][:300]}" for m in old_messages
            )
            summary = self._summarize_with_llm(history)
            existing = self._load_summary()
            combined = f"{existing}\n{summary}" if existing else summary
            self._save_summary(combined[-2000:])  # Trim summary length
        except Exception as e:
            logger.warning("summarize_failed", error=str(e))

    def _summarize_with_llm(self, history: str) -> str:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        from app.prompt.templates import SUMMARY_PROMPT

        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_API_BASE,
            temperature=0.0,
            max_tokens=300,
        )
        prompt = SUMMARY_PROMPT.format(history=history)
        result = llm.invoke([HumanMessage(content=prompt)])
        return result.content.strip()

    def _load_messages(self) -> List[Dict[str, str]]:
        data = self.store.get(self._messages_key)
        if data:
            try:
                return json.loads(data)
            except Exception:
                pass
        return []

    def _save_messages(self, messages: List[Dict[str, str]]):
        self.store.set(self._messages_key, json.dumps(messages, ensure_ascii=False))

    def _load_summary(self) -> str:
        return self.store.get(self._summary_key) or ""

    def _save_summary(self, summary: str):
        self.store.set(self._summary_key, summary)

    def clear(self):
        self.store.delete(self._messages_key)
        self.store.delete(self._summary_key)
        session_meta.clear_meta(self.session_id)
        logger.info("session_cleared", session_id=self.session_id)
