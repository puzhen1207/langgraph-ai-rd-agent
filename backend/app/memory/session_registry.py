"""Anonymous session ownership registry.

Conversation history is keyed by a client-supplied ``session_id``. Without an
ownership record, any caller that learns or guesses a session id can read or
delete someone else's conversation. Every session is therefore bound to the
anonymous client token that first used it.

The ownership key shares the store's TTL with the conversation itself and is
refreshed on every authorised write, so an active conversation never loses its
owner while its messages are still alive.

A side index, ``sessions_by_token:{token}``, lists every session a token owns.
It is kept in sync with the ownership key on claim and is consulted by the
``GET /chat/sessions`` endpoint.
"""
from functools import lru_cache
from typing import List, Optional, Union

from app.core.config import settings
from app.core.logging_config import get_logger
from app.memory.redis_store import InMemoryStore, RedisStore, get_redis_store

logger = get_logger(__name__)

Store = Union[RedisStore, InMemoryStore]


class SessionRegistry:
    """Maps ``session_id`` to the anonymous token that owns it."""

    def __init__(self, store: Optional[Store] = None):
        self.store: Store = store if store is not None else get_redis_store()

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}:owner"

    @staticmethod
    def _index_key(token: str) -> str:
        return f"sessions_by_token:{token}"

    def owner(self, session_id: str) -> Optional[str]:
        return self.store.get(self._key(session_id))

    def claim(self, session_id: str, token: Optional[str]) -> bool:
        """Authorise a chat request and bind a brand-new session.

        Returns ``False`` when the session already belongs to another token.
        A session used without any token (curl, Swagger) stays unowned, which
        grants no access to any session that is already owned.
        """
        owner = self.owner(session_id)
        if not token:
            return owner is None
        if owner is not None and owner != token:
            return False
        self.store.set(self._key(session_id), token)
        if owner is None:
            # Token index: add when transitioning from unowned → owned so we
            # can list the token's sessions without scanning the keyspace.
            self.store.sadd(self._index_key(token), session_id)
        else:
            # Re-claim on an ongoing conversation: refresh the index TTL so
            # the listing keeps pace with the session it points at (the owner
            # key above is renewed by the ``set`` call on every chat).
            self.store.expire(self._index_key(token), settings.REDIS_TTL)
        return True

    def is_owner(self, session_id: str, token: Optional[str]) -> bool:
        """Whether the caller may read or delete this session.

        Sessions with no recorded owner hold no messages to leak, so they stay
        readable to keep upgrade paths and first-load requests working.
        """
        owner = self.owner(session_id)
        return owner is None or owner == token

    def list_for_token(self, token: str) -> List[str]:
        """Every session this token has claimed.

        Order is left to the caller; ``SessionMeta`` carries the
        ``updated_at`` needed to sort by recency.
        """
        if not token:
            return []
        return self.store.smembers(self._index_key(token))

    def unbind(self, session_id: str) -> None:
        """Remove the ownership record and the token index entry.

        Used when a session is fully cleared. Safe to call even when the
        session had no recorded owner.
        """
        owner = self.owner(session_id)
        if owner:
            self.store.srem(self._index_key(owner), session_id)
        self.store.delete(self._key(session_id))


@lru_cache(maxsize=1)
def get_session_registry() -> SessionRegistry:
    return SessionRegistry()
