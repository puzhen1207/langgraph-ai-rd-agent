"""Tests for the multi-conversation feature: session meta, list endpoint, ownership."""
import time
import unittest

from fastapi.testclient import TestClient

from app.core import security
from app.memory import conversation, session_meta
from app.memory.redis_store import InMemoryStore
from app.memory.session_registry import SessionRegistry
from main import app


def _isolated_app() -> tuple[TestClient, SessionRegistry]:
    """Build a TestClient with a fresh in-memory store so tests cannot leak."""
    store = InMemoryStore()
    # Override module-level singletons for the duration of one test.
    conversation.get_redis_store = lambda: store
    session_meta.get_redis_store = lambda: store
    registry = SessionRegistry(store=store)
    security.get_session_registry = lambda: registry
    security.get_redis_store = lambda: store
    # ``chat.py`` did ``from app.memory.session_registry import get_session_registry``,
    # so it sees the binding in its own namespace, not the source module's.
    from app.api import chat as chat_module
    chat_module.get_session_registry = lambda: registry
    return TestClient(app), registry


class SessionRegistryIndexTests(unittest.TestCase):
    def test_claim_adds_to_token_index(self):
        store = InMemoryStore()
        registry = SessionRegistry(store=store)
        self.assertTrue(registry.claim("s1", "tok"))
        self.assertEqual(registry.list_for_token("tok"), ["s1"])
        # A second session for the same token should appear too.
        self.assertTrue(registry.claim("s2", "tok"))
        self.assertEqual(set(registry.list_for_token("tok")), {"s1", "s2"})

    def test_claim_does_not_index_for_existing_owner(self):
        store = InMemoryStore()
        registry = SessionRegistry(store=store)
        registry.claim("s1", "tok-a")
        # Re-claiming the same session with the same token must not double-add.
        registry.claim("s1", "tok-a")
        self.assertEqual(registry.list_for_token("tok-a"), ["s1"])

    def test_unbind_removes_from_token_index(self):
        store = InMemoryStore()
        registry = SessionRegistry(store=store)
        registry.claim("s1", "tok")
        registry.unbind("s1")
        self.assertEqual(registry.list_for_token("tok"), [])


class SessionMetaTests(unittest.TestCase):
    def test_first_user_message_seeds_title(self):
        store = InMemoryStore()
        session_meta.get_redis_store = lambda: store
        meta = session_meta.touch("s1", first_user_message="什么是气井压裂？")
        self.assertEqual(meta["title"], "什么是气井压裂？")
        self.assertAlmostEqual(meta["created_at"], meta["updated_at"], places=2)

    def test_long_first_message_is_truncated_with_ellipsis(self):
        store = InMemoryStore()
        session_meta.get_redis_store = lambda: store
        long = "一" * 60
        meta = session_meta.touch("s2", first_user_message=long)
        self.assertEqual(len(meta["title"]), 21)  # 20 chars + ellipsis
        self.assertTrue(meta["title"].endswith("…"))

    def test_subsequent_touch_updates_updated_at_only(self):
        store = InMemoryStore()
        session_meta.get_redis_store = lambda: store
        session_meta.touch("s3", first_user_message="你好")
        before = session_meta.get_meta("s3")
        time.sleep(0.01)
        session_meta.touch("s3")
        after = session_meta.get_meta("s3")
        self.assertEqual(before["title"], after["title"], "title must not change on later touches")
        self.assertEqual(before["created_at"], after["created_at"], "created_at must stay stable")
        self.assertGreater(after["updated_at"], before["updated_at"])

    def test_empty_message_yields_default_title(self):
        store = InMemoryStore()
        session_meta.get_redis_store = lambda: store
        meta = session_meta.touch("s4")
        self.assertEqual(meta["title"], "新对话")


class ListSessionsApiTests(unittest.TestCase):
    def setUp(self):
        self.client, self.registry = _isolated_app()

    def test_list_returns_only_own_sessions(self):
        self.registry.claim("s-a", "tok")
        self.client.post(
            "/api/v1/chat",
            json={"query": "第一个会话", "session_id": "s-a"},
            headers={"X-Client-Token": "tok"},
        )

        resp = self.client.get(
            "/api/v1/chat/sessions",
            headers={"X-Client-Token": "tok"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(len(body), 1)
        self.assertEqual(body[0]["session_id"], "s-a")
        self.assertEqual(body[0]["title"], "第一个会话")
        self.assertEqual(body[0]["message_count"], 2)  # user + assistant

    def test_list_orders_newest_first(self):
        self.registry.claim("old", "tok")
        self.client.post(
            "/api/v1/chat",
            json={"query": "老会话", "session_id": "old"},
            headers={"X-Client-Token": "tok"},
        )

        time.sleep(0.05)
        self.registry.claim("new", "tok")
        self.client.post(
            "/api/v1/chat",
            json={"query": "新会话", "session_id": "new"},
            headers={"X-Client-Token": "tok"},
        )

        body = self.client.get(
            "/api/v1/chat/sessions",
            headers={"X-Client-Token": "tok"},
        ).json()
        self.assertEqual([s["session_id"] for s in body], ["new", "old"])

    def test_list_isolated_between_tokens(self):
        self.registry.claim("s-a", "tok-a")
        self.client.post(
            "/api/v1/chat",
            json={"query": "甲的会话", "session_id": "s-a"},
            headers={"X-Client-Token": "tok-a"},
        )
        self.registry.claim("s-b", "tok-b")
        self.client.post(
            "/api/v1/chat",
            json={"query": "乙的会话", "session_id": "s-b"},
            headers={"X-Client-Token": "tok-b"},
        )

        a_list = self.client.get(
            "/api/v1/chat/sessions",
            headers={"X-Client-Token": "tok-a"},
        ).json()
        b_list = self.client.get(
            "/api/v1/chat/sessions",
            headers={"X-Client-Token": "tok-b"},
        ).json()
        self.assertEqual([s["session_id"] for s in a_list], ["s-a"])
        self.assertEqual([s["session_id"] for s in b_list], ["s-b"])

    def test_list_skips_husk_sessions(self):
        """Claimed-but-empty sessions (expired data / never used) are hidden."""
        self.registry.claim("husk", "tok")  # claimed, never messaged

        body = self.client.get(
            "/api/v1/chat/sessions",
            headers={"X-Client-Token": "tok"},
        ).json()
        self.assertEqual(body, [])

    def test_list_empty_when_no_sessions(self):
        resp = self.client.get(
            "/api/v1/chat/sessions",
            headers={"X-Client-Token": "tok-empty"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [])


if __name__ == "__main__":
    unittest.main()