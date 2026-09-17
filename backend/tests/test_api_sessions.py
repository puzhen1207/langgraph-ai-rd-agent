"""HTTP-level checks that one client cannot reach another client's session."""
import unittest

from fastapi.testclient import TestClient

from app.core import security
from app.memory.redis_store import InMemoryStore
from app.memory.session_registry import SessionRegistry
from main import app


class SessionIsolationApiTests(unittest.TestCase):
    def setUp(self):
        self.registry = SessionRegistry(store=InMemoryStore())
        self._original_getter = security.get_session_registry
        security.get_session_registry = lambda: self.registry
        self.client = TestClient(app)

    def tearDown(self):
        security.get_session_registry = self._original_getter

    def test_history_is_rejected_for_another_token(self):
        self.registry.claim("sess-1", "token-a")

        owner = self.client.get(
            "/api/v1/chat/session/sess-1/history",
            headers={"X-Client-Token": "token-a"},
        )
        self.assertEqual(owner.status_code, 200)

        stranger = self.client.get(
            "/api/v1/chat/session/sess-1/history",
            headers={"X-Client-Token": "token-b"},
        )
        self.assertEqual(stranger.status_code, 403)

        anonymous = self.client.get("/api/v1/chat/session/sess-1/history")
        self.assertEqual(anonymous.status_code, 403)

    def test_clearing_another_clients_session_is_rejected(self):
        self.registry.claim("sess-2", "token-a")

        stranger = self.client.delete(
            "/api/v1/chat/session/sess-2",
            headers={"X-Client-Token": "token-b"},
        )
        self.assertEqual(stranger.status_code, 403)

        owner = self.client.delete(
            "/api/v1/chat/session/sess-2",
            headers={"X-Client-Token": "token-a"},
        )
        self.assertEqual(owner.status_code, 200)

    def test_stranger_cannot_start_a_conversation_in_a_taken_session(self):
        self.registry.claim("sess-3", "token-a")

        response = self.client.post(
            "/api/v1/chat",
            json={"query": "hi", "session_id": "sess-3"},
            headers={"X-Client-Token": "token-b"},
        )
        # Rejected before the graph is ever invoked.
        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
