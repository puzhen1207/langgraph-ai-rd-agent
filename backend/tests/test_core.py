import asyncio
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from app.agent.nodes import rag_retrieve as rag_retrieve_module
from app.agent.nodes.intent_router import DEFAULT_INTENT, INTENT_CONFIG, get_intent_config
from app.agent.nodes.query_analyze import detect_intent
from app.agent.nodes.response_check import is_valid_response
from app.agent.state import AgentState
from app.api import chat as chat_module
from app.api import health as health_module
from app.core import security
from app.core.config import settings
from app.core.security import require_admin_key
from app.memory.conversation import ConversationMemory
from app.memory.redis_store import InMemoryStore, RedisStore
from app.memory.session_registry import SessionRegistry
from app.rag import ingest as ingest_module
from app.rag import kb_registry as kb_registry_module
from app.rag import reranker as reranker_module
from app.rag import retriever as retriever_module
from app.rag import vectorstore as vectorstore_module
from app.rag.chunker import SmartChunker
from app.rag.kb_registry import (
    DEFAULT_KB_ID,
    DuplicateKnowledgeBaseName,
    InvalidKnowledgeBaseName,
)
from app.rag.retriever import (
    _matches_filter,
    _tokenize,
    build_where_filter,
    reciprocal_rank_fusion,
)


class IntentTests(unittest.TestCase):
    def test_concept_question_uses_the_default_intent(self):
        self.assertEqual(detect_intent("什么是注意力机制"), "concept_qa")

    def test_unmatched_question_falls_back_to_the_default(self):
        # Nothing in the taxonomy claims this, so it must not be forced into a
        # narrow format.
        self.assertEqual(detect_intent("CRISPR 的脱靶效应怎么评估"), "concept_qa")

    def test_routes_literature_review(self):
        self.assertEqual(
            detect_intent("Transformer 的研究现状和主要进展"), "literature_review"
        )

    def test_routes_method_comparison(self):
        self.assertEqual(
            detect_intent("对比一下 Transformer 和 RNN 的区别"), "method_compare"
        )

    def test_routes_academic_writing(self):
        self.assertEqual(detect_intent("帮我润色这段摘要"), "academic_writing")


class ChunkerTests(unittest.TestCase):
    def test_long_text_always_moves_forward(self):
        chunker = SmartChunker(chunk_size=40, chunk_overlap=10)
        chunks = chunker._chunk_text("连续文本。" * 50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_markdown_heading_boundaries(self):
        chunks = SmartChunker(chunk_size=100)._chunk_markdown("# A\n内容\n## B\n内容")
        self.assertEqual(len(chunks), 2)


class RetrievalTests(unittest.TestCase):
    def test_chinese_tokenizer_emits_bigrams(self):
        tokens = _tokenize("播放器崩溃 ExoPlayer")
        self.assertIn("播放", tokens)
        self.assertIn("崩溃", tokens)
        self.assertIn("exoplayer", tokens)

    def test_rrf_merges_same_document_id(self):
        vector = [{
            "doc_id": "same",
            "content": "vector copy",
            "source": "guide.md",
            "score": 0.8,
            "metadata": {},
        }]
        lexical = [{
            "doc_id": "same",
            "content": "lexical copy",
            "source": "guide.md",
            "score": 2.0,
            "metadata": {},
        }]
        merged = reciprocal_rank_fusion(vector, lexical)
        self.assertEqual(len(merged), 1)
        self.assertGreater(merged[0]["score"], 0.03)


class ValidationTests(unittest.TestCase):
    def test_rejects_short_answer(self):
        valid, reason = is_valid_response("太短", "concept_qa")
        self.assertFalse(valid)
        self.assertIn("short", reason.lower())

    def test_comparison_answer_must_compare_something(self):
        answer = (
            "这两种方法都可以使用，具体要看实际的业务场景来定，并不存在一个绝对更好的选择，"
            "建议你结合自己的约束条件来权衡。"
        )
        valid, reason = is_valid_response(answer, "method_compare")
        self.assertFalse(valid)
        self.assertIn("compar", reason.lower())

    def test_a_real_comparison_passes(self):
        answer = (
            "| 维度 | 方法 A | 方法 B |\n|---|---|---|\n"
            "| 核心思路 | 基于统计 | 基于神经网络 |\n"
            "| 主要局限 | 泛化能力弱 | 需要大量标注数据 |\n"
            "若数据量很小，方法 A 更合适。"
        )
        valid, reason = is_valid_response(answer, "method_compare")
        self.assertTrue(valid, msg=reason)

    def test_other_intents_have_no_structural_requirement(self):
        answer = "这是个足够长的说明性回答，用于确认只有对比意图才有结构要求。" * 3
        valid, reason = is_valid_response(answer, "concept_qa")
        self.assertTrue(valid, msg=reason)


class MemoryTests(unittest.TestCase):
    def test_stores_both_sides_of_conversation(self):
        memory = ConversationMemory("test-session")
        memory.store = InMemoryStore()
        memory.add_user_message("问题")
        memory.add_assistant_message("回答")
        self.assertEqual([m["role"] for m in memory.get_messages()], ["user", "assistant"])


class SecurityTests(unittest.TestCase):
    def test_admin_key_is_enforced_when_configured(self):
        original = settings.ADMIN_API_KEY
        settings.ADMIN_API_KEY = "test-secret"
        try:
            with self.assertRaises(HTTPException):
                asyncio.run(require_admin_key("wrong"))
            asyncio.run(require_admin_key("test-secret"))
        finally:
            settings.ADMIN_API_KEY = original


class SessionRegistryTests(unittest.TestCase):
    """A session must be reachable only by the token that created it."""

    def setUp(self):
        self.registry = SessionRegistry(store=InMemoryStore())

    def test_first_claim_binds_the_session_to_the_caller(self):
        self.assertTrue(self.registry.claim("s1", "token-a"))
        self.assertEqual(self.registry.owner("s1"), "token-a")

    def test_another_token_cannot_claim_or_read(self):
        self.registry.claim("s1", "token-a")
        self.assertFalse(self.registry.claim("s1", "token-b"))
        self.assertFalse(self.registry.is_owner("s1", "token-b"))

    def test_tokenless_client_never_reaches_an_owned_session(self):
        self.registry.claim("s1", "token-a")
        self.assertFalse(self.registry.claim("s1", None))
        self.assertFalse(self.registry.is_owner("s1", None))

    def test_owner_keeps_access_on_repeat_requests(self):
        self.registry.claim("s1", "token-a")
        self.assertTrue(self.registry.claim("s1", "token-a"))
        self.assertTrue(self.registry.is_owner("s1", "token-a"))

    def test_session_without_a_token_stays_unowned(self):
        self.assertTrue(self.registry.claim("s2", None))
        self.assertIsNone(self.registry.owner("s2"))

    def test_session_never_seen_holds_nothing_to_protect(self):
        self.assertTrue(self.registry.is_owner("unknown-session", "token-a"))


class SessionGuardTests(unittest.TestCase):
    """The HTTP-level guards wired into the chat endpoints."""

    def setUp(self):
        self.registry = SessionRegistry(store=InMemoryStore())
        self._original_getter = security.get_session_registry
        security.get_session_registry = lambda: self.registry

    def tearDown(self):
        security.get_session_registry = self._original_getter

    def test_reading_another_clients_history_is_forbidden(self):
        security.claim_session("s1", "token-a")
        with self.assertRaises(HTTPException) as ctx:
            security.assert_session_owner("s1", "token-b")
        self.assertEqual(ctx.exception.status_code, 403)

    def test_owner_passes_the_read_guard(self):
        security.claim_session("s1", "token-a")
        security.assert_session_owner("s1", "token-a")

    def test_hijacking_a_session_on_chat_is_forbidden(self):
        security.claim_session("s1", "token-a")
        with self.assertRaises(HTTPException) as ctx:
            security.claim_session("s1", "token-b")
        self.assertEqual(ctx.exception.status_code, 403)

    def test_new_session_is_bound_on_first_message(self):
        security.claim_session("fresh-session", "token-a")
        self.assertEqual(self.registry.owner("fresh-session"), "token-a")


class RefusalDetectionTests(unittest.TestCase):
    """Refusals are whole-answer properties, not clauses inside an answer."""

    def test_chinese_refusal_is_rejected(self):
        # Long enough (> MIN_RESPONSE_LENGTH) that this exercises the refusal
        # detector rather than the "too short" guard.
        valid, reason = is_valid_response(
            "抱歉，我无法回答这个问题，因为当前知识库中确实没有与它相关的任何资料可供参考，"
            "建议你先补充相关文档再来提问。",
            "concept_qa",
        )
        self.assertFalse(valid)
        self.assertIn("refusal", reason.lower())

    def test_english_refusal_is_rejected(self):
        valid, _ = is_valid_response(
            "Sorry, I cannot answer that question right now, try the docs.",
            "concept_qa",
        )
        self.assertFalse(valid)

    def test_conditional_hedge_is_not_a_refusal(self):
        # The false positive that used to burn a retry: "我无法" sits behind a
        # conditional clause, in an answer that is otherwise complete.
        answer = (
            "如果超时时间设置过短，我无法保证分片请求一定成功，因此建议把 "
            "bufferForPlaybackMs 调大，并配合 RetryPolicy 做指数退避重试。"
        )
        valid, reason = is_valid_response(answer, "concept_qa")
        self.assertTrue(valid, msg=reason)

    def test_mid_sentence_hedge_is_not_a_refusal(self):
        answer = (
            "总的来说这个 crash 的根因是主线程 IO，我无法把它简单归到某一个模块，"
            "建议结合堆栈继续确认调用链。"
        )
        valid, reason = is_valid_response(answer, "concept_qa")
        self.assertTrue(valid, msg=reason)


class RetrievalFilterTests(unittest.TestCase):
    """The retriever has one filter rule, shared by both search halves."""

    def setUp(self):
        # Bypass __init__ so the test never touches ChromaDB.
        self.retriever = retriever_module.HybridRetriever.__new__(
            retriever_module.HybridRetriever
        )
        self.calls = []

        calls = self.calls

        class _EmptyWhenFiltered:
            def similarity_search(self, query, top_k=10, where=None):
                calls.append(where)
                if where:
                    return []
                return [
                    {
                        "doc_id": "d1",
                        "content": "Compose 列表用 LazyColumn 实现",
                        "source": "guide.md",
                        "score": 0.9,
                        "metadata": {"type": "markdown"},
                    }
                ]

        self.retriever.vs = _EmptyWhenFiltered()
        self._original_bm25 = retriever_module.get_bm25_index
        retriever_module.get_bm25_index = lambda: None

    def tearDown(self):
        retriever_module.get_bm25_index = self._original_bm25

    def test_filter_matches_chroma_where_semantics(self):
        self.assertTrue(_matches_filter({"type": "code"}, {"type": "code"}))
        self.assertFalse(_matches_filter({"type": "markdown"}, {"type": "code"}))
        self.assertTrue(_matches_filter({"type": "markdown"}, None))

    def test_filter_is_forwarded_to_the_vector_store(self):
        self.retriever.hybrid_search("query", where_filter={"type": "code"})
        self.assertEqual(self.calls[0], {"type": "code"})

    def test_empty_filtered_result_falls_back_to_unfiltered(self):
        docs = self.retriever.hybrid_search("query", where_filter={"type": "code"})
        # Filtered attempt first, then the unfiltered retry.
        self.assertEqual(self.calls, [{"type": "code"}, None])
        self.assertEqual(len(docs), 1)

    def test_no_fallback_when_the_filter_matches(self):
        docs = self.retriever.hybrid_search("query", where_filter=None)
        self.assertEqual(self.calls, [None])
        self.assertEqual(len(docs), 1)


class IntentFilterWiringTests(unittest.TestCase):
    """Retrieval breadth and filtering must come from the router config."""

    def _capture(self, intent: str, kb_ids=None):
        captured = {}

        class _StubRetriever:
            def hybrid_search(
                self,
                query,
                keywords=None,
                where_filter=None,
                fallback_filter=None,
                top_k=None,
            ):
                captured["where_filter"] = where_filter
                captured["fallback_filter"] = fallback_filter
                captured["top_k"] = top_k
                return []

        original = rag_retrieve_module.get_retriever
        rag_retrieve_module.get_retriever = lambda: _StubRetriever()
        try:
            state = AgentState(
                query="注意力机制是什么",
                intent=intent,
                query_keywords=[],
                kb_ids=kb_ids or [],
            )
            rag_retrieve_module.rag_retrieve_node(state)
        finally:
            rag_retrieve_module.get_retriever = original
        return captured

    def test_a_survey_searches_more_broadly_than_a_concept_question(self):
        broad = self._capture("literature_review")["top_k"]
        narrow = self._capture("concept_qa")["top_k"]
        self.assertGreater(broad, narrow)

    def test_unknown_intent_uses_the_default_config(self):
        captured = self._capture("not-an-intent")
        self.assertEqual(
            captured["top_k"], get_intent_config(DEFAULT_INTENT)["retrieve_k"]
        )
        self.assertIsNone(captured["where_filter"])

    def test_no_research_intent_narrows_by_chunk_type(self):
        # The old code-vs-prose distinction has no counterpart in a multi-domain
        # knowledge base, and a knob nothing sets is worse than no knob. The
        # mechanism stays (and is exercised above) for deployments that do
        # partition their collection by chunk type.
        for intent in INTENT_CONFIG:
            self.assertIsNone(get_intent_config(intent)["collection_filter"])

    def test_no_base_selection_searches_everything(self):
        captured = self._capture("concept_qa")
        self.assertIsNone(captured["where_filter"])
        self.assertIsNone(captured["fallback_filter"])

    def test_selected_bases_scope_the_vector_store_query(self):
        captured = self._capture("concept_qa", kb_ids=["kb_a", "kb_b"])
        self.assertEqual(captured["where_filter"], {"kb_id": {"$in": ["kb_a", "kb_b"]}})

    def test_the_base_scope_is_also_the_fallback(self):
        # So a question that matches nothing inside the selected bases degrades
        # to "no context" rather than silently searching excluded bases.
        captured = self._capture("concept_qa", kb_ids=["kb_a"])
        self.assertEqual(captured["fallback_filter"], captured["where_filter"])


class KnowledgeBaseFilterTests(unittest.TestCase):
    """The base scope is an instruction; the intent filter is only a hint."""

    def test_no_scope_and_no_intent_filter_means_no_clause(self):
        self.assertIsNone(build_where_filter(None, None))
        # An empty list means "every base" — the absence of a clause, not a
        # clause matching nothing.
        self.assertIsNone(build_where_filter(None, []))

    def test_scope_alone_becomes_a_single_clause(self):
        self.assertEqual(
            build_where_filter(None, ["kb_a"]), {"kb_id": {"$in": ["kb_a"]}}
        )

    def test_intent_filter_and_scope_are_combined_with_and(self):
        self.assertEqual(
            build_where_filter({"type": "code"}, ["kb_a", "kb_b"]),
            {"$and": [{"type": "code"}, {"kb_id": {"$in": ["kb_a", "kb_b"]}}]},
        )

    def test_filter_supports_in_and_and(self):
        meta = {"type": "markdown", "kb_id": "kb_a"}
        self.assertTrue(_matches_filter(meta, {"kb_id": {"$in": ["kb_a", "kb_b"]}}))
        self.assertFalse(_matches_filter(meta, {"kb_id": {"$in": ["kb_b"]}}))
        self.assertTrue(
            _matches_filter(meta, {"$and": [{"type": "markdown"}, {"kb_id": "kb_a"}]})
        )
        self.assertFalse(
            _matches_filter(meta, {"$and": [{"type": "code"}, {"kb_id": "kb_a"}]})
        )

    def test_unknown_operator_fails_closed(self):
        # Matching everything here would make the lexical half disagree with
        # Chroma instead of merely returning nothing.
        self.assertFalse(
            _matches_filter({"kb_id": "kb_a"}, {"kb_id": {"$regex": "a"}})
        )


class KnowledgeBaseScopeFallbackTests(unittest.TestCase):
    """A filter matching nothing must not widen the user's base scope."""

    def setUp(self):
        self.retriever = retriever_module.HybridRetriever.__new__(
            retriever_module.HybridRetriever
        )
        self.calls = []

        calls = self.calls

        class _OnlyUnfilteredMatches:
            def similarity_search(self, query, top_k=10, where=None):
                calls.append(where)
                if where:
                    return []
                return [
                    {
                        "doc_id": "d1",
                        "content": "随便一段内容",
                        "source": "a.md",
                        "score": 0.9,
                        "metadata": {},
                    }
                ]

        self.retriever.vs = _OnlyUnfilteredMatches()
        self._original_bm25 = retriever_module.get_bm25_index
        retriever_module.get_bm25_index = lambda: None
        self.addCleanup(
            setattr, retriever_module, "get_bm25_index", self._original_bm25
        )

    def test_base_only_scope_is_not_widened_when_it_matches_nothing(self):
        scope = build_where_filter(None, ["kb_a"])
        docs = self.retriever.hybrid_search(
            "q", where_filter=scope, fallback_filter=scope
        )
        self.assertEqual(self.calls, [scope])
        self.assertEqual(docs, [])

    def test_the_intent_hint_is_dropped_while_the_scope_survives(self):
        scope = build_where_filter(None, ["kb_a"])
        combined = build_where_filter({"type": "code"}, ["kb_a"])
        self.retriever.hybrid_search(
            "q", where_filter=combined, fallback_filter=scope
        )
        # Filtered attempt with the intent hint, then the same scope without it.
        self.assertEqual(self.calls, [combined, scope])


class KnowledgeBaseRegistryTests(unittest.TestCase):
    """Bases are named partitions; the registry is the only list of them."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.registry = kb_registry_module.KnowledgeBaseRegistry(
            path=Path(self._tmp.name) / "knowledge_bases.json"
        )

    def test_default_base_is_created_once(self):
        first = self.registry.ensure_default()
        second = self.registry.ensure_default()
        self.assertEqual(first.id, DEFAULT_KB_ID)
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(self.registry.list_all()), 1)

    def test_create_persists_and_survives_a_reload(self):
        created = self.registry.create("深度学习文献")
        self.assertTrue(created.id.startswith("kb_"))
        reloaded = kb_registry_module.KnowledgeBaseRegistry(path=self.registry.path)
        self.assertEqual([kb.name for kb in reloaded.list_all()], ["深度学习文献"])

    def test_names_are_unique_regardless_of_spacing_or_case(self):
        self.registry.create("Survey Papers")
        with self.assertRaises(DuplicateKnowledgeBaseName):
            self.registry.create("  survey   papers ")

    def test_blank_name_is_rejected(self):
        with self.assertRaises(InvalidKnowledgeBaseName):
            self.registry.create("   ")

    def test_rename_and_delete(self):
        kb = self.registry.create("临时")
        self.assertEqual(self.registry.update(kb.id, name="正式名称").name, "正式名称")
        self.registry.delete(kb.id)
        self.assertEqual(self.registry.list_all(), [])

    def test_renaming_onto_a_taken_name_is_rejected_and_changes_nothing(self):
        self.registry.create("A")
        second = self.registry.create("B")
        with self.assertRaises(DuplicateKnowledgeBaseName):
            self.registry.update(second.id, name="a")
        self.assertEqual(self.registry.get(second.id).name, "B")

    def test_a_base_keeps_its_own_name_when_renaming_to_it(self):
        kb = self.registry.create("A")
        self.assertEqual(self.registry.update(kb.id, name="A").name, "A")

    def test_unknown_ids_resolve_to_themselves(self):
        # A citation to a base deleted mid-stream stays traceable instead of
        # losing its label.
        self.assertEqual(self.registry.names_for(["kb_gone"])["kb_gone"], "kb_gone")


class _TempVectorStoreCase(unittest.TestCase):
    """Repoint the vector store at a throwaway Chroma directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._orig_persist = settings.CHROMA_PERSIST_DIR
        self._orig_collection = settings.CHROMA_COLLECTION
        settings.CHROMA_PERSIST_DIR = self._tmp.name
        settings.CHROMA_COLLECTION = "test_collection"
        vectorstore_module.get_vector_store.cache_clear()
        self.addCleanup(self._restore)

    def _restore(self):
        vectorstore_module.get_vector_store.cache_clear()
        settings.CHROMA_PERSIST_DIR = self._orig_persist
        settings.CHROMA_COLLECTION = self._orig_collection
        self._tmp.cleanup()

    def seed_one_chunk(self):
        store = vectorstore_module.get_vector_store()
        store.add_documents(
            [
                {
                    "content": "Compose 列表用 LazyColumn 实现",
                    "source": "a.md",
                    "type": "markdown",
                    "metadata": {},
                }
            ]
        )
        return store


class EmbeddingDimensionGuardTests(_TempVectorStoreCase):
    """A collection the current embedder cannot serve must fail loudly."""

    def test_collection_built_by_the_same_embedder_is_accepted(self):
        self.assertEqual(self.seed_one_chunk().count(), 1)

    def test_embedder_with_a_different_width_is_rejected(self):
        store = self.seed_one_chunk()

        class _WideEmbedder:
            # chromadb validates the embedding-function signature, so the
            # parameter must be named exactly `input`.
            def __call__(self, input):
                return [[0.0] * 8 for _ in input]

            def name(self):
                return "stub-wide"

        original = vectorstore_module._build_embedding_function
        vectorstore_module._build_embedding_function = lambda: _WideEmbedder()
        self.addCleanup(
            setattr, vectorstore_module, "_build_embedding_function", original
        )

        # Simulate a restart where the embedder now resolves differently while
        # the persisted collection keeps the vectors it was built with.
        store._collection = None
        store._ef = None

        with self.assertRaises(RuntimeError) as ctx:
            store.count()
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

        # And it keeps refusing, rather than failing once and then proceeding
        # with a collection it cannot query correctly.
        with self.assertRaises(RuntimeError):
            store.count()


class IngestPipelineTests(_TempVectorStoreCase):
    """Seeding a directory is idempotent because chunks are content-addressed."""

    def setUp(self):
        super().setUp()
        self.docs = Path(self._tmp.name) / "seed_docs"
        self.docs.mkdir()

    def test_directory_is_indexed_and_a_rerun_adds_nothing(self):
        (self.docs / "guide.md").write_text(
            "# Compose 列表\n\n用 LazyColumn 实现长列表，配合 remember 缓存状态。",
            encoding="utf-8",
        )
        (self.docs / "notes.txt").write_text(
            "弱网环境下应当给播放器配置更长的超时与重试策略。", encoding="utf-8"
        )
        (self.docs / "ignored.bin").write_text("not a document", encoding="utf-8")

        first = ingest_module.ingest_directory(self.docs)
        self.assertEqual(set(first.added_by_file), {"guide.md", "notes.txt"})
        self.assertGreater(first.added_total, 0)
        self.assertEqual(first.failed, {})

        second = ingest_module.ingest_directory(self.docs)
        self.assertEqual(second.added_total, 0)

    def test_missing_directory_is_not_an_error(self):
        summary = ingest_module.ingest_directory(self.docs / "does-not-exist")
        self.assertEqual(summary.added_by_file, {})
        self.assertEqual(summary.added_total, 0)


class KnowledgeBasePartitionTests(_TempVectorStoreCase):
    """Chunks carry their base, so scoping them is a metadata filter."""

    @staticmethod
    def _seed(kb_id: str, source: str, content: str) -> int:
        return vectorstore_module.get_vector_store().add_documents(
            [{"content": content, "source": source, "type": "markdown", "metadata": {}}],
            kb_id=kb_id,
        )

    def test_counts_are_scoped_per_base(self):
        store = vectorstore_module.get_vector_store()
        self._seed("kb_a", "a.md", "Compose 列表用 LazyColumn")
        self._seed("kb_b", "b.md", "MVCC 与隔离级别")
        self.assertEqual(store.count(), 2)
        self.assertEqual(store.count("kb_a"), 1)
        self.assertEqual(store.count("kb_b"), 1)

    def test_the_same_document_in_two_bases_is_stored_twice(self):
        # The base is part of the content hash, so an identical file uploaded to
        # two bases does not collide into a bogus "already indexed".
        self.assertEqual(self._seed("kb_a", "same.md", "同样的内容"), 1)
        self.assertEqual(self._seed("kb_b", "same.md", "同样的内容"), 1)

    def test_identical_content_in_the_same_base_is_idempotent(self):
        self.assertEqual(self._seed("kb_a", "same.md", "同样的内容"), 1)
        self.assertEqual(self._seed("kb_a", "same.md", "同样的内容"), 0)

    def test_stats_report_files_and_chunks_per_base(self):
        self._seed("kb_a", "a.md", "内容一")
        self._seed("kb_a", "b.md", "内容二")
        self._seed("kb_b", "c.md", "内容三")
        stats = vectorstore_module.get_vector_store().base_stats()
        self.assertEqual(stats["kb_a"], {"chunks": 2, "documents": 2})
        self.assertEqual(stats["kb_b"], {"chunks": 1, "documents": 1})

    def test_deleting_a_base_leaves_the_others_intact(self):
        store = vectorstore_module.get_vector_store()
        self._seed("kb_a", "a.md", "内容一")
        self._seed("kb_b", "b.md", "内容二")
        self.assertEqual(store.delete_by_kb("kb_a"), 1)
        self.assertEqual(store.count("kb_a"), 0)
        self.assertEqual(store.count("kb_b"), 1)

    def test_chunk_listing_can_be_scoped(self):
        self._seed("kb_a", "a.md", "内容一")
        self._seed("kb_b", "b.md", "内容二")
        store = vectorstore_module.get_vector_store()
        self.assertEqual(len(store.get_all_chunks()), 2)
        scoped = store.get_all_chunks("kb_a")
        self.assertEqual([chunk["source"] for chunk in scoped], ["a.md"])
        self.assertEqual(scoped[0]["kb_id"], "kb_a")

    def test_chunks_without_a_base_are_attributed_to_the_default(self):
        # Read-time defence for chunks written before partitioning existed:
        # reported under the default base rather than as unattached, which would
        # make them invisible in the UI while still occupying the index.
        self.assertEqual(
            vectorstore_module._kb_of({"source": "legacy.md"}), DEFAULT_KB_ID
        )
        self.assertEqual(vectorstore_module._kb_of(None), DEFAULT_KB_ID)
        self.assertEqual(vectorstore_module._kb_of({"kb_id": "kb_a"}), "kb_a")


class RerankerResolutionTests(unittest.TestCase):
    """A reranker that is enabled but inert is worse than one switched off."""

    def _bare_reranker(self, mode: str, model=None):
        reranker = reranker_module.BGEReranker.__new__(reranker_module.BGEReranker)
        reranker._model = model
        reranker._mode = mode
        return reranker

    def test_sigmoid_maps_logits_to_probabilities(self):
        from app.rag.onnx_reranker import _sigmoid

        self.assertAlmostEqual(_sigmoid(0.0), 0.5, places=6)
        self.assertGreater(_sigmoid(8.0), 0.999)
        self.assertLess(_sigmoid(-8.0), 0.001)
        # The naive form overflows to math domain error for large negatives.
        self.assertGreaterEqual(_sigmoid(-800.0), 0.0)

    def test_missing_onnx_assets_raise_an_actionable_error(self):
        from app.rag.onnx_reranker import OnnxCrossEncoderReranker

        missing = Path(tempfile.gettempdir()) / "definitely-not-a-reranker"
        reranker = OnnxCrossEncoderReranker(str(missing))

        # Loading is lazy, so construction succeeds and only scoring fails.
        with self.assertRaises(FileNotFoundError) as ctx:
            reranker.score([("q", "p")])
        message = str(ctx.exception)
        self.assertIn("onnx", message.lower())
        self.assertIn("RERANKER_ONNX_PATH", message)  # says how to fix it

    def test_both_tokenizer_layouts_resolve(self):
        # bge-reranker-base keeps tokenizer.json at the model root while bge-m3
        # keeps it inside onnx/. Requiring both in onnx/ would break a folder
        # downloaded straight from the Hub.
        from app.rag.onnx_reranker import OnnxCrossEncoderReranker

        for tokenizer_at_root in (True, False):
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                base = Path(tmp)
                (base / "onnx").mkdir()
                (base / "onnx" / "model.onnx").write_bytes(b"stub")
                tokenizer_dir = base if tokenizer_at_root else base / "onnx"
                (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

                model_file, tokenizer_file = OnnxCrossEncoderReranker(
                    str(base)
                )._resolve_files()

                with self.subTest(tokenizer_at_root=tokenizer_at_root):
                    self.assertEqual(model_file, base / "onnx" / "model.onnx")
                    self.assertEqual(tokenizer_file, tokenizer_dir / "tokenizer.json")

    def test_onnx_directory_is_detected_from_either_setting(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            onnx_dir = Path(tmp) / "bge-reranker-base" / "onnx"
            onnx_dir.mkdir(parents=True)
            (onnx_dir / "model.onnx").write_bytes(b"stub")
            (onnx_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            model_dir = str(onnx_dir.parent)

            for field in ("RERANKER_ONNX_PATH", "RERANKER_MODEL"):
                self.addCleanup(
                    setattr, settings, field, getattr(settings, field)
                )

            settings.RERANKER_ONNX_PATH = model_dir
            settings.RERANKER_MODEL = ""
            self.assertEqual(settings.reranker_onnx_dir, Path(model_dir))

            # Writing the local directory into RERANKER_MODEL must work too —
            # that is the field a user would naturally reach for.
            settings.RERANKER_ONNX_PATH = ""
            settings.RERANKER_MODEL = model_dir
            self.assertEqual(settings.reranker_onnx_dir, Path(model_dir))

            # A genuine HuggingFace repo id resolves to nothing.
            settings.RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
            self.assertIsNone(settings.reranker_onnx_dir)

    def test_all_three_backends_rank_by_their_scores(self):
        class _OnnxStub:
            def score(self, pairs):
                return [0.2, 0.8]

        class _FlagStub:
            def compute_score(self, pairs, normalize=True):
                return [0.2, 0.8]

        class _CrossEncoderStub:
            def predict(self, pairs):
                return [0.2, 0.8]

        for mode, model in (
            ("onnx", _OnnxStub()),
            ("flag", _FlagStub()),
            ("cross_encoder", _CrossEncoderStub()),
        ):
            reranker = self._bare_reranker(mode, model)
            docs = [
                {"content": "低相关", "source": "low.md"},
                {"content": "高相关", "source": "high.md"},
            ]
            result = reranker.rerank("问题", docs, top_k=2)
            with self.subTest(mode=mode):
                self.assertEqual([d["source"] for d in result], ["high.md", "low.md"])
                self.assertAlmostEqual(result[0]["rerank_score"], 0.8)

    def test_fused_order_is_kept_when_no_cross_encoder_resolves(self):
        reranker = self._bare_reranker("score_sort")
        docs = [
            {"content": "a", "source": "a.md", "score": 0.9},
            {"content": "b", "source": "b.md", "score": 0.5},
            {"content": "c", "source": "c.md", "score": 0.7},
        ]
        result = reranker.rerank("q", docs, top_k=2)
        # Ranking by `score` here would be a no-op — it already *is* the fused
        # order — which is exactly why this mode reports itself as inactive.
        self.assertEqual([d["source"] for d in result], ["a.md", "b.md"])

    def test_this_environment_reports_score_sort_rather_than_pretending(self):
        if settings.reranker_onnx_dir is not None:
            self.skipTest("an ONNX reranker is configured here")
        # No FlagEmbedding / sentence-transformers is expected in this venv. The
        # point is not that reranking is unavailable, but that the resolved mode
        # says so instead of claiming a precision pass it is not performing.
        self.assertEqual(reranker_module.BGEReranker().mode, "score_sort")


class OnnxEmbeddingConfigTests(unittest.TestCase):
    """Error paths that must work regardless of what is installed locally."""

    def test_missing_assets_raise_an_actionable_error(self):
        from app.rag.onnx_embedding import OnnxBGEEmbeddingFunction

        missing = Path(tempfile.gettempdir()) / "definitely-not-a-bge-model"
        embedder = OnnxBGEEmbeddingFunction(str(missing))

        # Loading is lazy, so construction succeeds and only the first real
        # embedding request fails.
        with self.assertRaises(FileNotFoundError) as ctx:
            embedder(["hello"])
        message = str(ctx.exception)
        self.assertIn("onnx", message.lower())
        self.assertIn("EMBEDDING_MODE", message)  # says how to recover

    def test_config_round_trips_through_the_chroma_interface(self):
        from app.rag.onnx_embedding import OnnxBGEEmbeddingFunction

        config = OnnxBGEEmbeddingFunction("some/dir").get_config()
        rebuilt = OnnxBGEEmbeddingFunction.build_from_config(config)
        self.assertEqual(rebuilt.model_path, "some/dir")
        self.assertEqual(OnnxBGEEmbeddingFunction.name(), "bge_m3_onnx")


class OnnxEmbeddingBehaviourTests(unittest.TestCase):
    """Runs only where a local ONNX BGE-M3 export is configured."""

    @classmethod
    def setUpClass(cls):
        from app.rag.onnx_embedding import OnnxBGEEmbeddingFunction

        model_path = settings.EMBEDDING_MODEL
        if settings.EMBEDDING_MODE != "onnx" or not (
            Path(model_path) / "onnx" / "model.onnx"
        ).is_file():
            raise unittest.SkipTest(
                "no local ONNX BGE-M3 export configured (set EMBEDDING_MODE=onnx "
                "and EMBEDDING_MODEL to a model directory)"
            )
        # One load for the whole class: the graph is a few GB in memory.
        cls.embedder = OnnxBGEEmbeddingFunction(model_path)

    @staticmethod
    def _cosine(left, right):
        return sum(a * b for a, b in zip(left, right))

    def test_emits_normalised_bge_m3_vectors(self):
        vectors = self.embedder(["Compose 列表用 LazyColumn 实现", "MySQL 的隔离级别"])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 1024)
        for vector in vectors:
            norm = sum(value * value for value in vector) ** 0.5
            self.assertAlmostEqual(norm, 1.0, places=4)

    def test_semantic_relatedness_beats_topic_mismatch(self):
        isolation, mvcc, compose = self.embedder(
            [
                "MySQL 的隔离级别",
                "MVCC 通过版本链实现一致性读",
                "Compose 列表用 LazyColumn 实现",
            ]
        )
        # The point of semantic search: no shared tokens, still related.
        self.assertGreater(
            self._cosine(isolation, mvcc), self._cosine(isolation, compose)
        )

    def test_empty_input_short_circuits(self):
        self.assertEqual(self.embedder([]), [])


class RedisHealthReportingTests(unittest.TestCase):
    """The health endpoint must not report Redis as healthy when it is absent."""

    @staticmethod
    def _fallback_store() -> RedisStore:
        # Bypass __init__ so the test never waits on a connection timeout.
        store = RedisStore.__new__(RedisStore)
        store._client = None
        store._fallback = InMemoryStore()
        return store

    def test_fallback_store_reports_its_backend(self):
        self.assertEqual(self._fallback_store().backend, "in_memory_fallback")

    def test_health_marks_redis_degraded_when_the_fallback_serves(self):
        class _StubVectorStore:
            def count(self) -> int:
                return 0

        originals = {
            "get_redis_store": health_module.get_redis_store,
            "get_vector_store": health_module.get_vector_store,
        }
        health_module.get_redis_store = self._fallback_store
        health_module.get_vector_store = _StubVectorStore
        for name, original in originals.items():
            self.addCleanup(setattr, health_module, name, original)

        payload = asyncio.run(health_module.health_check())
        redis_status = payload.services["redis"]

        # A set/get roundtrip would have succeeded against the fallback, which
        # is exactly how this used to be misreported as healthy.
        self.assertEqual(redis_status.status, "degraded")
        self.assertEqual(redis_status.backend, "in_memory_fallback")
        self.assertIn("restart", redis_status.error)


class SourceCollectionTests(unittest.TestCase):
    """Citations carry the base, because a filename alone is ambiguous."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._orig_path = settings.KB_REGISTRY_PATH
        settings.KB_REGISTRY_PATH = str(Path(self._tmp.name) / "knowledge_bases.json")
        kb_registry_module.get_kb_registry.cache_clear()
        self.addCleanup(self._restore)

    def _restore(self):
        kb_registry_module.get_kb_registry.cache_clear()
        settings.KB_REGISTRY_PATH = self._orig_path
        self._tmp.cleanup()

    def test_dedupes_while_keeping_relevance_order(self):
        docs = [
            {"source": "primary.md", "metadata": {"kb_id": "kb_a"}},
            {"source": "secondary.md", "metadata": {"kb_id": "kb_a"}},
            {"source": "primary.md", "metadata": {"kb_id": "kb_a"}},
            {"source": None, "metadata": {"kb_id": "kb_a"}},
        ]
        sources = chat_module._collect_sources(docs)
        self.assertEqual(
            [ref["document"] for ref in sources], ["primary.md", "secondary.md"]
        )

    def test_the_same_filename_in_two_bases_is_two_citations(self):
        docs = [
            {"source": "notes.md", "metadata": {"kb_id": "kb_a"}},
            {"source": "notes.md", "metadata": {"kb_id": "kb_b"}},
        ]
        sources = chat_module._collect_sources(docs)
        self.assertEqual(
            [ref["knowledge_base_id"] for ref in sources], ["kb_a", "kb_b"]
        )

    def test_a_citation_names_its_base(self):
        kb_registry_module.get_kb_registry().create("深度学习文献")
        kb_id = kb_registry_module.get_kb_registry().list_all()[0].id
        docs = [{"source": "attention.md", "metadata": {"kb_id": kb_id}}]
        self.assertEqual(
            chat_module._collect_sources(docs)[0]["knowledge_base_name"], "深度学习文献"
        )

    def test_an_unknown_base_stays_traceable_instead_of_vanishing(self):
        docs = [{"source": "notes.md", "metadata": {"kb_id": "kb_gone"}}]
        self.assertEqual(
            chat_module._collect_sources(docs)[0]["knowledge_base_name"], "kb_gone"
        )

    def test_handles_an_empty_retrieval(self):
        self.assertEqual(chat_module._collect_sources(None), [])
        self.assertEqual(chat_module._collect_sources([]), [])


class PromptContextTests(unittest.TestCase):
    """The model sees which base each passage came from, not just a filename."""

    def test_each_passage_is_labelled_with_its_base(self):
        from app.agent.agents.base import format_context

        docs = [
            {
                "content": "自注意力把序列中任意两个位置直接关联起来。",
                "source": "transformer.md",
                "metadata": {"kb_id": "kb_ml"},
            }
        ]
        context = format_context(docs)
        self.assertIn("transformer.md", context)
        self.assertIn("知识库：kb_ml", context)

    def test_no_retrieval_is_stated_explicitly(self):
        from app.agent.agents.base import format_context

        self.assertIn("未检索到", format_context([]))


if __name__ == "__main__":
    unittest.main()
