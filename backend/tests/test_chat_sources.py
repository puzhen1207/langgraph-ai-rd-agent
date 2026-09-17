"""Tests for chat.py's relevance-threshold filtering in source collection.

The reranker attaches a ``rerank_score`` to every doc when it actually runs.
``_collect_sources`` must:

- drop docs whose score is below ``RERANK_SCORE_THRESHOLD``
- keep docs that have no score at all (the reranker fell back to ``score_sort``
  and there is nothing to filter against)
- keep all docs when the threshold is set to 0

This is a unit test of the helper, not the graph: a full chat run drags in the
LLM, the retriever, and Chroma, none of which are interesting here.
"""
import unittest

from app.api.chat import _collect_sources, _passes_relevance_threshold
from app.core import config as config_module
from app.rag.kb_registry import DEFAULT_KB_ID


def _doc(source: str, score=None, kb_id: str = DEFAULT_KB_ID) -> dict:
    item = {"source": source, "content": f"body of {source}", "metadata": {"kb_id": kb_id}}
    if score is not None:
        item["rerank_score"] = score
    return item


class RelevanceThresholdTests(unittest.TestCase):
    def setUp(self):
        self._original = config_module.settings.RERANK_SCORE_THRESHOLD
        config_module.settings.RERANK_SCORE_THRESHOLD = 0.5

    def tearDown(self):
        config_module.settings.RERANK_SCORE_THRESHOLD = self._original

    def test_high_score_passes_threshold(self):
        self.assertTrue(_passes_relevance_threshold(_doc("a.md", 0.9)))

    def test_low_score_is_dropped(self):
        self.assertFalse(_passes_relevance_threshold(_doc("a.md", 0.2)))

    def test_score_just_below_threshold_is_dropped(self):
        # Boundary: the filter must be strict ``>=``, so a 0.499 score
        # does not squeak through a 0.5 threshold.
        self.assertFalse(_passes_relevance_threshold(_doc("a.md", 0.499)))

    def test_score_at_threshold_passes(self):
        self.assertTrue(_passes_relevance_threshold(_doc("a.md", 0.5)))

    def test_missing_score_means_reranker_fell_back_so_we_keep_it(self):
        # The reranker's score_sort fallback leaves the field absent. With no
        # number to compare we cannot honestly reject a passage; the operator
        # who set ``RERANK_SCORE_THRESHOLD`` asked us to filter noise from a
        # real score, not to second-guess a failed system that already
        # surfaced the issue in its startup log.
        self.assertTrue(_passes_relevance_threshold(_doc("a.md")))

    def test_zero_threshold_disables_filter(self):
        config_module.settings.RERANK_SCORE_THRESHOLD = 0.0
        # Even a clearly-irrelevant score must pass when the operator turned
        # the filter off — useful for A/B comparisons without a redeploy.
        self.assertTrue(_passes_relevance_threshold(_doc("a.md", 0.001)))


class CollectSourcesFilterTests(unittest.TestCase):
    def setUp(self):
        self._original = config_module.settings.RERANK_SCORE_THRESHOLD
        config_module.settings.RERANK_SCORE_THRESHOLD = 0.5

    def tearDown(self):
        config_module.settings.RERANK_SCORE_THRESHOLD = self._original

    def test_filtered_docs_do_not_appear_in_sources(self):
        docs = [
            _doc("high.md", 0.9),
            _doc("low.md", 0.1),
            _doc("mid.md", 0.6),
        ]
        result = _collect_sources(docs)
        self.assertEqual([ref["document"] for ref in result], ["high.md", "mid.md"])

    def test_all_filtered_yields_empty_sources(self):
        # The user-visible failure mode from the bug report: every passage was
        # below the threshold, so the frontend should now see zero chips.
        docs = [_doc("a.md", 0.1), _doc("b.md", 0.2), _doc("c.md", 0.3)]
        self.assertEqual(_collect_sources(docs), [])

    def test_missing_scores_are_not_filtered(self):
        # Score_sort fallback path. Two sources: one with a score above the
        # threshold, one with no score at all. Both must survive so the UI
        # does not silently swallow a working-but-noisy retriever.
        docs = [_doc("good.md", 0.8), _doc("fallback.md")]
        result = _collect_sources(docs)
        self.assertEqual([ref["document"] for ref in result], ["good.md", "fallback.md"])

    def test_filtered_docs_keep_relevance_order(self):
        # Two high-scoring docs from different rerank scores; the helper must
        # not shuffle the rest of the pipeline's relevance ordering.
        docs = [_doc("alpha.md", 0.7), _doc("bravo.md", 0.95), _doc("charlie.md", 0.55)]
        result = _collect_sources(docs)
        self.assertEqual([ref["document"] for ref in result], ["alpha.md", "bravo.md", "charlie.md"])

    def test_zero_threshold_keeps_everything(self):
        config_module.settings.RERANK_SCORE_THRESHOLD = 0.0
        docs = [_doc("a.md", 0.001), _doc("b.md", 0.0)]
        result = _collect_sources(docs)
        self.assertEqual([ref["document"] for ref in result], ["a.md", "b.md"])


if __name__ == "__main__":
    unittest.main()