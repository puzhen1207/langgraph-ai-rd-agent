"""
Node 3: RAG Retrieve
Performs hybrid BM25 + Vector retrieval from the selected knowledge bases.
"""
from typing import Dict, Any

from app.agent.nodes.intent_router import DEFAULT_INTENT, get_intent_config
from app.agent.state import AgentState
from app.core.logging_config import get_logger
from app.rag.retriever import build_where_filter, get_retriever

logger = get_logger(__name__)


def rag_retrieve_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    keywords = state.get("query_keywords", [])
    intent = state.get("intent", DEFAULT_INTENT)
    kb_ids = state.get("kb_ids") or []

    # The same config the router logged, so retrieval breadth and filtering have
    # exactly one definition and cannot drift away from the routing rule.
    config = get_intent_config(intent)

    # Two filters with different meanings, and therefore different fallbacks. The
    # per-intent collection filter is a retrieval hint; the knowledge-base scope
    # is the user's instruction. Dropping the former when it matches nothing is
    # the fix for an empty context; dropping the latter would silently answer
    # from bases the user deliberately excluded.
    kb_filter = build_where_filter(None, kb_ids)
    where_filter = build_where_filter(config["collection_filter"], kb_ids)

    logger.info(
        "rag_retrieve_start",
        query=query[:80],
        intent=intent,
        retrieve_k=config["retrieve_k"],
        kb_ids=kb_ids or "all",
        where_filter=where_filter,
    )

    try:
        retriever = get_retriever()
        docs = retriever.hybrid_search(
            query=query,
            keywords=keywords,
            where_filter=where_filter,
            fallback_filter=kb_filter,
            top_k=config["retrieve_k"],
        )
        logger.info("rag_retrieved", count=len(docs))
        return {"retrieved_docs": docs}
    except Exception as e:
        logger.error("rag_retrieve_failed", error=str(e))
        return {"retrieved_docs": [], "error": f"RAG retrieval failed: {e}"}
