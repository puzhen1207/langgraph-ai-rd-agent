"""
Node 4: Rerank
Trims the retrieved passages down to the most useful ones for the intent.
"""
from typing import Dict, Any

from app.agent.nodes.intent_router import DEFAULT_INTENT, get_intent_config
from app.agent.state import AgentState
from app.core.logging_config import get_logger
from app.rag.reranker import get_reranker

logger = get_logger(__name__)


def rerank_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    top_k = get_intent_config(state.get("intent", DEFAULT_INTENT))["rerank_k"]

    if not docs:
        logger.warning("rerank_skip", reason="no docs retrieved")
        return {"reranked_docs": []}

    logger.info("rerank_start", input_count=len(docs), top_k=top_k)

    try:
        reranker = get_reranker()
        reranked = reranker.rerank(query=query, docs=docs, top_k=top_k)
        logger.info("rerank_done", output_count=len(reranked))
        return {"reranked_docs": reranked}
    except Exception as e:
        logger.warning("rerank_failed_fallback", error=str(e))
        # Fallback: use top-k from original retrieval
        return {"reranked_docs": docs[:top_k]}
