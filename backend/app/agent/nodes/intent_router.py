"""
Node 2: Intent Router
Enriches the state with routing metadata based on the detected intent.
"""
from typing import Any, Dict

from app.agent.state import AgentState
from app.core.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_INTENT = "concept_qa"

# Single source of truth for per-intent routing AND retrieval behaviour.
# Only fields that are actually consumed live here: dead knobs (a declared but
# never-read prompt key or requires_rag flag) are worse than no knob at all,
# because they read as configuration while changing nothing.
INTENT_CONFIG = {
    "concept_qa": {
        "description": "概念与原理问答",
        # A single-concept question does not need a wide net, and a narrow one
        # keeps the prompt focused.
        "retrieve_k": 10,
        "rerank_k": 5,
        "collection_filter": None,
    },
    "literature_review": {
        "description": "文献综述与研究现状",
        # Surveying a field needs coverage rather than precision: widen the
        # search and keep more passages so the answer can report where sources
        # agree and disagree, instead of summarising the single best match.
        "retrieve_k": 24,
        "rerank_k": 10,
        "collection_filter": None,
    },
    "method_compare": {
        "description": "方法对比与选型",
        # Needs at least one passage on each side of the comparison.
        "retrieve_k": 16,
        "rerank_k": 8,
        "collection_filter": None,
    },
    "academic_writing": {
        "description": "学术写作与润色",
        # Grounded mainly in the draft the user pasted, so a little background
        # material is enough and a large context would only dilute the prompt.
        "retrieve_k": 8,
        "rerank_k": 4,
        "collection_filter": None,
    },
}


def get_intent_config(intent: str) -> Dict[str, Any]:
    """Single source of truth for per-intent routing and retrieval settings.

    Retrieval must read its filter and breadth from here rather than
    re-deriving them from ``intent``, so the rule cannot drift between the
    router and the retriever.
    """
    return INTENT_CONFIG.get(intent, INTENT_CONFIG[DEFAULT_INTENT])


def intent_router_node(state: AgentState) -> Dict[str, Any]:
    intent = state.get("intent", DEFAULT_INTENT)
    config = get_intent_config(intent)

    logger.info(
        "intent_routed",
        intent=intent,
        description=config["description"],
        retrieve_k=config["retrieve_k"],
        rerank_k=config["rerank_k"],
        collection_filter=config["collection_filter"],
    )

    return {
        "intent": intent,
    }
