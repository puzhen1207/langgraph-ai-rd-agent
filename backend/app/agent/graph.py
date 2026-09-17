"""
LangGraph State Machine - AI R&D Agent Workflow

Flow:
  START
    -> query_analyze      (extract keywords, detect intent)
    -> intent_router      (enrich routing metadata)
    -> rag_retrieve       (hybrid BM25 + vector search)
    -> rerank             (BGE-Reranker precision boost)
    -> memory_inject      (inject conversation history)
    -> llm_generate       (dispatch to specialized agent)
    -> response_check     (validate quality)
    -> END | retry (llm_generate)
"""
from functools import lru_cache

from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.nodes.query_analyze import query_analyze_node
from app.agent.nodes.intent_router import intent_router_node
from app.agent.nodes.rag_retrieve import rag_retrieve_node
from app.agent.nodes.rerank import rerank_node
from app.agent.nodes.memory_inject import memory_inject_node
from app.agent.nodes.llm_generate import llm_generate_node
from app.agent.nodes.response_check import response_check_node, should_retry
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def build_agent_graph():
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("query_analyze", query_analyze_node)
    graph.add_node("intent_router", intent_router_node)
    graph.add_node("rag_retrieve", rag_retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("memory_inject", memory_inject_node)
    graph.add_node("llm_generate", llm_generate_node)
    graph.add_node("response_check", response_check_node)

    # Entry point (langgraph 0.0.x API)
    graph.set_entry_point("query_analyze")
    graph.add_edge("query_analyze", "intent_router")
    graph.add_edge("intent_router", "rag_retrieve")
    graph.add_edge("rag_retrieve", "rerank")
    graph.add_edge("rerank", "memory_inject")
    graph.add_edge("memory_inject", "llm_generate")
    graph.add_edge("llm_generate", "response_check")

    # Conditional: retry or end
    graph.add_conditional_edges(
        "response_check",
        should_retry,
        {
            "retry": "llm_generate",
            "end": END,
        },
    )

    compiled = graph.compile()
    logger.info("agent_graph_built")
    return compiled


@lru_cache(maxsize=1)
def get_agent_graph():
    return build_agent_graph()
