"""
Node 6: LLM Generate
Dispatches to the specialised agent for the detected research intent.
"""
from typing import Dict, Any

from app.agent.nodes.intent_router import DEFAULT_INTENT
from app.agent.state import AgentState
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def _build_agent(intent: str):
    """Explicit branches rather than a registry: each agent module is then
    imported only when that intent is actually used."""
    if intent == "literature_review":
        from app.agent.agents.literature_review import LiteratureReviewAgent
        return LiteratureReviewAgent()
    if intent == "method_compare":
        from app.agent.agents.method_compare import MethodCompareAgent
        return MethodCompareAgent()
    if intent == "academic_writing":
        from app.agent.agents.academic_writing import AcademicWritingAgent
        return AcademicWritingAgent()

    from app.agent.agents.concept_qa import ConceptQAAgent
    return ConceptQAAgent()


def llm_generate_node(state: AgentState) -> Dict[str, Any]:
    intent = state.get("intent", DEFAULT_INTENT)
    iteration = state.get("iteration", 0)

    logger.info("llm_generate", intent=intent, iteration=iteration)

    agent = _build_agent(intent)

    try:
        response = agent.generate(state)
        logger.info("llm_generated", intent=intent, response_len=len(response))
        return {
            "final_response": response,
            "iteration": iteration + 1,
            "messages": [{"role": "assistant", "content": response}],
        }
    except Exception as e:
        logger.error("llm_generate_failed", error=str(e))
        error_msg = f"生成失败，请稍后重试。错误：{str(e)}"
        return {
            "final_response": error_msg,
            "iteration": iteration + 1,
            "error": str(e),
            "messages": [{"role": "assistant", "content": error_msg}],
        }
