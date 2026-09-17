"""
Node 5: Memory Inject
Injects conversation history and summary memory into context.
"""
from typing import Dict, Any

from app.agent.state import AgentState
from app.core.logging_config import get_logger
from app.memory.conversation import ConversationMemory

logger = get_logger(__name__)


def memory_inject_node(state: AgentState) -> Dict[str, Any]:
    session_id = state["session_id"]
    messages = state.get("messages", [])

    logger.info("memory_inject", session_id=session_id, history_len=len(messages))

    try:
        memory = ConversationMemory(session_id=session_id)
        context = memory.build_context(recent_turns=6)
        return {"memory_context": context}
    except Exception as e:
        logger.warning("memory_inject_failed", error=str(e))
        # Fallback: build context from current messages list
        recent = messages[-6:] if len(messages) > 6 else messages
        context_lines = []
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:300]
            context_lines.append(f"{role}: {content}")
        return {"memory_context": "\n".join(context_lines)}
