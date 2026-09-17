"""Base classes for the specialised research agents."""
from abc import ABC, abstractmethod
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agent.state import AgentState, RetrievedDoc
from app.core.config import settings
from app.core.logging_config import get_logger
from app.rag.kb_registry import DEFAULT_KB_ID, get_kb_registry

logger = get_logger(__name__)


def build_llm(temperature: float = 0.1, streaming: bool = True) -> ChatOpenAI:
    """Build a chat model that supports token callbacks.

    ``invoke`` still returns a complete message for synchronous requests. With
    LangChain callbacks attached, streaming models also expose real tokens to
    the SSE endpoint instead of replaying an already-completed response.
    """
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_API_BASE,
        temperature=temperature,
        streaming=streaming,
        max_tokens=4096,
    )


def format_context(docs: List[RetrievedDoc], max_chars: int = 6000) -> str:
    """Render retrieved passages for the prompt, each labelled with its base.

    The knowledge base is part of the label so the model can attribute a claim to
    a named collection ("根据深度学习知识库…") instead of citing a bare filename
    that may exist in several bases at once.
    """
    if not docs:
        return "（未检索到相关文档）"

    def kb_of(doc: RetrievedDoc) -> str:
        return (doc.get("metadata") or {}).get("kb_id") or DEFAULT_KB_ID

    names = get_kb_registry().names_for(kb_of(doc) for doc in docs)

    parts = []
    total = 0
    for i, doc in enumerate(docs, 1):
        source = doc.get("source", "unknown")
        kb_name = names.get(kb_of(doc), kb_of(doc))
        content = doc.get("content", "")
        snippet = f"[文档{i} | {source} | 知识库：{kb_name}]\n{content}"
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n---\n\n".join(parts)


class BaseAgent(ABC):
    def __init__(self):
        self.llm = build_llm()

    @abstractmethod
    def generate(self, state: AgentState) -> str:
        pass


class PromptedAgent(BaseAgent):
    """Shared generation path for agents that differ only in their prompt.

    The four research agents previously carried byte-identical ``generate``
    bodies differing in one template constant and one retry hint each. One
    implementation means the retrieval context, the retry wording and the
    logging shape cannot drift apart between intents.
    """

    system_prompt: str = ""
    retry_hint: str = ""
    temperature: float = 0.1

    def __init__(self):
        self.llm = build_llm(temperature=self.temperature)

    def generate(self, state: AgentState) -> str:
        query = state["query"]
        docs = state.get("reranked_docs") or state.get("retrieved_docs", [])
        memory_ctx = state.get("memory_context", "")
        iteration = state.get("iteration", 0)

        system_content = self.system_prompt.format(
            context=format_context(docs),
            memory_context=memory_ctx or "无历史对话",
        )
        if iteration > 0 and self.retry_hint:
            system_content += f"\n\n注意：{self.retry_hint}"

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=query),
        ]

        logger.info("agent_generate", agent=type(self).__name__, iteration=iteration)
        return self.llm.invoke(messages).content
