"""Chat API with synchronous responses and real SSE token streaming."""
import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field

from app.agent.graph import get_agent_graph
from app.agent.state import AgentState
from app.core.config import settings
from app.core.logging_config import get_logger
from app.core.security import assert_session_owner, claim_session, get_client_token
from app.memory.conversation import ConversationMemory
from app.memory.session_meta import get_meta
from app.memory.session_registry import get_session_registry
from app.rag.kb_registry import DEFAULT_KB_ID, get_kb_registry

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    knowledge_base_ids: Optional[List[str]] = Field(
        default=None,
        description=(
            "Knowledge bases this question may draw on. Omit — or send an empty "
            "list — to search every base."
        ),
    )


class SourceRef(BaseModel):
    document: str = Field(..., description="File the passage came from")
    knowledge_base_id: str
    knowledge_base_name: str


class ChatResponse(BaseModel):
    session_id: str
    query: str
    intent: str
    response: str
    retrieved_count: int
    sources: List[SourceRef] = Field(
        default_factory=list,
        description="Distinct documents behind this answer, most relevant first",
    )
    iteration: int
    is_valid: bool
    validation_reason: str


def _passes_relevance_threshold(doc: Dict[str, Any]) -> bool:
    """True when the doc should be shown as a citation.

    The reranker only attaches a ``rerank_score`` field when it actually ran
    (ONNX, FlagEmbedding, or CrossEncoder). When it falls back to
    ``score_sort`` — i.e. the candidate order is whatever hybrid search
    produced — the field is absent, and we cannot honestly call a passage
    "irrelevant". Keeping everything is the right thing in that case: the
    threshold exists to remove noise from a real score, not to second-guess a
    system that already failed and warned us about it in the startup log.

    A threshold of 0 disables the filter, which is useful for comparing the
    two behaviours without redeploying.
    """
    threshold = settings.RERANK_SCORE_THRESHOLD
    if threshold <= 0:
        return True
    if "rerank_score" not in doc:
        return True
    return float(doc.get("rerank_score") or 0.0) >= threshold


def _collect_sources(docs: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """Distinct documents behind an answer, kept in relevance order.

    Order matters for a research assistant: citations are shown to the user as
    received, so the first entry should be the passage the answer leaned on most,
    which is the order the reranker already produced.

    A document is identified together with the base holding it — the same
    filename can exist in two bases and mean different things there.

    Passages whose ``rerank_score`` is below :attr:`RERANK_SCORE_THRESHOLD` are
    dropped here. They are still fed to the LLM (so the model can choose
    whether to ignore or use them); only the *displayed* citations get
    filtered. The split keeps the answer grounded while preventing the UI from
    advertising sources the model did not actually rely on.

    Returns plain dicts so the same value can be handed to ``ChatResponse``
    (which validates them into :class:`SourceRef`) and serialised straight into
    the SSE ``done`` event.
    """
    items = [doc for doc in (docs or []) if _passes_relevance_threshold(doc)]
    kb_ids = {
        (doc.get("metadata") or {}).get("kb_id") or DEFAULT_KB_ID for doc in items
    }
    names = get_kb_registry().names_for(kb_ids)

    seen = set()
    refs: List[Dict[str, str]] = []
    for doc in items:
        source = doc.get("source")
        if not source:
            continue
        kb_id = (doc.get("metadata") or {}).get("kb_id") or DEFAULT_KB_ID
        key = (kb_id, source)
        if key in seen:
            continue
        seen.add(key)
        refs.append(
            {
                "document": source,
                "knowledge_base_id": kb_id,
                "knowledge_base_name": names.get(kb_id, kb_id),
            }
        )
    return refs


def _build_initial_state(req: ChatRequest, session_id: str) -> AgentState:
    memory = ConversationMemory(session_id=session_id)
    memory.add_user_message(req.query)
    return AgentState(
        session_id=session_id,
        query=req.query,
        kb_ids=list(req.knowledge_base_ids or []),
        intent="",
        query_keywords=[],
        retrieved_docs=[],
        reranked_docs=[],
        memory_context="",
        final_response="",
        is_valid=False,
        validation_reason="",
        iteration=0,
        messages=[{"role": "user", "content": req.query}],
        error=None,
    )


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, token: Optional[str] = Depends(get_client_token)):
    """Run the graph and return a complete response."""
    session_id = req.session_id or str(uuid.uuid4())
    claim_session(session_id, token)
    logger.info("chat_request", session_id=session_id, query=req.query[:80])
    try:
        graph = get_agent_graph()
        initial_state = _build_initial_state(req, session_id)
        final_state: AgentState = await asyncio.to_thread(graph.invoke, initial_state)

        response_text = final_state.get("final_response", "")
        ConversationMemory(session_id=session_id).add_assistant_message(response_text)

        return ChatResponse(
            session_id=session_id,
            query=req.query,
            intent=final_state.get("intent", "concept_qa"),
            response=response_text,
            retrieved_count=len(final_state.get("reranked_docs", [])),
            sources=_collect_sources(final_state.get("reranked_docs")),
            iteration=final_state.get("iteration", 0),
            is_valid=final_state.get("is_valid", False),
            validation_reason=final_state.get("validation_reason", ""),
        )
    except Exception as exc:
        logger.error("chat_error", error=str(exc), session_id=session_id)
        raise HTTPException(status_code=500, detail="Agent request failed") from exc


class _SSETokenCallback(BaseCallbackHandler):
    """Bridge blocking LangChain callbacks into an asyncio event queue."""

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self.loop = loop
        self.queue = queue
        self.tokens_emitted = 0

    def _put(self, event: Dict[str, Any]) -> None:
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, event)
        except RuntimeError:
            # The client disconnected and the request loop has already closed.
            pass

    def on_llm_start(self, *args, **kwargs) -> None:
        # A retry starts a new generation. The UI clears the invalid attempt.
        self.tokens_emitted = 0
        self._put({"type": "generation_start"})

    def on_chat_model_start(self, *args, **kwargs) -> None:
        self.on_llm_start(*args, **kwargs)

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            self.tokens_emitted += 1
            self._put({"type": "token", "content": token})


@router.post("/stream")
async def chat_stream(req: ChatRequest, token: Optional[str] = Depends(get_client_token)):
    """Stream graph progress and model tokens as Server-Sent Events."""
    session_id = req.session_id or str(uuid.uuid4())
    claim_session(session_id, token)
    logger.info("chat_stream_request", session_id=session_id)

    async def event_generator() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        callback = _SSETokenCallback(loop, queue)

        def emit(event: Dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def run_graph() -> None:
            latest_response = ""
            intent = "concept_qa"
            retrieved_count = 0
            sources: List[Dict[str, str]] = []
            iteration = 0
            is_valid = False
            validation_reason = ""
            try:
                graph = get_agent_graph()
                initial_state = _build_initial_state(req, session_id)
                for chunk in graph.stream(
                    initial_state,
                    config={"callbacks": [callback]},
                ):
                    for node_name, node_output in chunk.items():
                        if node_name == "intent_router":
                            intent = node_output.get("intent", intent)
                        elif node_name == "rerank":
                            retrieved_count = len(node_output.get("reranked_docs", []))
                            sources = _collect_sources(node_output.get("reranked_docs"))
                        elif node_name == "llm_generate":
                            latest_response = node_output.get("final_response", latest_response)
                            iteration = node_output.get("iteration", iteration)
                            # Some compatible providers do not expose token
                            # callbacks. Fall back to one complete event rather
                            # than simulated character slicing and sleeps.
                            if latest_response and callback.tokens_emitted == 0:
                                emit({"type": "token", "content": latest_response})
                        elif node_name == "response_check":
                            is_valid = node_output.get("is_valid", is_valid)
                            validation_reason = node_output.get(
                                "validation_reason", validation_reason
                            )

                        progress: Dict[str, Any] = {
                            "type": "progress",
                            "node": node_name,
                        }
                        if node_name == "intent_router":
                            progress["intent"] = intent
                        if node_name == "rerank":
                            # Sent early so the UI can show which sources the
                            # answer is being built from while it still streams.
                            progress["retrieved"] = retrieved_count
                            progress["sources"] = sources
                        emit(progress)

                if latest_response:
                    ConversationMemory(session_id=session_id).add_assistant_message(
                        latest_response
                    )
                emit({
                    "type": "done",
                    "intent": intent,
                    "retrieved_count": retrieved_count,
                    "sources": sources,
                    "iteration": iteration,
                    "is_valid": is_valid,
                    "validation_reason": validation_reason,
                })
            except Exception as exc:
                logger.error("stream_error", error=str(exc), session_id=session_id)
                emit({"type": "error", "message": "Agent streaming failed"})

        producer = asyncio.create_task(asyncio.to_thread(run_graph))
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
        try:
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event["type"] in {"done", "error"}:
                    break
        finally:
            if not producer.done():
                producer.cancel()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/session/{session_id}")
async def clear_session(session_id: str, token: Optional[str] = Depends(get_client_token)):
    assert_session_owner(session_id, token)
    memory = ConversationMemory(session_id=session_id)
    memory.clear()
    return {"message": f"Session {session_id} cleared"}


@router.get("/session/{session_id}/history")
async def get_history(session_id: str, token: Optional[str] = Depends(get_client_token)):
    assert_session_owner(session_id, token)
    memory = ConversationMemory(session_id=session_id)
    messages = memory.get_messages()
    return {"session_id": session_id, "messages": messages, "count": len(messages)}


class SessionSummary(BaseModel):
    session_id: str
    title: str
    created_at: float
    updated_at: float
    message_count: int


@router.get("/sessions", response_model=List[SessionSummary])
async def list_sessions(token: Optional[str] = Depends(get_client_token)):
    """List every session the calling token owns, newest first.

    "Husk" sessions — claimed but with neither meta nor messages (data expired
    under REDIS_TTL while the token index outlived it, or claimed and never
    used) — are skipped: they would otherwise render as "新对话 / 1970/1/1"
    with zero messages.
    """
    registry = get_session_registry()
    session_ids = registry.list_for_token(token or "")
    if not session_ids:
        return []

    summaries: List[SessionSummary] = []
    for sid in session_ids:
        meta = get_meta(sid) or {}
        messages = ConversationMemory(session_id=sid).get_messages()
        if not meta and not messages:
            continue
        created = float(meta.get("created_at") or 0.0)
        updated = float(meta.get("updated_at") or created)
        summaries.append(
            SessionSummary(
                session_id=sid,
                title=meta.get("title") or "新对话",
                created_at=created,
                updated_at=updated,
                message_count=len(messages),
            )
        )
    # Newest activity first; stable tiebreaker on session id so the order is
    # deterministic when two sessions share an updated_at timestamp.
    summaries.sort(key=lambda s: (-s.updated_at, s.session_id))
    return summaries
