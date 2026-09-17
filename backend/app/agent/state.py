import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict


class RetrievedDoc(TypedDict):
    doc_id: str
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]


class AgentState(TypedDict):
    # Input
    session_id: str
    query: str
    # Knowledge bases this question may draw on. Empty means every base.
    kb_ids: List[str]

    # Analysis
    intent: str  # "concept_qa" | "literature_review" | "method_compare" | "academic_writing"
    query_keywords: List[str]

    # Retrieval
    retrieved_docs: List[RetrievedDoc]
    reranked_docs: List[RetrievedDoc]

    # Memory
    memory_context: str

    # Generation
    final_response: str
    is_valid: bool
    validation_reason: str

    # Control
    iteration: int
    messages: Annotated[List[Dict[str, str]], operator.add]
    error: Optional[str]
