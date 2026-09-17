"""
Node 1: Query Analyze
Extracts keywords and routes the question to a research intent.
"""
import re
from typing import Dict, Any, List

from app.agent.state import AgentState
from app.core.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_INTENT = "concept_qa"

# Phrase-level markers rather than single words: a multi-character Chinese
# phrase is far less likely to fire on an unrelated question than a fragment
# like "架构" or "代码" used to.
INTENT_PATTERNS: Dict[str, tuple] = {
    "literature_review": (
        "研究现状", "文献综述", "综述", "有哪些工作", "相关研究", "研究进展",
        "前沿", "发展趋势", "研究脉络", "state of the art", "survey",
        "related work", "literature",
    ),
    "method_compare": (
        "对比", "比较", "区别", "差异", "哪个更好", "哪个更适合", "如何选择",
        "优缺点", "优劣", "选型", "相比", " versus ", " vs ", "compare",
        "trade-off", "tradeoff",
    ),
    "academic_writing": (
        "润色", "改写", "翻译", "摘要", "审稿", "投稿", "帮我写", "写一段",
        "起个标题", "措辞", "致谢", "参考文献格式", "polish", "rewrite",
        "abstract", "camera-ready", "cover letter", "reviewer",
    ),
    DEFAULT_INTENT: (
        "是什么", "什么是", "定义", "概念", "原理", "机制", "为什么",
        "解释", "含义", "作用", "如何理解", "what is", "how does", "mechanism",
    ),
}

# Intents that need a decisive signal before they take over from the default.
SPECIALISED_INTENTS = ("literature_review", "method_compare", "academic_writing")


def extract_keywords(query: str) -> List[str]:
    query_lower = query.lower()
    words = re.findall(r'[\w\u4e00-\u9fff]+', query_lower)
    return list(set(words))


def _score(query_lower: str, patterns: tuple) -> int:
    """Count how many distinct markers appear. Presence, not frequency."""
    return sum(1 for pattern in patterns if pattern in query_lower)


def detect_intent(query: str) -> str:
    """Route a research question to one of the specialised agents.

    Ties and empty signals resolve to ``concept_qa``: it is the general-purpose
    answer shape, and its markers are the least specific, so letting it win
    keeps a plain question from being pushed into a narrow format.
    """
    query_lower = query.lower()
    scores = {name: _score(query_lower, patterns) for name, patterns in INTENT_PATTERNS.items()}

    best_specialised = max(scores[name] for name in SPECIALISED_INTENTS)
    if best_specialised > 0 and best_specialised > scores[DEFAULT_INTENT]:
        for name in SPECIALISED_INTENTS:
            if scores[name] == best_specialised:
                return name

    return DEFAULT_INTENT


def query_analyze_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    logger.info("query_analyze", query=query[:100])

    keywords = extract_keywords(query)
    intent = detect_intent(query)

    logger.info("query_analyzed", intent=intent, keywords=keywords[:5])

    return {
        "query_keywords": keywords,
        "intent": intent,
        "iteration": 0,
        "error": None,
    }
