"""
Node 7: Response Check
Validates the generated response for quality and completeness.
Decides whether to retry or finalize.
"""
import re
from typing import Dict, Any, Optional

from app.agent.nodes.intent_router import DEFAULT_INTENT
from app.agent.state import AgentState
from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

REFUSAL_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"我不知道",
        r"我无法",
        r"抱歉，我没有",
        r"无法回答",
        r"i don't know",
        r"i cannot",
        r"i'm not sure",
        r"as an ai",
    )
]

# A refusal is a property of the whole answer, not of one clause inside it.
# "如果超时时间设置过短，我无法保证分片请求一定成功，因此建议…" is hedging inside
# an answer, not a refusal — matching the phrase anywhere in the text triggered
# pointless retries that replaced good answers with worse ones.
#
# Length cannot separate the two: measured samples put a genuine refusal at 45
# chars and a hedge at 39. Position does separate them — a refusal OPENS its
# sentence, a hedge sits behind a conditional clause.
REFUSAL_MAX_LENGTH = 200        # a long answer is an answer, not a refusal
REFUSAL_OPENING_WINDOW = 120    # refusals come first, not buried mid-text
MAX_REFUSAL_PREFIX = 2          # "这个我无法回答" still counts

_SENTENCE_BOUNDARY = re.compile(r"[。！？!?\n]+|(?<=\.)\s+")
_COURTESY_PREFIX = re.compile(r"^\s*(很?抱歉|不好意思|sorry)[，,。!！:：\s]*", re.IGNORECASE)

MIN_RESPONSE_LENGTH = 50

# A comparison answer that never compares anything is not answering the
# question. Deliberately loose — any single marker is enough — because a strict
# structural rule here previously caused pointless retries that replaced good
# answers with worse ones.
COMPARISON_MARKERS = ("对比", "相比", "区别", "差异", "优缺点", "优劣", "|")


def _opens_with_refusal(sentence: str) -> Optional[str]:
    """Return the refusal pattern this sentence opens with, if any."""
    stripped = _COURTESY_PREFIX.sub("", sentence.strip()).lower()
    for pattern in REFUSAL_PATTERNS:
        match = pattern.search(stripped)
        if match and match.start() <= MAX_REFUSAL_PREFIX:
            return pattern.pattern
    return None


def _refusal_pattern(response: str) -> Optional[str]:
    """Return a matching refusal pattern, or None if this is not a refusal.

    Known limitation: a hedge that opens its own sentence with "我无法" (e.g.
    "我无法确认这个配置是否正确") is indistinguishable from a refusal by lexical
    rules alone, and is deliberately treated as one.
    """
    stripped = response.strip()
    if len(stripped) > REFUSAL_MAX_LENGTH:
        return None

    opening = stripped[:REFUSAL_OPENING_WINDOW]
    for sentence in _SENTENCE_BOUNDARY.split(opening):
        found = _opens_with_refusal(sentence)
        if found:
            return found
    return None


def is_valid_response(response: str, intent: str) -> tuple[bool, str]:
    if not response or len(response.strip()) < MIN_RESPONSE_LENGTH:
        return False, "Response too short"

    refusal = _refusal_pattern(response)
    if refusal:
        return False, f"Refusal pattern detected: {refusal}"

    if intent == "method_compare" and not any(
        marker in response for marker in COMPARISON_MARKERS
    ):
        return False, "Comparison response compares nothing"

    return True, "OK"


def response_check_node(state: AgentState) -> Dict[str, Any]:
    response = state.get("final_response", "")
    intent = state.get("intent", DEFAULT_INTENT)
    iteration = state.get("iteration", 0)

    is_valid, reason = is_valid_response(response, intent)

    if not is_valid and iteration < settings.MAX_RETRY_ITERATIONS:
        logger.warning("response_invalid_retry", reason=reason, iteration=iteration)
        return {"is_valid": False, "validation_reason": reason}

    if not is_valid:
        logger.warning("response_invalid_max_retries", reason=reason)

    logger.info("response_check_done", is_valid=is_valid, reason=reason)
    # Preserve the actual result after max retries. Treating an invalid answer
    # as valid makes quality metrics and API consumers misleading.
    return {"is_valid": is_valid, "validation_reason": reason}


def should_retry(state: AgentState) -> str:
    is_valid = state.get("is_valid", True)
    iteration = state.get("iteration", 0)
    if not is_valid and iteration < settings.MAX_RETRY_ITERATIONS:
        return "retry"
    return "end"
