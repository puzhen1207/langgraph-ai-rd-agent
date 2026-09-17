"""Hybrid BM25/vector retrieval with Reciprocal Rank Fusion."""
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Union

from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.logging_config import get_logger
from app.rag.vectorstore import get_vector_store

logger = get_logger(__name__)

_bm25_index: Optional[BM25Okapi] = None
_bm25_docs: List[Dict[str, Any]] = []


def _tokenize(text: str) -> List[str]:
    """Tokenize English identifiers and Chinese text without extra services.

    Single Chinese characters plus bigrams provide much better lexical recall
    than treating an entire Chinese sentence as one token.
    """
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", lowered)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.extend(sequence)
        tokens.extend(sequence[i : i + 2] for i in range(len(sequence) - 1))
    return tokens


def build_bm25_index(
    records: List[Union[str, Dict[str, Any]]],
) -> Optional[BM25Okapi]:
    """Build the lexical index while retaining IDs and source metadata."""
    global _bm25_index, _bm25_docs
    normalized: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        if isinstance(record, str):
            normalized.append({
                "doc_id": f"legacy_{index}",
                "content": record,
                "source": "unknown",
                "metadata": {},
            })
        else:
            normalized.append({
                "doc_id": record.get("doc_id", f"record_{index}"),
                "content": record.get("content", ""),
                "source": record.get("source", "unknown"),
                "metadata": record.get("metadata", {}),
            })

    tokenized_records = [
        (record, _tokenize(record["content"]))
        for record in normalized
        if record["content"].strip()
    ]
    tokenized_records = [pair for pair in tokenized_records if pair[1]]
    _bm25_docs = [record for record, _ in tokenized_records]
    if not _bm25_docs:
        _bm25_index = None
        return None

    _bm25_index = BM25Okapi([tokens for _, tokens in tokenized_records])
    logger.info("bm25_index_built", doc_count=len(_bm25_docs))
    return _bm25_index


def reset_bm25_index() -> None:
    global _bm25_index, _bm25_docs
    _bm25_index = None
    _bm25_docs = []
    logger.info("bm25_index_reset")


def get_bm25_index() -> Optional[BM25Okapi]:
    global _bm25_index
    if _bm25_index is None:
        try:
            records = get_vector_store().get_all_records()
            if records:
                build_bm25_index(records)
            else:
                logger.warning("bm25_no_docs_in_store")
        except Exception as exc:
            logger.error("bm25_build_failed", error=str(exc))
    return _bm25_index


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank + 1)


def _document_key(doc: Dict[str, Any]) -> str:
    return doc.get("doc_id") or doc.get("content", "")[:200]


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Merge rankings using stable Chroma IDs rather than text prefixes."""
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    for rank, doc in enumerate(vector_results):
        key = _document_key(doc)
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_results):
        key = _document_key(doc)
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
        doc_map.setdefault(key, doc)

    merged = []
    for key in sorted(scores, key=scores.get, reverse=True)[:top_k]:
        doc = dict(doc_map[key])
        doc["score"] = round(scores[key], 6)
        merged.append(doc)
    return merged


def _matches_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        if "$in" in expected:
            return actual in expected["$in"]
        if "$eq" in expected:
            return actual == expected["$eq"]
        if "$ne" in expected:
            return actual != expected["$ne"]
        # Unknown operator: fail closed. Silently matching would make the lexical
        # half disagree with Chroma instead of merely returning nothing.
        return False
    return actual == expected


def _matches_filter(
    metadata: Dict[str, Any], where_filter: Optional[Dict[str, Any]]
) -> bool:
    """Mirror Chroma's ``where`` semantics, including ``$and`` and ``$in``.

    Both halves of the hybrid search must honour the exact same filter,
    otherwise the two rankings describe different result sets.
    """
    if not where_filter:
        return True
    if "$and" in where_filter:
        return all(
            _matches_filter(metadata, clause) for clause in where_filter["$and"]
        )
    return all(
        _matches_value(metadata.get(key), value)
        for key, value in where_filter.items()
    )


def build_where_filter(
    collection_filter: Optional[Dict[str, Any]] = None,
    kb_ids: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Combine the per-intent filter with the user's knowledge-base scope.

    Both dimensions end up as one clause because Chroma accepts a single
    ``where``. An empty ``kb_ids`` means "every base", which is the *absence* of
    a clause — passing ``{"kb_id": {"$in": []}}`` would instead match nothing.
    """
    clauses: List[Dict[str, Any]] = []
    if collection_filter:
        clauses.append(collection_filter)
    if kb_ids:
        clauses.append({"kb_id": {"$in": list(kb_ids)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class HybridRetriever:
    def __init__(self):
        self.vs = get_vector_store()

    def hybrid_search(
        self,
        query: str,
        keywords: Optional[List[str]] = None,
        where_filter: Optional[Dict[str, Any]] = None,
        fallback_filter: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid BM25 + vector retrieval, optionally restricted by a filter.

        ``fallback_filter`` is what to retry with when the filter matches
        nothing. It exists because the two halves of the filter have different
        failure modes: a per-intent collection filter is a retrieval *hint* and
        may safely be dropped, whereas the knowledge-base scope is a user
        instruction — silently widening it would answer from bases the user
        deliberately excluded.
        """
        top_k = top_k or settings.TOP_K_RETRIEVE

        docs = self._search(query, keywords, where_filter, top_k)
        if docs or where_filter is None or fallback_filter == where_filter:
            return docs

        # A filter that matches nothing must not silently starve the prompt:
        # generation would fall back to "answer from the model alone", which is
        # exactly where hallucinations come from.
        logger.info(
            "retrieval_filter_fallback",
            from_filter=where_filter,
            to_filter=fallback_filter,
        )
        return self._search(query, keywords, fallback_filter, top_k)

    def _search(
        self,
        query: str,
        keywords: Optional[List[str]],
        where_filter: Optional[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        vector_results: List[Dict[str, Any]] = []
        try:
            vector_results = self.vs.similarity_search(
                query=query, top_k=top_k, where=where_filter
            )
            logger.info("vector_search_done", count=len(vector_results))
        except Exception as exc:
            logger.warning("vector_search_failed", error=str(exc))

        bm25_results: List[Dict[str, Any]] = []
        try:
            bm25_idx = get_bm25_index()
            if bm25_idx:
                # Extracted keywords are lexical hints, so they join the BM25
                # term stream; the vector half keeps using the raw query.
                term_stream = f"{query} {' '.join(keywords)}" if keywords else query
                scores = bm25_idx.get_scores(_tokenize(term_stream))
                ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
                for index, score in ranked:
                    if score <= 0 or index >= len(_bm25_docs):
                        continue
                    record = _bm25_docs[index]
                    metadata = record.get("metadata", {})
                    if not _matches_filter(metadata, where_filter):
                        continue
                    bm25_results.append({
                        "doc_id": record["doc_id"],
                        "content": record["content"],
                        "source": record["source"],
                        "score": float(score),
                        "metadata": metadata,
                    })
                    if len(bm25_results) >= top_k:
                        break
                logger.info("bm25_search_done", count=len(bm25_results))
        except Exception as exc:
            logger.warning("bm25_search_failed", error=str(exc))

        if vector_results and bm25_results:
            return reciprocal_rank_fusion(vector_results, bm25_results, top_k=top_k)
        return vector_results or bm25_results


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
