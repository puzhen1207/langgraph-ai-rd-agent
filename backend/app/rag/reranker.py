"""
Cross-encoder reranking for precision, with honest degradation.

Resolution order, best first:

1. ``onnx``         — a local ONNX export; no PyTorch, no download
2. ``flag``         — FlagEmbedding (the official BGE reranker package)
3. ``cross_encoder`` — sentence-transformers CrossEncoder
4. ``score_sort``   — no cross-encoder available; ranks by the fused score

The last one is deliberately named rather than hidden. Ranking by ``score`` when
``score`` is already the RRF output is a no-op, and a reranker that looks enabled
while doing nothing is worse than one that is switched off — the log says which
mode resolved, and ``BGEReranker.mode`` reports it.
"""
from functools import lru_cache
from typing import Any, Dict, List, Sequence

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class BGEReranker:
    """BGE cross-encoder reranker, over ONNX or PyTorch."""

    def __init__(self):
        self._model = None
        self._mode: str = "none"
        self._init_model()

    @property
    def mode(self) -> str:
        """Which path actually resolved. ``score_sort`` means none did."""
        return self._mode

    def _init_model(self):
        if not settings.USE_RERANKER:
            logger.info("reranker_disabled")
            return

        # 1. ONNX cross-encoder. Same runtime as the embeddings, so this needs
        #    nothing installed and nothing downloaded at request time.
        onnx_dir = settings.reranker_onnx_dir
        if onnx_dir is not None:
            try:
                from app.rag.onnx_reranker import OnnxCrossEncoderReranker

                reranker = OnnxCrossEncoderReranker(
                    str(onnx_dir), max_length=settings.RERANKER_MAX_LENGTH
                )
                # Force the graph to load now: a corrupt or incomplete export
                # should surface here with a clear reason, not on a user's query.
                reranker.score([("warmup", "warmup")])
                self._model = reranker
                self._mode = "onnx"
                logger.info("reranker_loaded", mode="onnx", model=str(onnx_dir))
                return
            except Exception as e:
                # Warning, not info: a configured-but-unusable model is a
                # mistake worth noticing, unlike a path that was never set.
                logger.warning("onnx_reranker_failed", error=str(e))

        # 2. FlagEmbedding — the official BGE reranker package.
        try:
            from FlagEmbedding import FlagReranker

            try:
                import torch

                use_fp16 = bool(torch.cuda.is_available())
            except Exception:
                use_fp16 = False
            self._model = FlagReranker(
                settings.RERANKER_MODEL,
                use_fp16=use_fp16,
            )
            self._mode = "flag"
            logger.info(
                "reranker_loaded", mode="FlagEmbedding", model=settings.RERANKER_MODEL
            )
            return
        except Exception as e:
            logger.warning("flag_reranker_failed", error=str(e))

        # 3. sentence-transformers CrossEncoder, pinned to the configured model.
        #    It used to hardcode ms-marco-MiniLM here, which meant installing
        #    sentence-transformers silently downloaded a different, English-only
        #    model and ignored RERANKER_MODEL entirely.
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(settings.RERANKER_MODEL)
            self._mode = "cross_encoder"
            logger.info(
                "reranker_loaded",
                mode="CrossEncoder",
                model=settings.RERANKER_MODEL,
            )
            return
        except Exception as e:
            logger.warning("cross_encoder_failed", error=str(e))

        logger.warning(
            "reranker_unavailable_score_fallback",
            note=(
                "no cross-encoder resolved, so passages keep the fused RRF order. "
                "Set RERANKER_ONNX_PATH to a local bge-reranker ONNX export to "
                "enable the precision pass."
            ),
        )
        self._mode = "score_sort"

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not docs:
            return []

        if self._mode == "onnx":
            pairs = [(query, doc.get("content", "")) for doc in docs]
            return self._rank_by(docs, self._model.score(pairs), top_k)

        if self._mode == "flag":
            pairs = [[query, doc.get("content", "")] for doc in docs]
            scores = self._model.compute_score(pairs, normalize=True)
            if not isinstance(scores, list):
                scores = [scores]
            return self._rank_by(docs, scores, top_k)

        if self._mode == "cross_encoder":
            pairs = [(query, doc.get("content", "")) for doc in docs]
            return self._rank_by(docs, self._model.predict(pairs), top_k)

        # No cross-encoder: keep the order the fusion already produced.
        return docs[:top_k]

    @staticmethod
    def _rank_by(
        docs: List[Dict[str, Any]], scores: Sequence[float], top_k: int
    ) -> List[Dict[str, Any]]:
        """Attach scores and return the best ``top_k``, highest first.

        Shared by all three backends: their only difference is how a score is
        produced, and three copies of a sort is how the tie-breaking and the
        score field name drift apart.
        """
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)
        ranked = sorted(docs, key=lambda d: d.get("rerank_score", 0), reverse=True)
        return ranked[:top_k]


@lru_cache(maxsize=1)
def get_reranker() -> BGEReranker:
    return BGEReranker()
