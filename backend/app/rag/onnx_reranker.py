"""Cross-encoder reranking through an ONNX export.

A cross-encoder reads the query and the passage *together*, so it can separate
"relevant, worded differently" from "shares vocabulary but answers something
else" — precisely what a bi-encoder cannot do, since it never sees the two at the
same time. That is what this stage buys: RRF produces a recall-optimised list,
and this is the precision pass over it.

Why ONNX rather than FlagEmbedding: BAAI ships ``bge-reranker-base`` with an
``onnx/`` folder, and ``onnxruntime`` plus ``tokenizers`` are already
dependencies, so enabling reranking costs no new install and no PyTorch.
``bge-reranker-v2-m3`` scores better but ships PyTorch weights only, so it has to
go through the FlagEmbedding path instead.
"""
import math
from pathlib import Path
from typing import List, Sequence, Set, Tuple

from app.core.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_LENGTH = 512
_BATCH_SIZE = 8
_PAD_TOKEN_ID = 1  # <pad> for the XLM-RoBERTa tokenizer this family uses


def _sigmoid(value: float) -> float:
    """Map a raw logit to 0..1, matching FlagEmbedding's ``normalize=True``.

    Written out rather than pulled from numpy because a single float does not
    justify an array round trip, and the naive form overflows for large
    negatives.
    """
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


class OnnxCrossEncoderReranker:
    """ONNX cross-encoder scorer. The graph is loaded lazily on first use."""

    def __init__(self, model_path: str, max_length: int = DEFAULT_MAX_LENGTH):
        self.model_path = str(model_path)
        self.max_length = max_length
        self._session = None
        self._tokenizer = None
        self._input_names: Set[str] = set()
        self._output_name = ""

    @staticmethod
    def name() -> str:
        return "bge_reranker_onnx"

    def _resolve_files(self) -> Tuple[Path, Path]:
        """Locate the graph and its tokenizer inside the model directory.

        They are not always side by side. BAAI's ``bge-m3`` export keeps a
        tokenizer *inside* ``onnx/``, while ``bge-reranker-base`` puts only the
        graph there and leaves ``tokenizer.json`` at the model root. Both layouts
        are accepted, so a folder downloaded straight from the Hub works without
        being rearranged by hand first.
        """
        base = Path(self.model_path)
        model_candidates = (base / "onnx" / "model.onnx", base / "model.onnx")
        tokenizer_candidates = (
            base / "onnx" / "tokenizer.json",
            base / "tokenizer.json",
        )

        model_file = next((p for p in model_candidates if p.is_file()), None)
        tokenizer_file = next((p for p in tokenizer_candidates if p.is_file()), None)

        missing = []
        if model_file is None:
            missing.append(str(model_candidates[0]))
        if tokenizer_file is None:
            missing.append(str(base / "tokenizer.json"))
        if missing:
            raise FileNotFoundError(
                f"ONNX reranker assets missing under '{self.model_path}': "
                f"{', '.join(missing)}. Expected either "
                f"<dir>/onnx/model.onnx + <dir>/onnx/tokenizer.json (bge-m3 layout) "
                f"or <dir>/onnx/model.onnx + <dir>/tokenizer.json "
                f"(bge-reranker-base layout). Point RERANKER_ONNX_PATH at such a "
                f"directory, or set USE_RERANKER=false to rank by the fused score."
            )
        return model_file, tokenizer_file

    @staticmethod
    def _load_failure_message(model_file: Path, exc: Exception) -> str:
        """Explain the one failure mode that is not self-evident.

        onnxruntime cannot open a large self-contained graph through a path
        containing non-ASCII characters. On Windows it reports only
        ``system error number 13``, which reads as a permission problem and sends
        you auditing file permissions for no reason.

        The bge-m3 export escapes this because its weights live in a separate
        ``model.onnx_data`` file loaded through a different code path — so
        "the embedding model works fine from the same folder" is a reasonable
        conclusion and a wrong one.
        """
        message = f"Failed to load ONNX reranker from {model_file}: {exc}"
        if str(model_file).isascii():
            return message
        return (
            f"{message}\n"
            f"The model path contains non-ASCII characters, which onnxruntime "
            f"cannot use for a large self-contained graph. Move the model folder "
            f"to an ASCII-only path (for example D:/models/bge-reranker-base) and "
            f"point RERANKER_ONNX_PATH at it. Note that bge-m3 is unaffected "
            f"because its weights sit in a separate model.onnx_data file."
        )

    def _load(self) -> None:
        if self._session is not None:
            return

        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_file, tokenizer_file = self._resolve_files()
        try:
            self._session = ort.InferenceSession(
                str(model_file), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise RuntimeError(self._load_failure_message(model_file, exc)) from exc
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self._tokenizer.enable_truncation(max_length=self.max_length)

        # Read the actual signature instead of assuming it: the export carries
        # token_type_ids for some checkpoints and not for others (XLM-R has no
        # segment embeddings), and passing an unexpected input fails at runtime.
        self._input_names = {item.name for item in self._session.get_inputs()}
        self._output_name = self._session.get_outputs()[0].name
        logger.info(
            "onnx_reranker_ready",
            model=str(model_file),
            max_length=self.max_length,
        )

    def score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """Score (query, passage) pairs. Higher means more relevant."""
        if not pairs:
            return []

        import numpy as np

        self._load()
        scores: List[float] = []

        for start in range(0, len(pairs), _BATCH_SIZE):
            batch = pairs[start : start + _BATCH_SIZE]
            # encode(a, b) emits `<s> a </s> </s> b </s>`, the pair layout these
            # checkpoints were trained on. Gluing the strings together by hand
            # would produce a different token stream and quietly worse scores.
            encodings = [
                self._tokenizer.encode(query or " ", passage or " ")
                for query, passage in batch
            ]
            # Padded to the longest member of this batch, not to max_length:
            # attention is quadratic, and most pairs are far shorter than 512.
            width = max(len(enc.ids) for enc in encodings)
            ids = np.array(
                [enc.ids + [_PAD_TOKEN_ID] * (width - len(enc.ids)) for enc in encodings],
                dtype=np.int64,
            )
            mask = np.array(
                [
                    enc.attention_mask + [0] * (width - len(enc.attention_mask))
                    for enc in encodings
                ],
                dtype=np.int64,
            )

            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)

            logits = self._session.run([self._output_name], feeds)[0]
            scores.extend(_sigmoid(float(row[0])) for row in logits)

        return scores
