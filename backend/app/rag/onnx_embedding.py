"""BGE-M3 embeddings through the ONNX export bundled inside the model folder.

The ONNX graph already contains BGE-M3's dense head — CLS pooling followed by
L2 normalisation — so this module only has to tokenise and run inference. That
is verified against the graph itself rather than assumed: ``sentence_embedding``
matches a normalised CLS pooling of ``token_embeddings`` to within 1.5e-08
(float32 rounding).

Why ONNX instead of sentence-transformers: sentence-transformers pulls in
PyTorch, a ~2.5 GB CUDA-enabled wheel by default on Windows, to run inference
that this service only ever performs on CPU. ``onnxruntime`` and ``tokenizers``
are already present as chromadb dependencies, so this path adds nothing to the
install and is faster on CPU. ``EMBEDDING_MODE=local`` still selects the
sentence-transformers implementation for anyone who prefers it.
"""
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.core.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_LENGTH = 1024
_BATCH_SIZE = 16
_PAD_TOKEN_ID = 1  # <pad> for the XLM-RoBERTa tokenizer BGE-M3 is built on


class OnnxBGEEmbeddingFunction:
    """chromadb-compatible embedding function backed by onnxruntime.

    The model is loaded lazily, so constructing this object is free and a
    misconfigured path only fails when embeddings are actually requested.
    """

    def __init__(self, model_path: str, max_length: int = DEFAULT_MAX_LENGTH):
        self.model_path = str(model_path)
        self.max_length = max_length
        self._session = None
        self._tokenizer = None

    @staticmethod
    def name() -> str:
        return "bge_m3_onnx"

    def get_config(self) -> Dict[str, Any]:
        return {"model_path": self.model_path, "max_length": self.max_length}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "OnnxBGEEmbeddingFunction":
        return OnnxBGEEmbeddingFunction(
            model_path=config.get("model_path", ""),
            max_length=int(config.get("max_length", DEFAULT_MAX_LENGTH)),
        )

    def _resolve_files(self) -> Tuple[Path, Path]:
        """Locate the ONNX graph and tokenizer inside the model directory."""
        base = Path(self.model_path)
        model_file = base / "onnx" / "model.onnx"
        tokenizer_file = base / "onnx" / "tokenizer.json"

        missing = [str(p) for p in (model_file, tokenizer_file) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"ONNX BGE-M3 assets missing under '{self.model_path}': "
                f"{', '.join(missing)}. Point EMBEDDING_MODEL at a model directory "
                f"that contains an 'onnx' folder, or set EMBEDDING_MODE=offline to "
                f"run without semantic search."
            )
        return model_file, tokenizer_file

    def _load(self) -> None:
        if self._session is not None:
            return

        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_file, tokenizer_file = self._resolve_files()
        self._session = ort.InferenceSession(
            str(model_file), providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self._tokenizer.enable_truncation(max_length=self.max_length)
        logger.info(
            "onnx_embedding_ready",
            model=str(model_file),
            max_length=self.max_length,
        )

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        """Embed a batch of texts. ``input`` keeps chromadb's required name."""
        if not input:
            return []

        import numpy as np

        self._load()
        vectors: List[List[float]] = []

        for start in range(0, len(input), _BATCH_SIZE):
            batch = [text if text and text.strip() else " " for text in input[start : start + _BATCH_SIZE]]
            encodings = [self._tokenizer.encode(text) for text in batch]

            # Pad to the longest sequence in this batch rather than to
            # max_length: attention cost is quadratic, so padding every short
            # query out to 1024 tokens would be pure waste.
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

            sentence_embeddings = self._session.run(
                ["sentence_embedding"],
                {"input_ids": ids, "attention_mask": mask},
            )[0]
            vectors.extend(sentence_embeddings.tolist())

        return vectors
