"""
ChromaDB vector store with BGE-M3 or OpenAI embeddings.
"""
import hashlib
import math
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.logging_config import get_logger
from app.rag.kb_registry import DEFAULT_KB_ID

logger = get_logger(__name__)

_EMBED_DIM = 384


class _OfflineHashEmbeddingFunction:
    """Hash-based embedding for offline/testing use.

    Projects each token into a fixed-dim float vector via signed hashing,
    then L2-normalises. Deterministic but NOT semantically meaningful.
    Replace with a real embedding (local BGE-M3 or OpenAI) for production.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "offline_hash"

    def get_config(self) -> Dict[str, Any]:
        return {"dim": _EMBED_DIM}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "_OfflineHashEmbeddingFunction":
        return _OfflineHashEmbeddingFunction()

    def __call__(self, input: List[str]) -> List[List[float]]:
        result = []
        for text in input:
            vec = [0.0] * _EMBED_DIM
            lowered = text.lower()
            tokens = re.findall(r"[a-z0-9_]+", lowered)
            for sequence in re.findall(r"[\u4e00-\u9fff]+", lowered):
                tokens.extend(sequence)
                tokens.extend(
                    sequence[i : i + 2] for i in range(max(0, len(sequence) - 1))
                )
            for token in tokens:
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                idx = h % _EMBED_DIM
                sign = 1.0 if (h >> 8) & 1 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            result.append([v / norm for v in vec])
        return result


_DUMMY_KEYS = {"", "sk-placeholder", "your-api-key-here", "placeholder", "sk-xxx", "changeme"}


def _has_real_api_key() -> bool:
    key = settings.LLM_API_KEY or ""
    return bool(key) and key not in _DUMMY_KEYS and not key.startswith("sk-placeholder")


def _offline_hash(reason: str) -> "_OfflineHashEmbeddingFunction":
    """Last-resort embedder. Always logged loudly — retrieval quality drops."""
    logger.warning("embedding_mode", mode="offline_hash", note=reason)
    return _OfflineHashEmbeddingFunction()


def _build_embedding_function():
    # 1. ONNX — local BGE-M3 without pulling in PyTorch. Preferred local mode.
    if settings.EMBEDDING_MODE == "onnx":
        try:
            from app.rag.onnx_embedding import OnnxBGEEmbeddingFunction

            embedder = OnnxBGEEmbeddingFunction(
                model_path=settings.EMBEDDING_MODEL,
                max_length=settings.EMBEDDING_MAX_LENGTH,
            )
            # Force the graph to load now: a wrong path or a corrupt export then
            # degrades at startup with a clear reason, instead of silently
            # failing on the first user query.
            embedder(["warmup"])
            logger.info("embedding_mode", mode="onnx", model=settings.EMBEDDING_MODEL)
            return embedder
        except Exception as e:
            logger.warning("onnx_embedding_failed_fallback_hash", error=str(e))
            return _offline_hash(
                "Semantic search is DISABLED. Set EMBEDDING_MODEL to a local BGE-M3 "
                "directory containing an 'onnx' folder, e.g. "
                "EMBEDDING_MODEL=D:/models/bge-m3"
            )

    # 2. Local SentenceTransformer — the requirements.txt default
    if settings.EMBEDDING_MODE == "local":
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            embedder = SentenceTransformerEmbeddingFunction(
                model_name=settings.EMBEDDING_MODEL,
                device="cpu",
            )
            # Logged only after construction succeeds: chromadb imports
            # sentence-transformers lazily inside its constructor, so logging
            # first made the startup banner claim BGE-M3 while the hash embedder
            # was the one actually serving queries.
            logger.info("embedding_mode", mode="local", model=settings.EMBEDDING_MODEL)
            return embedder
        except Exception as e:
            logger.warning("local_embedding_failed_fallback_hash", error=str(e))
            # Fall straight to hash — do NOT try OpenAI with a non-OpenAI base URL
            return _offline_hash(
                "Semantic search is DISABLED. Install sentence-transformers "
                "(pip install -r requirements.txt), or switch to "
                "EMBEDDING_MODE=onnx which needs no PyTorch."
            )

    # 3. OpenAI-compatible embeddings — only when EMBEDDING_MODE=openai is explicit
    if settings.EMBEDDING_MODE == "openai" and _has_real_api_key():
        try:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
            logger.info("embedding_mode", mode="openai")
            return OpenAIEmbeddingFunction(
                api_key=settings.LLM_API_KEY,
                model_name=settings.OPENAI_EMBEDDING_MODEL,
                api_base=settings.LLM_API_BASE,
            )
        except Exception as e:
            logger.warning("openai_embedding_failed_fallback_hash", error=str(e))

    # 4. Offline hash-based embedding — always works, no network/model needed
    return _offline_hash(
        "Set EMBEDDING_MODE=onnx with a local model directory for semantic search"
    )


def _kb_of(metadata: Optional[Dict[str, Any]]) -> str:
    """The knowledge base a chunk belongs to.

    Chunks written before knowledge bases existed carry no ``kb_id``; they are
    attributed to the built-in base rather than being reported as unattached,
    which would make them invisible in the UI while still occupying the index.
    """
    return (metadata or {}).get("kb_id") or DEFAULT_KB_ID


class VectorStore:
    def __init__(self):
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection = None
        self._ef = None

    def _ensure_client(self) -> None:
        if self._ef is None:
            self._ef = _build_embedding_function()
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

    def _embedding_dimension(self) -> Optional[int]:
        """Probe the active embedder for the width of the vectors it produces."""
        try:
            probe = self._ef(["dimension probe"])
            return len(probe[0])
        except Exception as exc:
            logger.warning("embedding_probe_failed", error=str(exc))
            return None

    def _validate_embedding_dimension(self, collection) -> None:
        """Refuse to use a collection that the current embedder cannot serve.

        A silent mismatch is the worst outcome: with ``EMBEDDING_MODE=local`` any
        import failure of sentence-transformers quietly downgrades to the 384-dim
        offline hash embedder, and querying a 1024-dim BGE-M3 collection then
        raises an opaque Chroma error that never mentions the missing dependency.
        """
        try:
            stored = collection.get(limit=1, include=["embeddings"]).get("embeddings")
        except Exception as exc:
            logger.warning("stored_dimension_probe_failed", error=str(exc))
            return
        if stored is None or len(stored) == 0:
            return

        stored_dim = len(stored[0])
        current_dim = self._embedding_dimension()
        if current_dim is None or current_dim == stored_dim:
            return

        raise RuntimeError(
            f"Embedding dimension mismatch: collection "
            f"'{settings.CHROMA_COLLECTION}' stores {stored_dim}-dim vectors but the "
            f"active embedder (EMBEDDING_MODE={settings.EMBEDDING_MODE}, "
            f"EMBEDDING_MODEL={settings.EMBEDDING_MODEL}) produces "
            f"{current_dim}-dim vectors. The collection was indexed by a different "
            f"embedding model and cannot be queried with this one — an earlier run "
            f"may also have fallen back to the 384-dim offline hash embedder. "
            f"Either restore the previous embedding configuration, or point "
            f"CHROMA_PERSIST_DIR at a fresh directory and re-ingest: chunks are "
            f"content-addressed, so seeding rebuilds them cheaply."
        )

    def _init(self):
        if self._client is not None and self._collection is not None:
            return
        self._ensure_client()

        # Validated before assignment on purpose: a mismatch must leave the store
        # unusable so every later call reports the same actionable error instead
        # of failing once and then proceeding with a broken collection.
        collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._validate_embedding_dimension(collection)
        self._collection = collection
        logger.info(
            "vectorstore_initialized",
            collection=settings.CHROMA_COLLECTION,
            count=collection.count(),
        )

    @property
    def collection(self):
        self._init()
        return self._collection

    def add_documents(
        self, chunks: List[Dict[str, Any]], kb_id: str = DEFAULT_KB_ID
    ) -> int:
        self._init()
        if not chunks:
            return 0

        # Stable IDs make repeated uploads idempotent instead of silently
        # duplicating every chunk. The knowledge base is part of the key on
        # purpose: hashing source+content alone would make the same file
        # uploaded to two bases collide, and the second upload would be
        # discarded as "already stored" while belonging nowhere.
        unique_chunks: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            chunk_id = hashlib.sha256(
                f"{kb_id}\0{chunk.get('source', 'unknown')}\0{chunk['content']}".encode(
                    "utf-8"
                )
            ).hexdigest()
            unique_chunks.setdefault(chunk_id, chunk)

        # Only new chunks reach Chroma: upserting an ID that is already stored
        # re-runs the embedding model for no benefit, which is what makes it
        # affordable for the startup seeder to walk the same directory on every
        # boot.
        existing = set(
            self.collection.get(ids=list(unique_chunks), include=["metadatas"]).get(
                "ids", []
            )
        )
        new_ids = [chunk_id for chunk_id in unique_chunks if chunk_id not in existing]
        if not new_ids:
            logger.info("documents_upserted", new_count=0, total_processed=0, kb_id=kb_id)
            return 0

        documents = [unique_chunks[chunk_id]["content"] for chunk_id in new_ids]
        metadatas = [
            {
                "source": unique_chunks[chunk_id].get("source", "unknown"),
                "type": unique_chunks[chunk_id].get("type", "text"),
                **{
                    k: str(v)
                    for k, v in unique_chunks[chunk_id].get("metadata", {}).items()
                },
                # Written last: a chunk cannot declare a different base from the
                # one it is being stored into.
                "kb_id": kb_id,
            }
            for chunk_id in new_ids
        ]

        # Batch insert in smaller batches to avoid timeout on large PDFs
        batch_size = 100
        for i in range(0, len(new_ids), batch_size):
            self.collection.upsert(
                ids=new_ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )
        logger.info(
            "documents_upserted",
            new_count=len(new_ids),
            total_processed=len(unique_chunks),
            kb_id=kb_id,
        )
        return len(new_ids)

    def similarity_search(
        self,
        query: str,
        top_k: int = 10,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        self._init()
        kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(top_k, max(self.collection.count(), 1)),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        docs = []
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - dist  # cosine distance -> similarity
            docs.append(
                {
                    "doc_id": doc_id,
                    "content": doc,
                    "source": meta.get("source", "unknown"),
                    "score": round(score, 4),
                    "metadata": meta,
                }
            )
        return docs

    def get_all_documents(self) -> List[str]:
        """Return all document texts for BM25 index building."""
        self._init()
        result = self.collection.get(include=["documents", "metadatas"])
        return result.get("documents", [])

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Return stable IDs, text and metadata for lexical indexing."""
        self._init()
        result = self.collection.get(include=["documents", "metadatas"])
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        return [
            {
                "doc_id": doc_id,
                "content": content,
                "source": (metadata or {}).get("source", "unknown"),
                "metadata": metadata or {},
            }
            for doc_id, content, metadata in zip(ids, docs, metas)
        ]

    def get_all_chunks(self, kb_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return chunks with id, content, source, type and base for UI display."""
        self._init()
        kwargs: Dict[str, Any] = {"include": ["documents", "metadatas"]}
        if kb_id:
            kwargs["where"] = {"kb_id": kb_id}
        result = self.collection.get(**kwargs)
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        chunks = []
        for cid, doc, meta in zip(ids, docs, metas or [{}] * len(docs)):
            chunks.append({
                "id": cid,
                "content": doc,
                "source": (meta or {}).get("source", "unknown"),
                "type": (meta or {}).get("type", "text"),
                "kb_id": _kb_of(meta),
            })
        return chunks

    def count(self, kb_id: Optional[str] = None) -> int:
        """Chunk count, either across every base or within one."""
        self._init()
        if not kb_id:
            return self.collection.count()
        result = self.collection.get(where={"kb_id": kb_id}, include=[])
        return len(result.get("ids", []))

    def base_stats(self) -> Dict[str, Dict[str, int]]:
        """Chunk and distinct-file counts per knowledge base.

        Aggregated in one metadata sweep rather than a query per base, so the
        sidebar can render the whole list from a single round trip.
        """
        self._init()
        result = self.collection.get(include=["metadatas"])
        stats: Dict[str, Dict[str, Any]] = {}
        for meta in result.get("metadatas") or []:
            entry = stats.setdefault(_kb_of(meta), {"chunks": 0, "files": set()})
            entry["chunks"] += 1
            entry["files"].add((meta or {}).get("source", "unknown"))
        return {
            kb_id: {"chunks": entry["chunks"], "documents": len(entry["files"])}
            for kb_id, entry in stats.items()
        }

    def delete_by_kb(self, kb_id: str) -> int:
        """Drop every chunk owned by a knowledge base. Returns the count removed.

        The lexical index is invalidated rather than rebuilt: ``get_bm25_index``
        rebuilds lazily on the next query, so a delete stays cheap and a burst of
        deletes does not rebuild the index once per call.
        """
        self._init()
        where = {"kb_id": kb_id}
        try:
            ids = self.collection.get(where=where, include=[]).get("ids", [])
        except Exception as exc:
            logger.warning("kb_chunk_lookup_failed", kb_id=kb_id, error=str(exc))
            return 0
        if not ids:
            return 0

        self.collection.delete(where=where)
        from app.rag.retriever import reset_bm25_index
        reset_bm25_index()
        logger.info("kb_chunks_deleted", kb_id=kb_id, removed=len(ids))
        return len(ids)

    def delete_collection(self):
        """Drop the collection, deliberately bypassing the embedder sanity check.

        This is the documented recovery path for a dimension mismatch, so it has
        to work in exactly the situation where ``_init`` refuses to.
        """
        self._ensure_client()
        try:
            self._client.delete_collection(settings.CHROMA_COLLECTION)
        except Exception as exc:
            # Clearing an already-empty store is not an error worth failing on.
            logger.warning("collection_delete_skipped", error=str(exc))
        self._collection = None
        # Avoid serving deleted content from the process-local BM25 cache.
        from app.rag.retriever import reset_bm25_index
        reset_bm25_index()
        logger.info("collection_deleted", collection=settings.CHROMA_COLLECTION)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore()
