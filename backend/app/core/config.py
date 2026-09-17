from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings

# backend/app/core/config.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    APP_NAME: str = "AI 科研助手"
    # 3.0.0: the positioning change breaks three contracts at once — the intent
    # taxonomy, the `sources` shape (strings -> objects), and the chunk id scheme
    # (kb_id is now part of the content hash, so existing vectors must be
    # rebuilt). A minor bump would understate it.
    APP_VERSION: str = "3.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False

    # LLM
    LLM_API_KEY: str = "sk-placeholder"
    LLM_API_BASE: str = "https://api.deepseek.com/v1"
    LLM_MODEL: str = "deepseek-chat"

    # Embedding
    #   "onnx"    - local BGE-M3 through onnxruntime; no PyTorch dependency
    #   "local"   - local BGE-M3 through sentence-transformers
    #   "openai"  - OpenAI-compatible /embeddings endpoint
    #   "offline" - deterministic hash vectors; semantic search is disabled
    EMBEDDING_MODE: str = "local"
    # Either a HuggingFace repo id or a local model directory. For "onnx" this
    # must be a directory containing an "onnx/model.onnx" export.
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_MAX_LENGTH: int = 1024
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Reranker
    USE_RERANKER: bool = True
    # HuggingFace repo id, used by the FlagEmbedding / sentence-transformers paths.
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    # Local directory holding an ONNX cross-encoder export. Preferred path when
    # present: it needs neither PyTorch nor a download, and reuses the same
    # runtime as the embeddings. Expected layout (as BAAI ships for
    # bge-reranker-base): <dir>/onnx/model.onnx + <dir>/onnx/tokenizer.json.
    RERANKER_ONNX_PATH: str = ""
    RERANKER_MAX_LENGTH: int = 512
    # Reranker scores are calibrated 0..1 (sigmoid of the logit, normalised the
    # same way FlagEmbedding does it). Below this value a passage is "shares
    # vocabulary but answers something else" — strong enough a match to get into
    # the candidate list, weak enough that quoting it as a citation is
    # misleading. Set to 0 to disable the filter entirely.
    RERANK_SCORE_THRESHOLD: float = 0.5

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    CHROMA_COLLECTION: str = "rd_agent_docs"

    # Knowledge bases. One collection, partitioned by a `kb_id` metadata field;
    # this file holds only the names and descriptions, which is why it can stay
    # a small JSON document instead of another store to operate.
    KB_REGISTRY_PATH: str = "./data/knowledge_bases.json"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    # Conversation history lifespan in seconds (default 7 days).
    REDIS_TTL: int = 604800

    # RAG
    TOP_K_RETRIEVE: int = 10
    TOP_K_RERANK: int = 5
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # Agent
    MAX_RETRY_ITERATIONS: int = 2

    # Optional admin protection. When set, document-management endpoints
    # require a matching X-Admin-Key header.
    ADMIN_API_KEY: str = ""

    # Knowledge base seeding. When enabled, every supported file in the seed
    # directory is indexed at startup, so a fresh clone can answer questions
    # about the bundled sample documents instead of retrieving nothing.
    # Re-running is cheap: chunks are content-addressed and already-stored ones
    # are skipped before the embedding model is invoked.
    AUTO_SEED_DOCS: bool = True
    SEED_DOCS_DIR: str = ""  # empty -> /app/docs (Docker) or <repo>/docs

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    @property
    def seed_docs_path(self) -> Optional[Path]:
        """First existing seed directory, or None when there is nothing to seed."""
        candidates = []
        if self.SEED_DOCS_DIR.strip():
            candidates.append(Path(self.SEED_DOCS_DIR))
        candidates.append(Path("/app/docs"))  # docker-compose read-only mount
        candidates.append(REPO_ROOT / "docs")  # local checkout
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return None

    @property
    def reranker_onnx_dir(self) -> Optional[Path]:
        """Local directory holding an ONNX cross-encoder, or None if there is none.

        Accepts either setting. ``RERANKER_MODEL`` is a repo id elsewhere in the
        code, but writing a *local directory* there is the natural thing to do,
        and silently disabling reranking because the path landed in the other
        field would be a bad trade for saving one lookup.
        """
        for candidate in (self.RERANKER_ONNX_PATH, self.RERANKER_MODEL):
            text = (candidate or "").strip()
            if not text:
                continue
            path = Path(text)
            if (path / "onnx" / "model.onnx").is_file():
                return path
        return None

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
