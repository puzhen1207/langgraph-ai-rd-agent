"""
AI R&D Agent - FastAPI Entry Point
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.api import chat, documents, health
from app.core.config import settings
from app.core.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def _seed_documents() -> int:
    """Index the bundled sample corpus into the default base.

    Returns the number of chunks added. The sample documents get their own named
    base rather than being mixed into whatever the user creates first, so the
    built-in corpus stays identifiable in the sidebar and in citations.
    """
    from app.rag.ingest import ingest_directory
    from app.rag.kb_registry import DEFAULT_KB_ID

    seed_dir = settings.seed_docs_path
    if seed_dir is None:
        logger.info("seed_skipped", reason="no seed directory found")
        return 0

    summary = ingest_directory(seed_dir, kb_id=DEFAULT_KB_ID)
    if summary.added_total or summary.failed:
        logger.info(
            "seed_docs_done",
            directory=str(seed_dir),
            kb_id=DEFAULT_KB_ID,
            files=len(summary.added_by_file),
            new_chunks=summary.added_total,
            failed=len(summary.failed),
        )
    else:
        logger.info("seed_docs_unchanged", directory=str(seed_dir))
    return summary.added_total


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup", app=settings.APP_NAME, version=settings.APP_VERSION)

    if not settings.ADMIN_API_KEY.strip():
        # Fail-open keeps local development frictionless, but it also means the
        # document endpoints accept uploads and a full knowledge-base wipe from
        # anyone who can reach the port.
        logger.warning(
            "admin_key_not_configured",
            note="document management endpoints are unprotected; "
            "set ADMIN_API_KEY before exposing this service",
        )

    # The built-in base must exist before anything is written to it: uploads and
    # seeds both validate their target against the registry.
    try:
        from app.rag.kb_registry import get_kb_registry

        registry = get_kb_registry()
        default_kb = registry.ensure_default()
        logger.info(
            "kb_registry_ready",
            path=str(registry.path),
            count=len(registry.list_all()),
            default_kb=default_kb.name,
        )
    except Exception as e:
        logger.error("kb_registry_failed", error=str(e))

    # Pre-warm: seed the sample corpus, then initialize the vector store and
    # lexical index so the first request is not the one paying for it.
    try:
        from app.rag.vectorstore import get_vector_store

        vector_store = get_vector_store()
        logger.info("vectorstore_ready", doc_count=vector_store.count())

        seeded = 0
        if settings.AUTO_SEED_DOCS:
            seeded = await asyncio.to_thread(_seed_documents)

        if seeded == 0 and vector_store.count() > 0:
            # ingest_directory already refreshed the index for anything it added;
            # this path covers a restart against an already-warm volume.
            from app.rag.retriever import build_bm25_index
            all_records = vector_store.get_all_records()
            build_bm25_index(all_records)
            logger.info("bm25_index_ready", doc_count=len(all_records))

        # Report which reranking path resolved. "score_sort" means passages keep
        # the fused order — a legitimate configuration, but one worth stating at
        # startup instead of leaving to be inferred from a puzzling answer.
        from app.rag.reranker import get_reranker

        reranker_mode = get_reranker().mode
        logger.info(
            "reranker_ready",
            mode=reranker_mode,
            active=reranker_mode != "score_sort",
        )
    except Exception as e:
        # Deliberately not swallowed at warning level: an embedding dimension
        # mismatch surfaces here, and this message is what makes it actionable.
        logger.error("warmup_failed", error=str(e))

    yield
    logger.info("shutdown")


app = FastAPI(
    title=settings.APP_NAME,
    description="跨领域科研问答助手：多知识库 RAG 检索、引用溯源与 LangGraph 工作流",
    version=settings.APP_VERSION,
    docs_url="/swagger",
    redoc_url="/redoc",
    lifespan=lifespan,
    swagger_ui_parameters={
        "deepLinking": True,
        "persistAuthorization": True,
        "docExpansion": "none",
    },
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics. Kept out of the OpenAPI schema: /metrics is scrape
# plumbing, not part of the client-facing contract.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# Routers
app.include_router(health.router)
app.include_router(chat.router, prefix=settings.API_V1_STR)
app.include_router(documents.router, prefix=settings.API_V1_STR)


# 美观的 API 文档（Scalar UI）
# /docs 与 /scalar 都渲染 Scalar；原 Swagger 移到 /swagger，ReDoc 保留在 /redoc
_SCALAR_HTML = """<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>AI R&D Agent · API 文档</title>
    <style>body { margin: 0; }</style>
  </head>
  <body>
    <script id="api-reference" data-url="/openapi.json"></script>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
  </body>
</html>"""


@app.get("/docs", include_in_schema=False)
@app.get("/scalar", include_in_schema=False)
async def scalar_docs() -> HTMLResponse:
    return HTMLResponse(_SCALAR_HTML)
