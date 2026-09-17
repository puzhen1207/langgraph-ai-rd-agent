"""Health Check & System Info API"""
import time
from typing import Dict, Optional

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging_config import get_logger
from app.memory.redis_store import get_redis_store
from app.rag.vectorstore import get_vector_store

logger = get_logger(__name__)
router = APIRouter(tags=["Health"])

_start_time = time.time()


class ServiceStatus(BaseModel):
    """Per-dependency probe result.

    A loose ``dict`` would publish an empty schema, leaving clients with no
    contract at all. Declaring the fields keeps the health payload typed while
    ``response_model_exclude_none`` keeps the wire format unchanged.
    """

    status: str
    doc_count: Optional[int] = None
    error: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key_set: Optional[bool] = None
    enabled: Optional[bool] = None
    backend: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    version: str
    services: Dict[str, ServiceStatus]


@router.get("/health", response_model=HealthResponse, response_model_exclude_none=True)
async def health_check():
    uptime = time.time() - _start_time
    services: Dict[str, ServiceStatus] = {}

    # Check ChromaDB
    try:
        vs = get_vector_store()
        doc_count = vs.count()
        services["chromadb"] = ServiceStatus(status="ok", doc_count=doc_count)
    except Exception as e:
        services["chromadb"] = ServiceStatus(status="error", error=str(e))

    # Check Redis. The roundtrip alone proves nothing: the in-memory fallback
    # answers it just as happily as a real server, which is how this endpoint
    # used to report "ok" while the logs said "Timeout connecting to server".
    # The resolved backend is therefore reported rather than inferred.
    try:
        store = get_redis_store()
        store.set("__health__", "1", ex=10)
        reachable = store.get("__health__") == "1"
        if store.backend != "redis":
            services["redis"] = ServiceStatus(
                status="degraded",
                backend=store.backend,
                error="Redis unreachable; using the in-process fallback, so "
                "sessions are per-process and are lost on restart",
            )
        else:
            services["redis"] = ServiceStatus(
                status="ok" if reachable else "degraded",
                backend=store.backend,
            )
    except Exception as e:
        services["redis"] = ServiceStatus(status="error", error=str(e))

    # LLM config
    key = (settings.LLM_API_KEY or "").strip()
    dummy_keys = {"", "sk-placeholder", "your-api-key-here", "placeholder", "changeme"}
    key_is_configured = key not in dummy_keys and not key.startswith("sk-placeholder")
    services["llm"] = ServiceStatus(
        status="ok" if key_is_configured else "degraded",
        model=settings.LLM_MODEL,
        base_url=settings.LLM_API_BASE,
        api_key_set=key_is_configured,
    )

    services["admin_auth"] = ServiceStatus(
        status="ok" if settings.ADMIN_API_KEY.strip() else "degraded",
        enabled=bool(settings.ADMIN_API_KEY.strip()),
    )

    overall = "ok" if all(s.status == "ok" for s in services.values()) else "degraded"

    return HealthResponse(
        status=overall,
        uptime_seconds=round(uptime, 2),
        version=settings.APP_VERSION,
        services=services,
    )


_INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>AI 研发助手 · AI R&D Agent</title>
    <style>
      * { margin: 0; padding: 0; box-sizing: border-box; }
      body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
        min-height: 100vh;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #312e81 100%);
        color: #e2e8f0;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 40px 20px;
      }
      .container { max-width: 760px; width: 100%; text-align: center; }
      h1 { font-size: 40px; font-weight: 700; margin-bottom: 12px; }
      .sub { font-size: 16px; color: #94a3b8; margin-bottom: 20px; }
      .badge {
        display: inline-block; padding: 4px 14px; border-radius: 999px;
        background: rgba(99, 102, 241, .2); border: 1px solid rgba(99, 102, 241, .5);
        color: #a5b4fc; font-size: 13px; margin-bottom: 32px;
      }
      .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }
      .card {
        display: block; padding: 22px 16px; border-radius: 14px;
        background: rgba(255, 255, 255, .06); border: 1px solid rgba(255, 255, 255, .12);
        text-decoration: none; color: #e2e8f0; transition: .2s;
      }
      .card:hover { background: rgba(255, 255, 255, .12); transform: translateY(-3px); border-color: rgba(129, 140, 248, .6); }
      .card .icon { font-size: 26px; display: block; margin-bottom: 8px; }
      .card .name { font-size: 15px; font-weight: 600; }
      .card .desc { font-size: 12px; color: #94a3b8; margin-top: 4px; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>🤖 AI 研发助手</h1>
      <p class="sub">面向 Android 客户端研发场景的企业级 AI Agent 系统</p>
      <span class="badge">__APP_VERSION__</span>
      <div class="grid">
        <a class="card" href="/docs"><span class="icon">📖</span><span class="name">API 文档</span><span class="desc">Scalar · 美观交互</span></a>
        <a class="card" href="/swagger"><span class="icon">🧪</span><span class="name">Swagger UI</span><span class="desc">经典调试界面</span></a>
        <a class="card" href="/redoc"><span class="icon">📄</span><span class="name">ReDoc</span><span class="desc">简洁只读文档</span></a>
        <a class="card" href="/health"><span class="icon">❤️</span><span class="name">健康检查</span><span class="desc">服务状态</span></a>
        <a class="card" href="http://localhost:5173"><span class="icon">🌐</span><span class="name">前端界面</span><span class="desc">聊天应用入口</span></a>
      </div>
    </div>
  </body>
</html>"""

# The page is a plain string because its CSS is full of braces, so the version
# is injected by token instead of turning the whole document into an f-string.
_INDEX_HTML = _INDEX_HTML.replace("__APP_VERSION__", f"v{settings.APP_VERSION}")


@router.get("/")
async def root() -> HTMLResponse:
    return HTMLResponse(_INDEX_HTML)
