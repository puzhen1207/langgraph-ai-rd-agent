"""Knowledge-base administration, upload and inspection APIs.

A knowledge base is a named partition of the single vector collection. Uploads
declare the base they belong to, and every later read can be scoped to one, to
several, or — by omitting the scope — to all of them.
"""
import hashlib
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core.logging_config import get_logger
from app.core.security import require_admin_key
from app.rag.ingest import ALLOWED_EXTENSIONS, ingest_bytes, ingest_text
from app.rag.kb_registry import (
    DEFAULT_KB_ID,
    DEFAULT_KB_NAME,
    DuplicateKnowledgeBaseName,
    InvalidKnowledgeBaseName,
    KnowledgeBaseError,
    KnowledgeBaseNotFound,
    get_kb_registry,
)
from app.rag.vectorstore import get_vector_store

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])

MAX_FILE_SIZE_MB = 50
MAX_TEXT_SIZE_BYTES = 5 * 1024 * 1024


class KnowledgeBaseInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    created_at: str = ""
    is_default: bool = False
    document_count: int = Field(0, description="Distinct source files in this base")
    chunk_count: int = Field(0, description="Indexed passages in this base")


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=60)
    description: str = Field(default="", max_length=200)


class KnowledgeBaseUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=60)
    description: Optional[str] = Field(default=None, max_length=200)


class DocumentUploadResponse(BaseModel):
    file_id: str = Field(
        ...,
        description="Content-addressed id: re-uploading identical bytes yields the same value",
    )
    filename: str
    knowledge_base_id: str
    knowledge_base_name: str
    chunks_added: int
    total_docs: int
    status: str


class DocumentStatsResponse(BaseModel):
    total_documents: int
    collection_name: str
    knowledge_base_count: int
    knowledge_base_id: Optional[str] = None


class ChunkItem(BaseModel):
    id: str
    content: str
    source: str
    type: str
    kb_id: str


def _file_id(content: bytes) -> str:
    """Stable id for an upload payload.

    A per-request uuid4 made this field useless: it could not identify a
    document across requests, so nothing could reference or delete "that upload".
    Content addressing gives the same bytes the same id, which is also what the
    chunk ids already do.
    """
    return hashlib.sha256(content).hexdigest()[:16]


def _require_kb(kb_id: str) -> str:
    """Validate an explicit base id, returning its display name."""
    kb = get_kb_registry().get(kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail=f"知识库不存在：{kb_id}")
    return kb.name


def _handle_registry_error(exc: KnowledgeBaseError) -> HTTPException:
    if isinstance(exc, KnowledgeBaseNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, DuplicateKnowledgeBaseName):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, InvalidKnowledgeBaseName):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


def _info(kb, document_count: int = 0, chunk_count: int = 0) -> KnowledgeBaseInfo:
    return KnowledgeBaseInfo(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        created_at=kb.created_at,
        is_default=kb.id == DEFAULT_KB_ID,
        document_count=document_count,
        chunk_count=chunk_count,
    )


# --------------------------------------------------------------- knowledge bases


@router.get("/knowledge-bases", response_model=List[KnowledgeBaseInfo])
async def list_knowledge_bases():
    """Every base, with its passage and file counts.

    Counts are aggregated in one metadata sweep rather than a query per base, so
    the sidebar renders from a single round trip.
    """
    registry = get_kb_registry()
    registry.ensure_default()
    stats = get_vector_store().base_stats()

    infos: List[KnowledgeBaseInfo] = []
    known = set()
    for kb in registry.list_all():
        counts = stats.get(kb.id, {})
        infos.append(
            _info(kb, counts.get("documents", 0), counts.get("chunks", 0))
        )
        known.add(kb.id)

    # Chunks whose base is absent from the registry — a hand-edited registry, or
    # data restored from a backup. Surfaced so they stay visible and deletable
    # instead of sitting in the index unreachably.
    for kb_id, counts in sorted(stats.items()):
        if kb_id in known:
            continue
        infos.append(
            KnowledgeBaseInfo(
                id=kb_id,
                name=kb_id,
                description="（注册表中缺失，可能是历史数据）",
                document_count=counts["documents"],
                chunk_count=counts["chunks"],
            )
        )
    return infos


@router.post(
    "/knowledge-bases",
    response_model=KnowledgeBaseInfo,
    status_code=201,
    dependencies=[Depends(require_admin_key)],
)
async def create_knowledge_base(payload: KnowledgeBaseCreate):
    try:
        kb = get_kb_registry().create(payload.name, payload.description)
    except KnowledgeBaseError as exc:
        raise _handle_registry_error(exc) from exc
    logger.info("kb_create_requested", kb_id=kb.id, name=kb.name)
    return _info(kb)


@router.patch(
    "/knowledge-bases/{kb_id}",
    response_model=KnowledgeBaseInfo,
    dependencies=[Depends(require_admin_key)],
)
async def update_knowledge_base(kb_id: str, payload: KnowledgeBaseUpdate):
    try:
        kb = get_kb_registry().update(
            kb_id, name=payload.name, description=payload.description
        )
    except KnowledgeBaseError as exc:
        raise _handle_registry_error(exc) from exc
    stats = get_vector_store().base_stats().get(kb_id, {})
    return _info(kb, stats.get("documents", 0), stats.get("chunks", 0))


@router.delete(
    "/knowledge-bases/{kb_id}", dependencies=[Depends(require_admin_key)]
)
async def delete_knowledge_base(kb_id: str):
    """Delete a base and every passage in it."""
    if kb_id == DEFAULT_KB_ID:
        raise HTTPException(
            status_code=400,
            detail=(
                f"「{DEFAULT_KB_NAME}」不能删除：内置示例语料与早期文档都在其中，"
                "删除会让这些内容既无归属也无法再检索。可以逐个删除其中的文档。"
            ),
        )

    registry = get_kb_registry()
    kb = registry.get(kb_id)
    removed = await run_in_threadpool(get_vector_store().delete_by_kb, kb_id)

    if kb is not None:
        try:
            registry.delete(kb_id)
        except KnowledgeBaseError as exc:
            raise _handle_registry_error(exc) from exc

    logger.info("kb_delete_requested", kb_id=kb_id, chunks_removed=removed)
    return {
        "message": f"知识库{'「' + kb.name + '」' if kb else kb_id}已删除",
        "knowledge_base_id": kb_id,
        "chunks_removed": removed,
    }


@router.delete(
    "/knowledge-bases/{kb_id}/documents",
    dependencies=[Depends(require_admin_key)],
)
async def clear_knowledge_base(kb_id: str):
    """Remove every passage in one base, keeping the base itself.

    Needed because the built-in base cannot be deleted — without this, the
    sample corpus that ships with the project could never be cleared out.
    """
    _require_kb(kb_id)
    removed = await run_in_threadpool(get_vector_store().delete_by_kb, kb_id)
    logger.info("kb_cleared", kb_id=kb_id, chunks_removed=removed)
    return {"message": f"已清空该知识库的 {removed} 段索引", "chunks_removed": removed}


# ------------------------------------------------------------------------ upload


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    dependencies=[Depends(require_admin_key)],
)
async def upload_document(
    file: UploadFile = File(...),
    knowledge_base_id: str = Form(default=DEFAULT_KB_ID),
    doc_type: Optional[str] = Form(default=None),
):
    filename = file.filename or "unknown"
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}",
        )

    kb_name = _require_kb(knowledge_base_id)

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB",
        )

    try:
        added, total = await run_in_threadpool(
            ingest_bytes, content, filename, extension, knowledge_base_id
        )
        logger.info(
            "document_uploaded",
            filename=filename,
            kb_id=knowledge_base_id,
            added=added,
            total=total,
        )
        return DocumentUploadResponse(
            file_id=_file_id(content),
            filename=filename,
            knowledge_base_id=knowledge_base_id,
            knowledge_base_name=kb_name,
            chunks_added=added,
            total_docs=total,
            status="indexed" if added else "already_indexed",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("upload_failed", filename=filename, error=str(exc))
        raise HTTPException(status_code=500, detail="Document indexing failed") from exc


@router.post(
    "/upload-text",
    response_model=DocumentUploadResponse,
    dependencies=[Depends(require_admin_key)],
)
async def upload_text(
    content: str = Form(...),
    source: str = Form(default="manual_input"),
    knowledge_base_id: str = Form(default=DEFAULT_KB_ID),
    doc_type: str = Form(default="markdown"),
):
    if len(content.encode("utf-8")) > MAX_TEXT_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Text content exceeds 5MB")

    kb_name = _require_kb(knowledge_base_id)
    added, total = await run_in_threadpool(
        ingest_text, content, source, doc_type, knowledge_base_id
    )
    return DocumentUploadResponse(
        file_id=_file_id(content.encode("utf-8")),
        filename=source,
        knowledge_base_id=knowledge_base_id,
        knowledge_base_name=kb_name,
        chunks_added=added,
        total_docs=total,
        status="indexed" if added else "already_indexed",
    )


# -------------------------------------------------------------------- inspection


@router.get("/stats", response_model=DocumentStatsResponse)
async def get_stats(knowledge_base_id: Optional[str] = None):
    vector_store = get_vector_store()
    registry = get_kb_registry()
    registry.ensure_default()
    return DocumentStatsResponse(
        total_documents=vector_store.count(knowledge_base_id),
        collection_name=vector_store.collection.name,
        knowledge_base_count=len(registry.list_all()),
        knowledge_base_id=knowledge_base_id,
    )


@router.get(
    "/chunks",
    response_model=List[ChunkItem],
    dependencies=[Depends(require_admin_key)],
)
async def list_chunks(
    knowledge_base_id: Optional[str] = Query(default=None),
    limit: int = Query(default=500, ge=1, le=2000),
):
    chunks = await run_in_threadpool(
        get_vector_store().get_all_chunks, knowledge_base_id
    )
    return chunks[:limit]


@router.delete("/clear", dependencies=[Depends(require_admin_key)])
async def clear_documents():
    """Drop every passage in every base. Base names and descriptions survive.

    Kept for operations and tests; the UI removes one base at a time, which is
    what a user actually wants and does not require re-creating the bases they
    still intend to keep.
    """
    get_vector_store().delete_collection()
    return {"message": "All passages cleared; knowledge bases kept"}
