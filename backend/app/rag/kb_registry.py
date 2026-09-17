"""Named knowledge bases, partitioned inside a single vector collection.

A knowledge base is a name plus the ``kb_id`` stamped onto every chunk it owns.
The collection stays singular on purpose: one embedding model, one HNSW index,
one dimension check, and querying several bases at once is a filter away instead
of a fan-out across N collections. Deleting a base is therefore a metadata
delete, not a collection drop.

The registry itself is a JSON file. It holds a handful of rows that must survive
restarts and stay readable by eye — a store of its own would be overkill, and it
keeps the "which bases exist" question answerable without touching the vector
index.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional

from pydantic import BaseModel

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Carries the bundled sample corpus, and every chunk written before knowledge
# bases existed. It cannot be deleted: those chunks would be orphaned into
# something the UI can neither name nor remove.
DEFAULT_KB_ID = "default"
DEFAULT_KB_NAME = "默认知识库"
MAX_NAME_LENGTH = 60

# Serialises read-modify-write on the JSON file. The API is the only writer, and
# a lost update here would silently drop a base.
_REGISTRY_LOCK = Lock()


class KnowledgeBase(BaseModel):
    id: str
    name: str
    description: str = ""
    created_at: str = ""


class KnowledgeBaseError(ValueError):
    """Registry rejection; the API layer maps these onto 4xx responses."""


class InvalidKnowledgeBaseName(KnowledgeBaseError):
    pass


class DuplicateKnowledgeBaseName(KnowledgeBaseError):
    pass


class KnowledgeBaseNotFound(KnowledgeBaseError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_name(raw: str) -> str:
    """Collapse whitespace so '深度  学习 ' and '深度 学习' are the same name."""
    return " ".join((raw or "").split())


class KnowledgeBaseRegistry:
    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path else Path(settings.KB_REGISTRY_PATH)

    @property
    def path(self) -> Path:
        return self._path

    # ---------------------------------------------------------------- storage

    def _read(self) -> List[KnowledgeBase]:
        if not self._path.exists():
            return []
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Treated as empty rather than fatal: a developer can fix the file by
            # hand, and refusing to start over a listing would be worse.
            logger.error("kb_registry_unreadable", path=str(self._path), error=str(exc))
            return []
        items: List[KnowledgeBase] = []
        for entry in payload.get("knowledge_bases", []):
            try:
                items.append(KnowledgeBase(**entry))
            except Exception as exc:
                logger.warning("kb_entry_skipped", entry=str(entry)[:120], error=str(exc))
        return items

    def _write(self, items: List[KnowledgeBase]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "knowledge_bases": [item.model_dump() for item in items],
        }
        # Write-then-rename: a crash mid-write leaves the previous registry
        # intact, instead of a truncated file that reads as "no knowledge bases".
        tmp = self._path.parent / f"{self._path.name}.tmp"
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, self._path)

    # ------------------------------------------------------------------ reads

    def list_all(self) -> List[KnowledgeBase]:
        return self._read()

    def get(self, kb_id: str) -> Optional[KnowledgeBase]:
        return next((item for item in self._read() if item.id == kb_id), None)

    def exists(self, kb_id: str) -> bool:
        return self.get(kb_id) is not None

    def names_for(self, kb_ids: Iterable[str]) -> Dict[str, str]:
        """Display names for the given ids, in one pass over the registry.

        Unknown ids resolve to themselves rather than being dropped: an answer
        that cites a base deleted mid-stream should stay traceable, not lose its
        citation silently.
        """
        wanted = {kb_id for kb_id in kb_ids if kb_id}
        names = {item.id: item.name for item in self._read() if item.id in wanted}
        for kb_id in wanted - names.keys():
            names[kb_id] = kb_id
        return names

    # ----------------------------------------------------------------- writes

    def ensure_default(self) -> KnowledgeBase:
        """Create the built-in base on first use. Idempotent."""
        with _REGISTRY_LOCK:
            items = self._read()
            existing = next((item for item in items if item.id == DEFAULT_KB_ID), None)
            if existing is not None:
                return existing
            kb = KnowledgeBase(
                id=DEFAULT_KB_ID,
                name=DEFAULT_KB_NAME,
                description="内置示例语料，以及早期未分类的文档",
                created_at=_now(),
            )
            items.insert(0, kb)
            self._write(items)
        logger.info("kb_default_created", kb_id=kb.id, name=kb.name)
        return kb

    def create(self, name: str, description: str = "") -> KnowledgeBase:
        clean = normalize_name(name)
        self._validate_name(clean)

        with _REGISTRY_LOCK:
            items = self._read()
            self._reject_duplicate(items, clean)
            kb = KnowledgeBase(
                id=f"kb_{uuid.uuid4().hex[:10]}",
                name=clean,
                description=(description or "").strip(),
                created_at=_now(),
            )
            items.append(kb)
            self._write(items)
        logger.info("kb_created", kb_id=kb.id, name=kb.name)
        return kb

    def update(
        self,
        kb_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> KnowledgeBase:
        with _REGISTRY_LOCK:
            items = self._read()
            index = next(
                (i for i, item in enumerate(items) if item.id == kb_id), None
            )
            if index is None:
                raise KnowledgeBaseNotFound(f"知识库不存在：{kb_id}")

            if name is not None:
                clean = normalize_name(name)
                self._validate_name(clean)
                self._reject_duplicate(items, clean, exclude_id=kb_id)
                items[index].name = clean
            if description is not None:
                items[index].description = description.strip()

            self._write(items)
            updated = items[index]
        logger.info("kb_updated", kb_id=kb_id, name=updated.name)
        return updated

    def delete(self, kb_id: str) -> KnowledgeBase:
        """Remove the registry entry. Callers drop the chunks separately."""
        with _REGISTRY_LOCK:
            items = self._read()
            target = next((item for item in items if item.id == kb_id), None)
            if target is None:
                raise KnowledgeBaseNotFound(f"知识库不存在：{kb_id}")
            self._write([item for item in items if item.id != kb_id])
        logger.info("kb_deleted", kb_id=kb_id, name=target.name)
        return target

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _validate_name(clean: str) -> None:
        if not clean:
            raise InvalidKnowledgeBaseName("知识库名称不能为空")
        if len(clean) > MAX_NAME_LENGTH:
            raise InvalidKnowledgeBaseName(
                f"知识库名称不能超过 {MAX_NAME_LENGTH} 个字符"
            )

    @staticmethod
    def _reject_duplicate(
        items: List[KnowledgeBase], name: str, exclude_id: Optional[str] = None
    ) -> None:
        key = name.casefold()
        for item in items:
            if item.id != exclude_id and item.name.casefold() == key:
                raise DuplicateKnowledgeBaseName(f"已存在同名知识库「{item.name}」")


@lru_cache(maxsize=1)
def get_kb_registry() -> KnowledgeBaseRegistry:
    return KnowledgeBaseRegistry()
