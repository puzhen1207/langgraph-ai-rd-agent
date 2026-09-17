"""Document ingestion pipeline.

Shared by the upload API and the startup seeder so a document is parsed,
chunked and indexed exactly one way.

All functions here are blocking (embedding is CPU-bound): call them from
``run_in_threadpool`` when inside an async request handler.
"""
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

from app.core.logging_config import get_logger
from app.rag.chunker import SmartChunker
from app.rag.kb_registry import DEFAULT_KB_ID
from app.rag.loader import DocumentLoader
from app.rag.retriever import build_bm25_index
from app.rag.vectorstore import get_vector_store

logger = get_logger(__name__)

# Single definition of what the knowledge base accepts. Note that .docx is
# converted to Markdown before it reaches DocumentLoader, so it lives here and
# not in the loader's own SUPPORTED_EXTENSIONS.
ALLOWED_EXTENSIONS = {
    ".md",
    ".txt",
    ".kt",
    ".java",
    ".html",
    ".wiki",
    ".docx",
    ".pdf",
}


class IngestSummary(NamedTuple):
    """Outcome of indexing a whole directory."""

    added_by_file: Dict[str, int]
    failed: Dict[str, str]

    @property
    def added_total(self) -> int:
        return sum(self.added_by_file.values())


def _docx_to_text(content: bytes) -> str:
    """Convert a .docx payload to Markdown, keeping tables intact."""
    from docx import Document as DocxDocument
    from docx.oxml.ns import qn
    from docx.table import Table

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        doc = DocxDocument(tmp_path)
        parts = []
        for block in doc.element.body:
            tag = block.tag.split("}")[-1]
            if tag == "p":
                text = "".join(r.text for r in block.findall(".//" + qn("w:t")))
                if text.strip():
                    parts.append(text.strip())
            elif tag == "tbl":
                table = Table(block, doc)
                markdown_rows = []
                for row_index, row in enumerate(table.rows):
                    cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                    markdown_rows.append("| " + " | ".join(cells) + " |")
                    if row_index == 0:
                        markdown_rows.append("|" + "---|" * len(cells))
                if markdown_rows:
                    parts.append("\n".join(markdown_rows))
        return "\n\n".join(parts)
    finally:
        os.unlink(tmp_path)


def _chunk_bytes(content: bytes, filename: str, extension: str) -> List[Dict[str, Any]]:
    loader = DocumentLoader()
    chunker = SmartChunker()

    if extension == ".docx":
        raw_docs = loader.load_text_content(
            _docx_to_text(content), source=filename, doc_type="markdown"
        )
    else:
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            raw_docs = loader.load_file(tmp_path)
            for document in raw_docs:
                document["source"] = filename
                document.setdefault("metadata", {})["source"] = filename
        finally:
            os.unlink(tmp_path)

    if not raw_docs:
        raise ValueError(f"No readable content found in {filename}")
    return chunker.chunk_documents(raw_docs)


def rebuild_bm25_index() -> int:
    """Rebuild the lexical index from the vector store. Returns the doc count."""
    records = get_vector_store().get_all_records()
    build_bm25_index(records)
    return len(records)


def _index_chunks(
    chunks: List[Dict[str, Any]],
    kb_id: str = DEFAULT_KB_ID,
    rebuild_bm25: bool = True,
) -> Tuple[int, int]:
    """Embed chunks into a knowledge base, then refresh BM25 if asked.

    Returns ``(added, chunks_in_that_base)``.
    """
    vector_store = get_vector_store()
    added = vector_store.add_documents(chunks, kb_id=kb_id)
    if rebuild_bm25:
        rebuild_bm25_index()
    return added, vector_store.count(kb_id)


def ingest_bytes(
    content: bytes,
    filename: str,
    extension: str,
    kb_id: str = DEFAULT_KB_ID,
    rebuild_bm25: bool = True,
) -> Tuple[int, int]:
    """Parse, chunk and index one document payload into a knowledge base."""
    chunks = _chunk_bytes(content, filename, extension)
    return _index_chunks(chunks, kb_id=kb_id, rebuild_bm25=rebuild_bm25)


def ingest_text(
    content: str,
    source: str,
    doc_type: str = "markdown",
    kb_id: str = DEFAULT_KB_ID,
) -> Tuple[int, int]:
    """Index a raw text payload that never touched the filesystem."""
    loader = DocumentLoader()
    chunks = SmartChunker().chunk_documents(
        loader.load_text_content(content=content, source=source, doc_type=doc_type)
    )
    return _index_chunks(chunks, kb_id=kb_id)


def ingest_directory(
    directory: Path,
    kb_id: str = DEFAULT_KB_ID,
    extensions: Optional[Iterable[str]] = None,
) -> IngestSummary:
    """Index every supported file directly inside ``directory`` into one base.

    Non-recursive: only documents placed in the directory root are seeded, so
    scratch or reference material can live in a subfolder.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return IngestSummary({}, {})

    allowed = set(extensions or ALLOWED_EXTENSIONS)
    added_by_file: Dict[str, int] = {}
    failed: Dict[str, str] = {}

    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        try:
            added, _ = ingest_bytes(
                path.read_bytes(),
                path.name,
                path.suffix.lower(),
                kb_id=kb_id,
                rebuild_bm25=False,
            )
            added_by_file[path.name] = added
        except Exception as exc:
            logger.warning("ingest_file_failed", file=path.name, error=str(exc))
            failed[path.name] = str(exc)

    if any(added_by_file.values()):
        # One rebuild for the whole batch instead of one per file. Skipped
        # entirely when every chunk was already stored, so a restart against a
        # warm volume does not pay for a redundant index rebuild.
        rebuild_bm25_index()

    return IngestSummary(added_by_file, failed)
