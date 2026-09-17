"""
Kotlin/Markdown-aware document chunker.
- Markdown: split on headings (##, ###) to preserve semantic sections
- Kotlin: split on class/fun boundaries (done in loader), then size-based
- General: recursive character splitter with overlap
"""
import re
from typing import Any, Dict, List

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class SmartChunker:
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def chunk_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        for doc in docs:
            doc_type = doc.get("type", "text")
            content = doc.get("content", "")
            if not content.strip():
                continue

            if doc_type == "markdown":
                raw_chunks = self._chunk_markdown(content)
            elif doc_type == "code":
                raw_chunks = self._chunk_code(content)
            else:
                raw_chunks = self._chunk_text(content)

            for i, chunk_text in enumerate(raw_chunks):
                if chunk_text.strip():
                    chunks.append(
                        {
                            "content": chunk_text.strip(),
                            "source": doc.get("source", "unknown"),
                            "type": doc_type,
                            "metadata": {
                                **doc.get("metadata", {}),
                                "chunk_index": i,
                                "chunk_total": len(raw_chunks),
                            },
                        }
                    )
        logger.info("chunking_done", input_docs=len(docs), output_chunks=len(chunks))
        return chunks

    def _chunk_markdown(self, text: str) -> List[str]:
        """Split on ## headings, then size-limit each section.
        Tables (lines starting with |) are kept intact as a single chunk."""
        sections = re.split(r'\n(?=#{1,3} )', text)
        chunks = []
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                chunks.extend(self._chunk_markdown_section(section))
        return chunks

    def _chunk_markdown_section(self, text: str) -> List[str]:
        """Split a large section while keeping Markdown tables intact."""
        lines = text.split('\n')
        chunks: List[str] = []
        buffer: List[str] = []
        in_table = False

        for line in lines:
            is_table_line = bool(re.match(r'^\s*\|', line)) or bool(re.match(r'^\s*\|?[-:]+\|', line))

            if is_table_line:
                in_table = True
                buffer.append(line)
            else:
                if in_table:
                    # Table just ended — flush table as its own chunk
                    table_text = '\n'.join(buffer)
                    if len(table_text) <= self.chunk_size * 2:
                        chunks.append(table_text)
                    else:
                        # Oversized table: keep as-is (better than breaking mid-row)
                        chunks.append(table_text)
                    buffer = []
                    in_table = False

                # Regular line — accumulate and split at chunk_size boundary
                buffer.append(line)
                current = '\n'.join(buffer)
                if len(current) > self.chunk_size:
                    chunks.extend(self._chunk_text(current))
                    buffer = []

        if buffer:
            remaining = '\n'.join(buffer)
            if remaining.strip():
                if in_table or len(remaining) <= self.chunk_size:
                    chunks.append(remaining)
                else:
                    chunks.extend(self._chunk_text(remaining))
        return chunks

    def _chunk_code(self, text: str) -> List[str]:
        """Code blocks: split by blank lines if too large, preserve structure."""
        if len(text) <= self.chunk_size * 2:
            return [text]
        lines = text.split("\n")
        chunks, current, current_len = [], [], 0
        for line in lines:
            if current_len + len(line) > self.chunk_size and current:
                chunks.append("\n".join(current))
                # Overlap: keep last N lines
                overlap_start = max(0, len(current) - 5)
                current = current[overlap_start:]
                current_len = sum(len(l) for l in current)
            current.append(line)
            current_len += len(line)
        if current:
            chunks.append("\n".join(current))
        return chunks

    # Sentence-end characters in priority order (searched via rfind in window)
    _SENT_ENDS = ['。\n', '！\n', '？\n', '\n\n', '。', '！', '？',
                  '…', '.\n', '!\n', '?\n', '. ', '! ', '? ', '\n']

    def _find_break(self, text: str, start: int, end: int) -> int:
        """Return the best break position within [start, end), or -1."""
        for sep in self._SENT_ENDS:
            pos = text.rfind(sep, start, end)
            if pos != -1 and pos > start:
                return pos + len(sep)   # break AFTER the separator
        # Last resort: break at a space so we don't cut mid-word
        pos = text.rfind(' ', start, end)
        if pos != -1 and pos > start:
            return pos + 1
        return -1

    def _chunk_text(self, text: str) -> List[str]:
        """Sentence-aware splitter supporting Chinese and English."""
        if len(text) <= self.chunk_size:
            return [text]
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                bp = self._find_break(text, start, end)
                if bp != -1:
                    end = bp
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            # Ensure start always moves forward to prevent infinite loop
            next_start = end - self.chunk_overlap
            start = next_start if next_start > start else end
        return chunks
