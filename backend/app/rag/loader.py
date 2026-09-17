"""
Document Loader - 支持 Markdown / Wiki / Kotlin 源码解析
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Any

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """Loads documents from various formats into a unified dict format."""

    SUPPORTED_EXTENSIONS = {".md", ".txt", ".kt", ".java", ".wiki", ".html", ".pdf"}

    def load_file(self, file_path: str) -> List[Dict[str, Any]]:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".kt":
            return self._load_kotlin(file_path)
        elif ext == ".java":
            return self._load_java(file_path)
        elif ext in (".md", ".txt", ".wiki"):
            return self._load_text(file_path)
        elif ext == ".html":
            return self._load_html(file_path)
        elif ext == ".pdf":
            return self._load_pdf(file_path)
        else:
            logger.warning("unsupported_extension", ext=ext, file=file_path)
            return []

    def load_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        docs = []
        for root, _, files in os.walk(dir_path):
            for fname in files:
                fpath = os.path.join(root, fname)
                ext = Path(fpath).suffix.lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    try:
                        loaded = self.load_file(fpath)
                        docs.extend(loaded)
                        logger.info("loaded_file", path=fpath, count=len(loaded))
                    except Exception as e:
                        logger.error("load_file_failed", path=fpath, error=str(e))
        return docs

    def _load_java(self, file_path: str) -> List[Dict[str, Any]]:
        """Load Java source as code while preserving language metadata."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()
        filename = Path(file_path).name
        return [{
            "content": content,
            "source": filename,
            "type": "code",
            "metadata": {
                "source": filename,
                "type": "code",
                "language": "java",
            },
        }]

    def load_text_content(
        self,
        content: str,
        source: str = "upload",
        doc_type: str = "markdown",
    ) -> List[Dict[str, Any]]:
        return [
            {
                "content": content,
                "source": source,
                "type": doc_type,
                "metadata": {"source": source, "type": doc_type},
            }
        ]

    def _load_text(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        fname = Path(file_path).name
        return [
            {
                "content": content,
                "source": fname,
                "type": "markdown",
                "metadata": {"source": fname, "file_path": file_path, "type": "markdown"},
            }
        ]

    def _load_kotlin(self, file_path: str) -> List[Dict[str, Any]]:
        """Kotlin source: split by top-level class/object/fun declarations."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        fname = Path(file_path).name
        # Find class/object/fun blocks
        pattern = r'(?:^|\n)((?:(?:public|private|internal|open|abstract|data|sealed|enum)\s+)*(?:class|object|fun|interface)\s+\w+[^{]*\{)'
        matches = list(re.finditer(pattern, content, re.MULTILINE))

        if not matches:
            return [
                {
                    "content": content,
                    "source": fname,
                    "type": "code",
                    "metadata": {"source": fname, "type": "code", "language": "kotlin"},
                }
            ]

        docs = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            block = content[start:end].strip()
            if len(block) > 30:
                docs.append(
                    {
                        "content": block,
                        "source": fname,
                        "type": "code",
                        "metadata": {
                            "source": fname,
                            "type": "code",
                            "language": "kotlin",
                            "block_index": i,
                        },
                    }
                )
        return docs

    def _load_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text and tables from PDF using pdfplumber.

        Each page becomes a separate document to preserve context boundaries.
        Tables are converted to Markdown table syntax.
        """
        try:
            import pdfplumber
        except ImportError:
            logger.error("pdfplumber_not_installed")
            raise ImportError("pdfplumber is required for PDF support: pip install pdfplumber")

        fname = Path(file_path).name
        docs = []

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                parts: List[str] = []

                # Extract plain text (fast path)
                text = page.extract_text() or ''
                if text.strip():
                    parts.append(text.strip())

                # Extract tables and append as Markdown (lightweight, no filter)
                try:
                    raw_tables = page.extract_tables()
                    for rows in (raw_tables or []):
                        if not rows:
                            continue
                        md_rows = []
                        for i, row in enumerate(rows):
                            cells = [str(c).strip().replace('\n', ' ') if c is not None else '' for c in row]
                            md_rows.append('| ' + ' | '.join(cells) + ' |')
                            if i == 0:
                                md_rows.append('|' + '---|' * len(cells))
                        parts.append('\n'.join(md_rows))
                except Exception:
                    pass

                page_content = '\n\n'.join(p for p in parts if p.strip())
                if page_content.strip():
                    docs.append({
                        'content': page_content,
                        'source': fname,
                        'type': 'markdown',
                        'metadata': {
                            'source': fname,
                            'type': 'pdf',
                            'page': page_num,
                            'total_pages': total_pages,
                        },
                    })

        logger.info("pdf_loaded", file=fname, pages=len(docs))
        return docs

    def _load_html(self, file_path: str) -> List[Dict[str, Any]]:
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.texts = []

            def handle_data(self, data):
                stripped = data.strip()
                if stripped:
                    self.texts.append(stripped)

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        extractor = TextExtractor()
        extractor.feed(raw)
        content = "\n".join(extractor.texts)
        fname = Path(file_path).name
        return [
            {
                "content": content,
                "source": fname,
                "type": "html",
                "metadata": {"source": fname, "type": "html"},
            }
        ]
