"""Index a directory of documents into a named knowledge base.

Usage (from anywhere):

    python backend/scripts/seed_docs.py [directory] [--kb <id-or-name>]

Defaults to the same directory the application seeds at startup:
``SEED_DOCS_DIR`` if set, else ``/app/docs`` (the docker-compose mount), else
``<repo>/docs``. ``--kb`` accepts an existing base id or name, or a new name, in
which case the base is created. Without ``--kb`` the built-in default base is
used, matching what startup seeding does.

Re-running is safe and cheap — chunks are content-addressed, so anything already
stored is skipped before the embedding model runs.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"

# Allow ``python backend/scripts/seed_docs.py`` without installing the backend
# as a package or exporting PYTHONPATH.
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _resolve_kb(spec):
    """Resolve --kb to a base, creating it when the value is a new name."""
    from app.rag.kb_registry import KnowledgeBaseError, get_kb_registry

    registry = get_kb_registry()
    if not spec:
        return registry.ensure_default()

    for kb in registry.list_all():
        if spec in (kb.id, kb.name):
            return kb

    try:
        created = registry.create(spec, "由 seed_docs.py 创建")
    except KnowledgeBaseError as exc:
        print(f"Cannot create knowledge base '{spec}': {exc}", file=sys.stderr)
        return None
    print(f"Created knowledge base '{created.name}' ({created.id})")
    return created


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        help="要索引的目录，默认为项目的 docs/ 目录",
    )
    parser.add_argument(
        "--kb",
        default=None,
        help="目标知识库的 id 或名称；不存在则按名称新建。默认写入内置默认知识库",
    )
    args = parser.parse_args()

    from app.core.config import settings
    from app.rag.ingest import ingest_directory
    from app.rag.vectorstore import get_vector_store

    directory = (
        Path(args.directory).resolve() if args.directory else settings.seed_docs_path
    )

    if directory is None or not directory.is_dir():
        print("No seed directory found. Pass one explicitly:")
        print("  python backend/scripts/seed_docs.py <directory> [--kb <name>]")
        return 1

    kb = _resolve_kb(args.kb)
    if kb is None:
        return 1

    summary = ingest_directory(directory, kb_id=kb.id)

    for name, added in sorted(summary.added_by_file.items()):
        print(f"  {name}: {added} new chunk(s)")
    for name, error in sorted(summary.failed.items()):
        print(f"  {name}: FAILED - {error}", file=sys.stderr)

    vector_store = get_vector_store()
    print()
    print(f"Directory      : {directory}")
    print(f"Knowledge base : {kb.name} ({kb.id})")
    print(f"Files indexed  : {len(summary.added_by_file)}")
    print(f"New chunks     : {summary.added_total}")
    print(f"Chunks in base : {vector_store.count(kb.id)}")
    print(f"Chunks total   : {vector_store.count()}")

    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
