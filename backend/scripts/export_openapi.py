"""Export the FastAPI OpenAPI schema as a committed build artifact.

The schema is the single source of truth for the frontend's TypeScript types.
It is written to the repository root so CI can verify that the contract did not
drift, without having to boot Redis, ChromaDB or an LLM provider.

Usage (from anywhere):

    python backend/scripts/export_openapi.py [output_path]

Default output: ``<repo_root>/openapi.json``
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"

# Allow ``python backend/scripts/export_openapi.py`` without installing the
# backend as a package or exporting PYTHONPATH.
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

DEFAULT_OUTPUT = REPO_ROOT / "openapi.json"


def render_schema() -> str:
    """Build the OpenAPI document as deterministic, human-readable JSON."""
    from main import app  # noqa: WPS433 - imported late so sys.path is ready

    schema = app.openapi()
    return json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main() -> int:
    output = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUTPUT

    try:
        rendered = render_schema()
    except Exception as exc:  # pragma: no cover - surfaced to the caller
        print(f"ERROR: could not build the OpenAPI schema: {exc}", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    existing = output.read_text(encoding="utf-8") if output.exists() else None

    if existing == rendered:
        print(f"OpenAPI schema already up to date: {output}")
        return 0

    # Pin the newline so the drift check in CI (Linux) matches a Windows run.
    output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"OpenAPI schema written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
