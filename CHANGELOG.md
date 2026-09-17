# Changelog

> `APP_VERSION`（`backend/app/core/config.py`）是版本号的唯一出处，这里只记录历史。

## Unreleased

### Reranking actually wires up now

- Added an ONNX cross-encoder path (`app/rag/onnx_reranker.py`). `bge-reranker-base`
  ships an ONNX export, and `onnxruntime` / `tokenizers` are already dependencies,
  so turning on the precision pass costs no new install and no PyTorch. Set
  `RERANKER_ONNX_PATH` and it takes effect on restart.
- The resolved mode is now reported rather than left implicit: startup logs
  `reranker_ready mode=... active=...`, and `BGEReranker.mode` exposes it. When
  nothing resolves, the mode is the literal string `score_sort` — which is what it
  does, and which makes clear that the stage is a no-op.
- Fixed the sentence-transformers fallback ignoring `RERANKER_MODEL`: it hardcoded
  `cross-encoder/ms-marco-MiniLM-L-6-v2`, so merely installing
  sentence-transformers downloaded a different, English-only model and reranked
  with it.
- Collapsed three near-identical ranking bodies into `_rank_by`, so the score field
  name and the tie-breaking cannot drift between backends.
- `RERANKER_ONNX_PATH` accepts either a dedicated setting or a local directory
  written into `RERANKER_MODEL` — putting the path in the "wrong" field should not
  silently disable reranking.

No model is bundled and none is downloaded at request time: reranking stays at
`score_sort` until a model is placed on disk and configured.

## v3.0.0 - 2026-09-14

### 为什么是 major

三条破坏性变更同时落地，`2.1.0` 会低估它：

- **意图体系重定位**：`doc_qa` / `code_gen` / `debug` / `arch_analyze` 换成四个科研意图，
  提示词口径完全不同——同一句话拿到的回答会变。
- **`sources` 契约变更**：从字符串数组变成对象数组（`document` / `knowledge_base_id` /
  `knowledge_base_name`），按旧结构解析的客户端会直接失败。
- **分片 ID 变更**：内容哈希现在包含 `kb_id`，已有向量库必须重建才能带上归属。

### Repositioned as a cross-domain research assistant

- Replaced the Android-specific intent taxonomy (`doc_qa` / `code_gen` / `debug` /
  `arch_analyze`) with research capabilities: `concept_qa`, `literature_review`,
  `method_compare`, `academic_writing`.
- Rewrote all four system prompts for an unspecified discipline. Answers must label
  anything not backed by a retrieved passage instead of presenting it as a finding,
  and the review prompt must not invent authors, years or venues.
- Retrieval breadth is now per-intent (24 passages for a survey, 10 for a concept
  question) rather than one global constant.

### Named knowledge bases

- The vector collection is partitioned by a `kb_id` metadata field instead of being
  one flat corpus. Bases can be created, named, renamed and deleted via
  `POST|PATCH|DELETE /api/v1/documents/knowledge-bases`, and deleting one drops its
  passages with it.
- Uploads declare a target base (`knowledge_base_id`); every read — chunk listing,
  stats, retrieval — can be scoped to one base, to several, or to all of them.
- `POST /api/v1/chat` accepts `knowledge_base_ids`. Omitting it, or sending an empty
  list, searches everything. An unknown id returns no context rather than silently
  widening to every base: the per-intent collection filter may be dropped when it
  matches nothing, but the user's chosen scope may not — which is why retrieval now
  takes a separate `fallback_filter`.
- The built-in 「默认知识库」 holds the bundled sample corpus and anything written
  before partitioning existed. It cannot be deleted, since its passages would be
  orphaned into something the UI can neither name nor remove, but it can be emptied
  via `DELETE /api/v1/documents/knowledge-bases/default/documents`.
- Content hashes now include the base, so the same file uploaded to two bases is
  indexed twice rather than the second upload being discarded as already stored.
  Pre-existing data must be re-ingested; see the README note.
- `2.x` migration: point `CHROMA_PERSIST_DIR` at a fresh directory and re-run
  `seed_docs.py`. Re-running without doing so is safe — old chunks simply stay
  attributed to the default base through the read-time fallback — but they will not
  carry the new ids.

### Citations

- `sources` is now a list of objects (`document`, `knowledge_base_id`,
  `knowledge_base_name`) instead of bare filenames, and the same filename in two
  bases counts as two citations.
- The prompt labels each passage with its base, so an answer can attribute a claim
  to a named collection rather than a bare filename.
- The `rerank` progress event carries `sources` too, so the UI can show citations
  while the answer is still streaming.

### Frontend

- Rebuilt around the research positioning: ink-on-paper palette, serif headings,
  cross-domain example prompts, and per-answer citation chips.
- Replaced the Android capability selector with four research modes.

### Fixed

- `/health` reported Redis as healthy while the in-process fallback was serving,
  because a set/get roundtrip cannot tell the two apart. The resolved backend is now
  reported and the fallback is marked degraded.
- Seeding a directory rebuilt the BM25 index even when nothing had been added.

### Removed

- Dead code: the `is_code_related` state field, the never-read `requires_rag` and
  `system_prompt_key` intent-config keys, and four byte-identical `generate()`
  implementations across the agent classes.

## v2.0.1 - 2026-07-18

- Added a Python 3.11 lightweight runtime profile for Windows/CPU development.
- Documented the supported offline embedding mode and local startup commands.

## v2.0.0 - 2026-07-18

### Security

- Removed the real-looking API key from `.env.example` and added secret-safe ignore files.
- Added optional `X-Admin-Key` protection for knowledge-base administration APIs.
- Stopped returning raw internal exception details from public chat and upload endpoints.

### Streaming and memory

- Replaced buffered, simulated SSE output with graph progress events and real LangChain token callbacks.
- Saved assistant replies for streaming sessions.
- Added generation reset events so retries replace invalid partial answers.
- Included intent, retrieval count and validation metadata in the final SSE event.

### Retrieval

- Added stable content IDs and idempotent document upserts.
- Preserved source and metadata in BM25 results.
- Added Chinese character and bigram tokenization.
- Applied code-document filtering to both vector and BM25 retrieval.
- Fused rankings by stable ID and reset BM25 when the collection is cleared.
- Disabled FP16 reranking on CPU.

### Operations and quality

- Fixed backend container health checks and health-status aggregation.
- Moved parsing, embedding and index rebuilding off the async event loop.
- Added Java source loading, unit tests and GitHub Actions CI.
- Preserved invalid response status after retries instead of reporting it as valid.

## v1.0.0

- Original project implementation, preserved in Git with secrets sanitized.
