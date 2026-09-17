# AI 科研助手 · LangGraph 跨领域研究问答

> 当前版本：**v3.0.0**。历史版本以 Git 标签保存：`v1.0.0`（原型）、`v2.0.0` / `v2.0.1`（工程化加固）。
> v3 是把项目从「Android 研发助手」重定位为跨领域科研助手的破坏性变更。

> 基于 RAG 的跨领域科研问答助手。上传你的文献与资料，助手**只依据它们作答**，
> 并标注每条结论的来源。不预设学科——领域完全由你的语料决定。

## 本次重点改进（重定位为科研助手）

- **多知识库**：可按领域创建并命名任意多个知识库，上传时指定归属，提问时勾选检索范围，
  回答会标出每处引用来自哪个知识库。
- 意图体系从「Android 研发」换成通用科研能力：概念与原理 / 文献综述 / 方法对比 / 学术写作。
- 四个系统提示词全部重写为跨领域口径，去掉了所有具体领域的默认假设。
- 回答携带**引用来源**（含所属知识库、去重、按相关度排序），检索一完成就通过 SSE 下发，便于边生成边显示。
- 检索宽度按意图调整（综述 24 段 / 概念问答 10 段），不再是全局固定值。
- 前端整体重做：墨与纸配色、衬线标题、跨领域示例问题、知识库面板与来源芯片。
- 死代码清理：`is_code_related` 状态字段、`requires_rag` 与 `system_prompt_key` 死配置、
  四个 Agent 里逐字重复的 `generate()` 实现。

一条重要边界：这是**文献问答**助手，不是写作代笔。`academic_writing` 只做润色、翻译、
摘要凝练与审稿回复，且要求保留原文事实与限定条件。

完整版本记录见 [`CHANGELOG.md`](CHANGELOG.md)，安全部署要求见 [`SECURITY.md`](SECURITY.md)。

> 从 v1 升级并沿用旧 Chroma 数据时，请删除 `CHROMA_PERSIST_DIR` 后重新导入：
> 分片 ID 现在包含所属知识库，旧分片需要重建才能带上归属。


## 架构总览

```
┌────────────────────────────────────────────────────────────┐
│  React Frontend  (Vite + TailwindCSS)                      │
│  SSE 流式聊天 · 文档上传 · 研究能力选择 · 实时进度           │
└──────────────────────┬─────────────────────────────────────┘
                       │ HTTP / SSE
┌──────────────────────▼─────────────────────────────────────┐
│  FastAPI Backend                                           │
│  /api/v1/chat  /api/v1/documents  /health  /metrics        │
└──────────────────────┬─────────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────────┐
│  LangGraph 状态机                                          │
│  query_analyze → intent_router → rag_retrieve → rerank     │
│                → memory_inject → llm_generate → response_check │
└──────┬───────────────────────────────────────┬─────────────┘
       │                                       │
┌──────▼──────┐                       ┌───────▼────────────┐
│  RAG System │                       │  Memory System      │
│ ChromaDB    │                       │  Redis + InMemory   │
│ BGE-M3 向量 │                       │  Window + Summary   │
│ BM25 混合   │                       └────────────────────┘
│ 交叉重排    │
└─────────────┘
```

## 核心研究能力

| 意图 | 能力 | 回答结构 | 触发特征 |
|------|------|---------|---------|
| `concept_qa` | 概念与原理问答（默认） | 结论 → 为什么 / 如何起作用 → 标注依据 | 是什么、为什么、机制、原理 |
| `literature_review` | 文献综述与研究现状 | 脉络 / 路线 / 共识与分歧 / 未决问题 / 覆盖缺口 | 研究现状、进展、综述 |
| `method_compare` | 方法对比与选型 | 对比表格 → 逐项依据 → 有条件的选型建议 | 对比、区别、优缺点、如何选择 |
| `academic_writing` | 学术写作与润色 | 先给修改稿 → 再列修改点与理由 | 润色、翻译、摘要、审稿回复 |

命中不了任何特征的普通提问会落到 `concept_qa`，不会被强行塞进某种固定格式。

## 技术栈

**Backend**
- `LangGraph 0.0.51` - 状态机工作流。代码用的是 `set_entry_point` / `add_conditional_edges`
  等 0.0.x API，与 `requirements.txt` 的精确锁版本一致；升级到 0.1+ 需按新的 `StateGraph` API 迁移。
- `LangChain 0.2.5` + `langchain-openai 0.1.13` - LLM 集成
- `ChromaDB` - 向量数据库（持久化）
- `BGE-M3` - 本地 Embedding 模型（1024 维，ONNX 路径无需 PyTorch）
- 交叉编码器重排（可选）- 支持本地 ONNX / FlagEmbedding / sentence-transformers；
  未配置时按 RRF 融合分排序，启动日志会写明实际生效的模式
- `rank-bm25` - BM25 稀疏检索
- `Redis` - 会话记忆缓存（含内存 fallback）
- `FastAPI` + `uvicorn` - 异步 Web 框架

**Frontend**
- `React 18` + `TypeScript` + `Vite`
- `TailwindCSS` - 原子化样式
- `react-markdown` + `react-syntax-highlighter` - Markdown & 代码高亮
- SSE 流式输出

## 快速启动

### 1. 准备环境变量

```bash
cp backend/.env.example backend/.env
# 编辑 .env，填入 LLM_API_KEY（DeepSeek 或 OpenAI）
```

### 2. 本地开发（无 Docker）

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Windows/CPU 环境如不需要下载 BGE 模型，可使用 Python 3.11 的轻量模式：

```bash
python -m venv .venv
.venv\\Scripts\\python -m pip install -r requirements-lite.txt
set EMBEDDING_MODE=offline
set USE_RERANKER=false
.venv\\Scripts\\python -m uvicorn main:app --port 8000
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
# 访问 http://localhost:5173
```

**前后端类型契约（自动生成）**

前端的接口类型全部派生自后端 OpenAPI schema，不要手写。后端接口有改动时执行：

```bash
python backend/scripts/export_openapi.py   # 后端代码 -> openapi.json
cd frontend && npm run api:types           # openapi.json -> src/types/api.generated.ts
```

`openapi.json` 与 `src/types/api.generated.ts` 都要提交。CI 会重新生成并比对，
与代码不一致时直接失败——避免出现"后端改了字段名、前端编译通过但运行时静默 undefined"。

### 3. Docker Compose（推荐生产）

```bash
# 启动所有服务（Backend + Frontend + Redis）
docker compose up -d

# 查看日志
docker compose logs -f backend

# 访问
# 前端: http://localhost
# API文档: http://localhost:8000/docs
# 健康检查: http://localhost:8000/health
# 指标: http://localhost:8000/metrics
```

## 知识库与文档

每个知识库是一个命名独立的领域集合：上传时指定目标库，提问时勾选要检索的库，
回答下方标出引用了哪些文件、以及它们分别来自哪个库。

### 方式一：前端（推荐）

在左侧「知识库」面板：

1. 点「新建知识库」，输入领域名（如「深度学习文献」「临床指南」）。名称唯一，重名会被直接拒绝。
2. 在「添加文献」里选好目标库，再上传文件。支持 `.md` `.txt` `.pdf` `.docx` `.html` `.wiki` `.kt` `.java`。
3. 用每行前的勾选框决定检索范围。只勾「深度学习文献」时，其他库的文档不会被引用，
   也不会出现在引用列表里。
4. 回答下方会列出文件名与所属库，例如 `attention_notes.md · 深度学习文献`。

### 方式二：API

```bash
# 建库
curl -X POST http://localhost:8000/api/v1/documents/knowledge-bases \
  -H "Content-Type: application/json" \
  -d '{"name": "深度学习文献"}'

# 上传到该库（knowledge_base_id 取自上面的响应）
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "knowledge_base_id=kb_xxxxxxxxxx" \
  -F "file=@docs/transformer_attention.md"

# 只在这个库里提问
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "自注意力机制解决了什么问题", "knowledge_base_ids": ["kb_xxxxxxxxxx"]}'
```

`knowledge_base_ids` 省略或传空数组表示检索全部知识库。传入不存在的 id 不会报错，
而是检索不到内容——回答会明确说明知识库中没有相关依据，不会被悄悄放宽成全库检索。

### 方式三：启动时自动导入（默认开启）

服务启动时会扫描 `docs/` 目录（Docker 下是 `/app/docs`，已由 compose 只读挂载），
把里面所有受支持的文件索引进**默认知识库**。这是为了让全新克隆的仓库开箱即可问答，
而不是检索到 0 条文档。

```bash
python backend/scripts/seed_docs.py                                   # 导入内置示例语料
python backend/scripts/seed_docs.py /path/to/docs --kb 深度学习文献    # 导入指定库（不存在则新建）
```

只扫描目录**根层**、不递归，所以参考资料可以放在子目录里而不被索引。
分片 ID 是内容哈希（含所属知识库），重复执行只会跳过已存在的分片，不会重复调用嵌入模型。
同一个文件上传到两个库会被各自索引——这是有意的：同名文件在不同领域里含义不同。
可用 `AUTO_SEED_DOCS=false` 关闭启动导入。

### 附带的示例语料

`docs/` 里随仓库附带的几篇文档用于演示跨领域检索，都是概览性整理：

| 文件 | 领域 |
|---|---|
| `transformer_attention.md` | 深度学习 |
| `crispr_gene_editing.md` | 生物医学 |
| `monetary_policy_transmission.md` | 经济学 |
| `compose_guide.md` / `crash_analysis.md` | 软件工程 |

它们只是**演示数据**，不代表助手的能力边界——换成你自己的文献即可，助手不预设学科。

## 配置说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_API_KEY` | - | **必填**，DeepSeek/OpenAI API Key |
| `LLM_API_BASE` | DeepSeek | API 基础 URL |
| `LLM_MODEL` | deepseek-chat | 模型名称 |
| `EMBEDDING_MODE` | local | `local`(BGE-M3) 或 `openai` |
| `USE_RERANKER` | true | 是否启用交叉编码器重排 |
| `RERANKER_ONNX_PATH` | 空 | 本地 ONNX 重排模型目录；填了即启用精排（最省依赖的方式） |
| `RERANKER_MODEL` | BAAI/bge-reranker-v2-m3 | FlagEmbedding / sentence-transformers 路径用的模型 |
| `TOP_K_RETRIEVE` | 10 | 检索返回数量 |
| `TOP_K_RERANK` | 5 | 重排后保留数量 |
| `ADMIN_API_KEY` | 空 | 设置后，知识库增删、文档上传、预览和清空接口要求 `X-Admin-Key` |
| `AUTO_SEED_DOCS` | true | 启动时索引 `docs/` 目录到默认知识库 |
| `SEED_DOCS_DIR` | 空 | 覆盖默认的种子目录 |
| `KB_REGISTRY_PATH` | `./data/knowledge_bases.json` | 知识库名称与描述的存储位置 |

### 关于嵌入模式

| `EMBEDDING_MODE` | 实现 | 依赖 | 说明 |
|---|---|---|---|
| `onnx` | BGE-M3 via onnxruntime | `onnxruntime` + `tokenizers` | 不需 PyTorch，CPU 推理更快 |
| `local` | BGE-M3 via sentence-transformers | `sentence-transformers` + torch（数 GB） | `requirements.txt` 的默认路径 |
| `openai` | OpenAI 兼容 `/embeddings` | 网络 + `LLM_API_KEY` | 省本地资源 |
| `offline` | 离线哈希向量（384 维） | 无 | 兜底，**语义检索不可用** |

`EMBEDDING_MODEL` 既可以是 HF 仓库名，也可以是本地模型目录。走 `onnx` 时指向一个
包含 `onnx/model.onnx` 的目录即可，模型不会联网下载。

**为什么提供 `onnx` 这条路**：`sentence-transformers` 会连带安装 PyTorch，Windows 上
默认拉的是 CUDA 版本（约 2.5 GB），而本服务的嵌入推理始终跑在 CPU 上。
BGE-M3 的模型目录本身就带一份 ONNX 导出，且该导出**已包含稠密头**
（CLS pooling + L2 归一化）——已验证 `sentence_embedding` 与 `token_embeddings`
的归一化 CLS 池化在 float32 精度内一致（最大误差 1.5e-08）。
`onnxruntime` 与 `tokenizers` 本来就随 chromadb 安装，因此这条路径零新增体积。

任一路径失败都会**降级**到离线哈希嵌入并在日志中明确写出当前真正生效的模式：

```
embedding_mode  mode=offline_hash  note=Semantic search is DISABLED...
```

若已入库的向量维度与当前嵌入器不一致（例如从 `offline` 切到 `onnx`，384 → 1024），
服务会**拒绝使用该集合**并给出明确报错，而不是让 Chroma 抛出难以定位的底层异常。
恢复方式：改回原嵌入配置，或把 `CHROMA_PERSIST_DIR` 指向新目录后重新导入
（分片是内容寻址的，重新 seed 即可重建）。

### 关于重排

重排是**可选的精度补充**：混合检索（BM25 + 向量 + RRF）负责把候选捞全，
交叉编码器再对每个候选与问题做一次联合打分，把真正回答问题的段落排到前面。
这也是双编码器做不到的事——它从不把问题和段落放在一起看。

解析顺序（`app/rag/reranker.py`，前一个不可用则尝试下一个）：

| 模式 | 依赖 | 说明 |
|---|---|---|
| `onnx` | `onnxruntime` + `tokenizers`（已随 chromadb 安装） | 本地 ONNX 交叉编码器，**零新增依赖、不需 PyTorch** |
| `flag` | `FlagEmbedding` + PyTorch | 官方 BGE 重排包 |
| `cross_encoder` | `sentence-transformers` + PyTorch | 通用 CrossEncoder |
| `score_sort` | 无 | 没有可用的交叉编码器，保持 RRF 融合顺序 |

**启用方式**（最省事的一条）：下载 `BAAI/bge-reranker-base` 的 `onnx/` 子目录
（约 1.06 GB，官方自带 ONNX 导出），放到任意目录后指定路径：

```bash
RERANKER_ONNX_PATH=D:/models/bge-reranker-base
```

重启后启动日志会写明实际生效的模式：

```
reranker_ready  mode=onnx         active=True     ← 精排已生效
reranker_ready  mode=score_sort   active=False    ← 未配置，按融合分排序
```

⚠️ 注意 `score_sort` 的含义：此时重排节点按 `score` 再排一次，而 `score`
**就是 RRF 的输出**，所以这一步等价于 no-op。这不是 bug，但也不能对外说
"用了交叉编码器精排"——启动日志和 `BGEReranker.mode` 会如实反映。

`bge-reranker-v2-m3` 效果更好，但官方只提供 PyTorch 权重、没有 ONNX 导出，
需要走 FlagEmbedding 路径并额外安装 PyTorch。

### 生产部署注意

`ADMIN_API_KEY` 默认为空，此时文档管理接口（上传 / 片段预览 / **清空知识库**）
不校验任何凭据。默认配置是为了本地开发方便，对外暴露前请务必设置该变量——
启动日志会在未配置时输出 `admin_key_not_configured` 警告。

## LangGraph 工作流

```
用户问题
    │
    ▼
query_analyze      ← 提取关键词、检测意图
    │
    ▼
intent_router      ← 路由到 concept_qa / literature_review / method_compare / academic_writing
    │
    ▼
rag_retrieve       ← BM25 + Vector 混合检索（RRF 融合）
    │
    ▼
rerank             ← 交叉编码器精排（未配置模型时保持融合顺序）
    │
    ▼
memory_inject      ← 注入 Window Memory + Summary Memory
    │
    ▼
llm_generate       ← 调用专用 Agent（DeepSeek/GPT）
    │
    ▼
response_check     ← 质量校验（长度 / 拒识 / 对比结构）
    │
   ┌┴────────┐
   │   重试   │ ← 最多 MAX_RETRY_ITERATIONS 次
   └─────────┘
    │
    ▼
   输出
```

## API 文档

启动后访问 `http://localhost:8000/docs` 查看完整 Swagger 文档。

主要接口：
- `POST /api/v1/chat` / `POST /api/v1/chat/stream` - 同步问答 / SSE 流式问答
- `GET` `POST /api/v1/documents/knowledge-bases` - 列出 / 新建知识库
- `PATCH` `DELETE /api/v1/documents/knowledge-bases/{id}` - 重命名 / 删除知识库
- `DELETE /api/v1/documents/knowledge-bases/{id}/documents` - 清空该库内容（保留库本身）
- `POST /api/v1/documents/upload` - 上传文献到指定知识库
- `GET /api/v1/documents/stats` - 分片统计
- `GET /api/v1/chat/session/{id}/history` - 历史记录
- `DELETE /api/v1/chat/session/{id}` - 清除会话
- `GET /health` - 健康检查
- `GET /metrics` - Prometheus 指标

`POST /api/v1/chat` 的 `sources` 记录了本次回答依据的文档及**各自所属的知识库**，已按相关度排序：

```json
{
  "intent": "method_compare",
  "retrieved_count": 8,
  "sources": [
    {
      "document": "transformer_attention.md",
      "knowledge_base_id": "default",
      "knowledge_base_name": "默认知识库"
    },
    {
      "document": "attention_notes.md",
      "knowledge_base_id": "kb_79f3a3f0e3",
      "knowledge_base_name": "深度学习文献"
    }
  ],
  "is_valid": true
}
```

同一个文件名出现在两个库里时是两条不同的引用——它们本就指向不同的内容。

流式接口的 `rerank` 进度事件也会带上 `sources`，因此前端可以在答案还在生成时就先显示引用来源。

### 会话隔离

会话按浏览器匿名身份隔离。客户端首次生成一个 UUID 并通过 `X-Client-Token`
请求头发送，后端在首次对话时把 `session_id` 绑定到该 token：

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-Client-Token: $(uuidgen)" \
  -d '{"query": "什么是注意力机制"}' -i
```

`history` 与 `DELETE` 接口会校验归属，token 不匹配返回 `403`。
不带该请求头的调用仍可创建新会话，但无法访问任何已被绑定的会话。

## 项目结构

```
langgraph-ai-rd-agent/
├── backend/
│   ├── app/
│   │   ├── agent/
│   │   │   ├── graph.py          # LangGraph 状态机主图
│   │   │   ├── state.py          # AgentState TypedDict
│   │   │   ├── nodes/            # 7个节点实现
│   │   │   └── agents/           # 4个专用 Agent
│   │   ├── rag/
│   │   │   ├── loader.py         # MD / PDF / DOCX / 代码 加载器
│   │   │   ├── chunker.py        # 智能分块（标题感知）
│   │   │   ├── onnx_embedding.py # ONNX BGE-M3 嵌入（无需 PyTorch）
│   │   │   ├── vectorstore.py    # ChromaDB，按 kb_id 逻辑分区
│   │   │   ├── kb_registry.py    # 知识库注册表（名称 / 描述 / 默认库）
│   │   │   ├── retriever.py      # BM25+Vector 混合检索 + 库范围过滤
│   │   │   ├── reranker.py       # 重排（未启用时按融合分排序）
│   │   │   └── ingest.py         # 统一导入流水线（上传与启动导入共用）
│   │   ├── memory/
│   │   │   ├── conversation.py   # Window + Summary Memory
│   │   │   ├── session_registry.py # 会话归属（匿名 token 绑定）
│   │   │   └── redis_store.py    # Redis（含内存 fallback）
│   │   ├── prompt/
│   │   │   └── templates.py      # 限定式 Prompt 模板
│   │   ├── api/
│   │   │   ├── chat.py           # 聊天 API（同步/SSE，含知识库范围与引用）
│   │   │   ├── documents.py      # 知识库 CRUD + 上传 + 片段预览
│   │   │   └── health.py         # 健康检查
│   │   └── core/
│   │       ├── config.py         # 配置管理
│   │       ├── security.py       # Admin Key + 会话归属校验
│   │       └── logging_config.py # structlog 日志
│   ├── scripts/
│   │   ├── export_openapi.py     # 导出 openapi.json（类型契约源）
│   │   └── seed_docs.py          # 手动把目录导入指定知识库
│   ├── tests/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── openapi.json                  # 提交的 API 契约（由后端生成）
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx  # 主问答界面（含空状态与检索范围提示）
│   │   │   ├── MessageBubble.tsx  # 回答气泡 + 引用来源（文件 · 知识库）
│   │   │   ├── ResearchModes.tsx  # 四种研究能力选择
│   │   │   ├── KnowledgeBasePanel.tsx # 知识库新建/命名/勾选/删除
│   │   │   ├── DocumentUpload.tsx # 上传到指定知识库 + 片段预览
│   │   │   └── Sidebar.tsx        # 品牌与侧边栏
│   │   ├── researchModes.ts       # 研究能力的单一数据源
│   │   ├── hooks/
│   │   │   ├── useChat.ts         # 问答状态管理（token 帧节流）
│   │   │   └── useKnowledgeBases.ts # 知识库列表与检索范围
│   │   ├── services/api.ts        # HTTP + SSE 客户端（携带匿名身份）
│   │   └── types/
│   │       ├── index.ts           # 领域类型（派生自生成文件）
│   │       └── api.generated.ts   # openapi-typescript 生成，勿手改
│   ├── nginx.conf
│   └── Dockerfile
├── docs/                          # 示例语料（跨领域，启动时自动索引）
│   ├── transformer_attention.md        # 深度学习
│   ├── crispr_gene_editing.md          # 生物医学
│   ├── monetary_policy_transmission.md # 经济学
│   ├── compose_guide.md                # 软件工程
│   └── crash_analysis.md               # 软件工程
├── docker-compose.yml
└── README.md
```
