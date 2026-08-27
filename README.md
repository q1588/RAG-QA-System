# 📚 基于大模型的 RAG 知识库问答系统

一个**能跑、能讲透、可量化**的 RAG（检索增强生成）知识库问答系统：用户注册登录 → 上传文档 → 向量化建库 → 多轮问答（每次回答自动带上知识来源片段）。后端 `FastAPI + SQLAlchemy(async) + LangChain + Milvus + 在线模型 API`，前端为 FastAPI 托管的单文件聊天页，支持 `Docker + MySQL + Milvus` 一键部署。

> v2 升级点（相对 v1）：
> 1. **向量库替换为 Milvus**：本地开发用 Milvus Lite 单文件（零依赖），Docker/生产连接独立 Milvus 服务；
> 2. **Embedding 升级为在线大参数模型**：OpenAI 兼容协议，默认 `BAAI/bge-m3`（1024 维），可换 `text-embedding-3-large` 等；
> 3. **语义切分补充递归切分**：三种策略可切换（recursive / semantic / hybrid），文档解析逐页/逐段容错；
> 4. **引入 RAG 评估数据集**：`scripts/eval_rag.py` 量化 Hit@K / Recall@K / Precision@K / MRR@K，不再肉眼看效果；
> 5. **模型调用改为 API 服务**：全链路异步（async embedding + Milvus 检索 + LLM ainvoke），天然支持高并发。

## ✨ 功能一览

| 模块 | 功能点 |
|---|---|
| 🔐 认证模块 | 注册 / 登录 / 当前用户；bcrypt 加盐哈希（永不存明文）；捕获唯一约束冲突事务回滚防重复注册；JWT 无状态鉴权 |
| 💬 消息持久化 | `users / conversations / messages` 三表设计；异步写入、按会话 / 按类型查询、内容更新；最近 N 条历史过滤倒序截取 → 多轮对话记忆 |
| 🔍 RAG 检索 | 上传 txt / md / pdf / docx → 切块（递归/语义/混合）→ 向量化 → **Milvus** 相似度检索 → 在线大模型生成；回答附带命中的来源片段（可溯源） |
| 📊 RAG 评估 | 内置评估数据集（`data/eval_dataset.jsonl`），一键跑出召回指标，支持多策略 / 多 Top-K 对比 |
| 🚀 部署 | 本地 SQLite + Milvus Lite 零依赖即跑；`docker compose` 一键 MySQL + Milvus + API 部署；.env 配置化 |

## 🏗️ 系统架构

`
┌─────────────── 前端（static/index.html，vanilla JS）───────────────┐
│  注册/登录 → JWT → 会话列表 → 聊天框 → 展示答案+来源片段              │
└──────────────────────────────┬────────────────────────────────────┘
                               │ HTTP /api/*
┌──────────────────────────────▼────────────────────────────────────┐
│                     FastAPI（app/main.py）                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────────────────┐  │
│  │ /auth   │  │ /chat   │  │ /rag    │  │ 依赖注入 deps.py       │  │
│  │ 注册登录 │  │ 会话消息 │  │ 上传检索 │  │ OAuth2 Bearer 解析    │  │
│  └────┬────┘  └────┬────┘  └────┬────┘  └──────────────────────┘  │
└───────┼────────────┼────────────┼─────────────────────────────────┘
        │ SQLAlchemy  │            │ LangChain 1.x（rag_service.py，全异步）
        │  async      │            ├─ OpenAI 兼容 Embedding API（BAAI/bge-m3）
┌───────▼────┐ ┌──────▼────┐      ├─ 递归/语义/混合切块（chunking.py）
│ MySQL/SQLite│ │  Milvus   │      ├─ Milvus 向量库（vector_store.py）
│ users/      │ │ 向量库    │      └─ 在线 LLM API（ChatOpenAI，ainvoke）
│ conversations││ Lite/服务 │
│ messages    │ │           │
└────────────┘ └───────────┘
`

**一次完整问答的数据流（能讲透的 RAG 链路）：**

1. 用户携带 JWT 调用 `POST /api/rag/query`，传入 `conversation_id + question`；
2. 服务端校验会话归属（防越权）；
3. `memory.build_history` 从 `messages` 表读取最近 N 条有效历史，组装多轮上下文；
4. `rag_service.rag_query`（全异步）：问题向量化（归一化）→ Milvus 相似度检索 Top-K（返回余弦相似度，越大越相关）→ 低于相关性阈值直接如实回答 → 否则拼装【历史+知识库+问题】提示词 → 在线 LLM API 异步生成；
5. 将用户问题、助手回答两条消息**异步写入 messages 表**（消息持久化）；
6. 返回答案 + `source_docs`（命中的知识片段 + relevance 分数，前端可展开查看）。

## 🚀 快速启动（本地，Windows）

**前置 1：Python 依赖**

`
cd RAG-QA-System
../.venv/Scripts/pip install -r requirements-dev.txt   # 基础 + RAG + pytest（跑单测用）
`

**前置 2：配置外部模型服务（`.env`，参考 `.env.example`）**

`
copy .env.example .env
# 编辑 .env，至少填写一个 Key：
EMBEDDING_API_KEY=sk-xxx    # SiliconFlow / OpenAI / 阿里云百炼 等
LLM_API_KEY=sk-xxx
`

不填 Key 也可以启动：系统自动降级为本地 Ollama（`EMBEDDING_PROVIDER=ollama` / `LLM_PROVIDER=ollama`）。

**启动服务：**

`
../.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
`

**打开使用：**

- 聊天页面：http://127.0.0.1:8000/
- Swagger 接口文档：http://127.0.0.1:8000/docs （登录接口右上角有 **Authorize** 按钮，可在线调试）
- 向量库：默认 `./milvus_store/rag.db`（Milvus Lite 嵌入式，首次向量化自动创建）

**给知识库导入文档**：登录后点击左侧「📤 上传文档」，选择 txt/md/pdf/docx，系统会自动上传并向量化入库（侧边栏可看到已上传文档），即可在聊天页提问。

> 若端口 8000 被占用，换端口后通过 `SMOKE_BASE_URL=http://127.0.0.1:8001 ../.venv/Scripts/python.exe scripts/smoke_test.py` 跑冒烟测试。

## 📊 RAG 评估（量化召回效果，升级需求 #4）

内置 14 条中文评估问题（`data/eval_dataset.jsonl`，语料 `data/eval_corpus/`），每条带 ground-truth 文本片段，可跨切块策略复评：

`
# 离线自检（假 Embedding，确定性可复现，无需 Key）
../.venv/Scripts/python.exe scripts/eval_rag.py --embeddings fake

# 真实效果（使用 .env 配置的 Embedding API）
../.venv/Scripts/python.exe scripts/eval_rag.py --embeddings real

# 对比三种切块策略 + 多个 Top-K
../.venv/Scripts/python.exe scripts/eval_rag.py --embeddings real --strategies recursive,semantic,hybrid --top-k 3,5
`

输出 Markdown 表格（Hit@K / Recall@K / Precision@K / MRR@K）+ `eval_report.json`（含每题检索明细）。把结果存为基线，每次调整切块/检索参数后重跑即可回归对比，替代「肉眼挑几条试试」。

> ⚠️ 注意：
> - `--embeddings fake` 仅用于流程验证（确定性、可复现），**不代表真实效果**；上线基线请用 `--embeddings real`。
> - 切换不同维度的 Embedding 模型后，向量集合需重建：向量化时携带 `rebuild=true`（或删除 `milvus_store/` 后重跑）。
> - `RAG_RELEVANCE_THRESHOLD` 默认 0.5 为初始值，建议按 eval 报告的分数分布校准（不同模型无关文本余弦基线不同）。

## 🧪 测试

`
../.venv/Scripts/python -m pytest tests -v
`

单测覆盖：安全（bcrypt/JWT）、多轮记忆过滤、**切块策略（递归/语义/混合）**、**文档解析容错**、**Milvus 向量库读写/过滤删除/幂等覆盖**（Milvus Lite + 确定性 Embedding，无需外部服务）、**评估指标计算**。

端到端冒烟（需真实 API Key 或本地 Ollama）：

`
../.venv/Scripts/python -m uvicorn app.main:app --port 8000 &   # 先起服务
../.venv/Scripts/python scripts/smoke_test.py                    # 注册→登录→会话→问答→消息断言
`

## 📮 接口文档（Postman / Swagger）

| 方法 | 路径 | 鉴权 | 说明 |
|---|---|---|---|
| GET | `/health` | 无 | 健康检查（服务 + 数据库 + 向量库状态） |
| POST | `/api/auth/register` | 无 | 注册，`{username, password}`；重复用户名 → 409 |
| POST | `/api/auth/login` | 无 | OAuth2 表单登录 → `{access_token}` |
| GET | `/api/auth/me` | Bearer | 当前用户信息 |
| POST | `/api/chat/conversations` | Bearer | 创建会话 |
| GET | `/api/chat/conversations` | Bearer | 我的会话列表 |
| GET | `/api/chat/conversations/{id}/messages` | Bearer | 按会话查消息（可带 `?msg_type=`） |
| PATCH | `/api/chat/messages/{id}` | Bearer | 更新消息内容 |
| POST | `/api/rag/upload` | Bearer | 上传文档（multipart，txt/md/pdf/docx） |
| GET | `/api/rag/files` | Bearer | 已上传文档列表 |
| DELETE | `/api/rag/files/{filename}` | Bearer | 删除文档（清除向量分片 + 移除文件） |
| POST | `/api/rag/vectorize` | Bearer | 向量化已上传文档，可带 `strategy`（recursive/semantic/hybrid）；幂等（同名文档先清旧分片） |
| POST | `/api/rag/query` | Bearer | RAG 问答 `{conversation_id, question, top_k?}`；命中片段带 `relevance`（0~1），低于阈值如实返回「知识库无相关内容」 |

## 🐳 Docker 部署

`
docker compose up -d --build
# API: http://localhost:8000   MySQL: rag_db   Milvus: localhost:19530
`

详见 [docs/部署文档.md](docs/部署文档.md)。

## 🗂️ 项目结构

`
RAG-QA-System/
├── app/
│   ├── main.py            # 入口：建表、CORS、路由、静态页、Milvus 初始化
│   ├── core/
│   │   ├── config.py      # 配置（.env，pydantic-settings；Milvus/Embedding/LLM/切块）
│   │   ├── security.py    # bcrypt 密码哈希 + JWT 签发/校验
│   │   └── deps.py        # OAuth2PasswordBearer + 当前用户依赖
│   ├── db/                # session.py / models.py / init_db.py
│   ├── schemas/           # Pydantic 请求/响应模型
│   ├── api/               # auth / chat / rag 三个路由
│   └── services/
│       ├── embeddings.py      # 在线 Embedding 工厂（OpenAI 兼容 + Ollama 兜底 + L2 归一化）
│       ├── llm.py             # 在线 LLM 工厂（ChatOpenAI，异步 ainvoke）
│       ├── document_loader.py # 文档解析容错（编码回退 / 逐页提取 / 友好报错）
│       ├── chunking.py        # 递归 / 语义 / 混合切块
│       ├── vector_store.py    # Milvus 向量库封装（懒加载单例 / 幂等写入）
│       ├── rag_service.py     # 全异步 RAG 编排：加载→切块→向量化→检索→生成
│       ├── eval_metrics.py    # 检索评估指标（Hit/Recall/Precision/MRR，纯函数）
│       └── memory.py          # 最近 N 条历史 → 多轮记忆
├── data/
│   ├── eval_dataset.jsonl     # RAG 评估数据集（14 条中文问题 + ground truth）
│   └── eval_corpus/           # 评估语料（与线上 upload_files 解耦）
├── scripts/
│   ├── smoke_test.py          # 端到端冒烟测试
│   └── eval_rag.py            # 召回评估脚本（多策略 / 多 Top-K 对比）
├── static/index.html          # 单文件聊天页
├── tests/                     # 单测（含切块 / 文档容错 / Milvus / 评估指标）
├── Dockerfile / docker-compose.yml / .env.example
└── docs/部署文档.md
`

## 📌 可扩展方向

- **混合检索**：Milvus 全文检索（BM25）+ 向量语义融合打分；
- **联网检索**：本地知识库检索不到时，接入 Tavily 等工具做实时联网增强；
- **流式输出**：SSE 逐字返回答案，提升体验；
- **多租户**：向量库 metadata 中标记 `user_id`，实现用户级知识库隔离；
- **重排（Rerank）**：检索 Top-K 后用 BGE-Reranker 精排，进一步提升召回精度。

---

项目由单一 `main.py` 重构为模块化工程，旧版保留在 [examples/legacy_main.py](examples/legacy_main.py)。