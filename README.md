# 📚 基于大模型的 RAG 知识库问答系统

一个**能跑、能讲透**的 RAG（检索增强生成）知识库问答系统：用户注册登录 → 上传文档 → 向量化建库 → 多轮问答（每次回答自动带上知识来源片段）。后端 `FastAPI + SQLAlchemy(async) + LangChain + Chroma + Ollama`，前端为 FastAPI 托管的单文件聊天页，支持 `Docker + MySQL` 一键部署。


## ✨ 功能一览

| 模块 | 功能点 |
|---|---|
| 🔐 认证模块 | 注册 / 登录 / 当前用户；bcrypt 加盐哈希（永不存明文）；捕获唯一约束冲突事务回滚防重复注册；JWT 无状态鉴权 |
| 💬 消息持久化 | `users / conversations / messages` 三表设计；异步写入、按会话 / 按类型查询、内容更新；最近 N 条历史过滤倒序截取 → 多轮对话记忆 |
| 🔍 RAG 检索 | 上传 txt / md / pdf / docx → 文本切分 → 向量化 → Chroma 相似度检索 → 大模型生成；回答附带命中的来源片段（可溯源） |
| 🚀 部署 | 本地 SQLite 零依赖即跑；`docker compose` 一键 MySQL + API 部署；.env 配置化 |

## 🏗️ 系统架构

```
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
        │ SQLAlchemy  │            │ LangChain 1.x（rag_service.py）
        │  async      │            ├─ OllamaEmbeddings(nomic-embed-text)
┌───────▼────┐ ┌──────▼────┐      ├─ RecursiveCharacterTextSplitter
│ MySQL/SQLite│ │ Chroma    │      ├─ Chroma 向量库
│ users/      │ │ 向量库    │      └─ ChatOllama(qwen2.5:3b)
│ conversations││ chroma_store│
│ messages    │ │           │
└────────────┘ └───────────┘
```

**一次完整问答的数据流（能讲透的 RAG 链路）：**

1. 用户携带 JWT 调用 `POST /api/rag/query`，传入 `conversation_id + question`；
2. 服务端校验会话归属（防越权）；
3. `memory.build_history` 从 `messages` 表读取最近 N 条有效历史，组装多轮上下文；
4. `rag_service.rag_query`：问题向量化 → Chroma 相似度检索 Top-K 片段（返回原始距离，越小越相似）→ 拼装【历史+知识库+问题】提示词 → qwen2.5:3b 生成；
5. 将用户问题、助手回答两条消息**异步写入 messages 表**（消息持久化）；
6. 返回答案 + `source_docs`（命中的知识片段，前端可展开查看，体现 RAG 可解释性）。

## 🚀 快速启动（本地，Windows）

**前置**：安装 [Ollama](https://ollama.com/) 并拉取模型（已装可跳过）：

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

**1. 安装依赖**（使用项目已有 venv）：

```bash
cd 8.11
../.venv/Scripts/pip install -r requirements.txt
```

**2. 启动服务**（确保 Ollama 在运行）：

```bash
../.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
```

**3. 打开使用**：

- 聊天页面：http://127.0.0.1:8000/
- Swagger 接口文档：http://127.0.0.1:8000/docs （登录接口右上角有 **Authorize** 按钮，可在线调试）

**4. 给知识库导入文档**：登录后在 `/docs` 里调用 `upload` 上传 txt/md/pdf/docx → `vectorize` 向量化 → 即可在聊天页提问。

> 若端口 8000 被占用，换端口后通过 `SMOKE_BASE_URL=http://127.0.0.1:8001 ../.venv/Scripts/python.exe scripts/smoke_test.py` 跑冒烟测试。

## 🧪 端到端冒烟测试

```bash
../.venv/Scripts/python -m uvicorn app.main:app --port 8000 &   # 先起服务
../.venv/Scripts/python scripts/smoke_test.py                    # 注册→登录→会话→问答→消息断言
```

单测（安全 / 记忆过滤）：

```bash
../.venv/Scripts/python -m pytest tests -v
```

## 📮 接口文档（Postman / Swagger）

| 方法 | 路径 | 鉴权 | 说明 |
|---|---|---|---|
| POST | `/api/auth/register` | 无 | 注册，`{username, password}`；重复用户名 → 409 |
| POST | `/api/auth/login` | 无 | OAuth2 表单登录 → `{access_token}` |
| GET | `/api/auth/me` | Bearer | 当前用户信息 |
| POST | `/api/chat/conversations` | Bearer | 创建会话 |
| GET | `/api/chat/conversations` | Bearer | 我的会话列表 |
| GET | `/api/chat/conversations/{id}/messages` | Bearer | 按会话查消息（可带 `?msg_type=`） |
| PATCH | `/api/chat/messages/{id}` | Bearer | 更新消息内容 |
| POST | `/api/rag/upload` | Bearer | 上传文档（multipart，txt/md/pdf/docx） |
| POST | `/api/rag/vectorize` | Bearer | 向量化已上传文档 |
| POST | `/api/rag/query` | Bearer | RAG 问答 `{conversation_id, question, top_k?}` |

## 🐳 Docker 部署

```bash
docker compose up -d --build
# API:  http://localhost:8000    MySQL: rag_db
```

详见 [docs/部署文档.md](docs/部署文档.md)。

## 🗂️ 项目结构

```
8.11/
├── app/
│   ├── main.py            # 入口：建表、CORS、路由、静态页
│   ├── core/
│   │   ├── config.py      # 配置（.env，pydantic-settings）
│   │   ├── security.py    # bcrypt 密码哈希 + JWT 签发/校验
│   │   └── deps.py        # OAuth2PasswordBearer + 当前用户依赖
│   ├── db/
│   │   ├── session.py     # 异步引擎（SQLite/MySQL 双驱动）
│   │   ├── models.py      # User / Conversation / Message
│   │   └── init_db.py     # 建表
│   ├── schemas/           # Pydantic 请求/响应模型
│   ├── api/               # auth / chat / rag 三个路由
│   └── services/
│       ├── rag_service.py # 文档加载、切分、向量化、检索、生成
│       └── memory.py      # 最近 N 条历史 → 多轮记忆
├── static/index.html      # 单文件聊天页
├── scripts/smoke_test.py  # 端到端冒烟测试
├── tests/                 # 单元测试
├── Dockerfile / docker-compose.yml / .env.example
└── docs/部署文档.md
```


## 📌 可扩展方向

- **混合检索**：BM25 关键词 + 向量语义融合打分（OpenSearch Hybrid），已在本项目早期迭代验证过思路；
- **联网检索**：本地知识库检索不到时，接入 Tavily 等工具做实时联网增强；
- **流式输出**：SSE 逐字返回答案，提升体验；
- **多租户**：向量库 metadata 中标记 `user_id`，实现用户级知识库隔离。

---

项目由单一 `main.py` 重构为模块化工程，旧版保留在 [examples/legacy_main.py](examples/legacy_main.py)。
