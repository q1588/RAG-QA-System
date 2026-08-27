# -*- coding: utf-8 -*-
"""全局配置：读取 .env，环境变量优先于 .env 文件。"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # 项目根目录（RAG-QA-System/）


class Settings(BaseSettings):
    """集中管理所有可配置项，避免散落在各模块的魔法数字。

    本版本相对旧版的主要升级点：
    - 向量库：Chroma 本地目录  ->  Milvus（本地开发用 Milvus Lite 单文件，
      Docker/生产用独立 Milvus 服务，见 docker-compose.yml）；
    - Embedding：Ollama 本地小模型  ->  在线 API（OpenAI 兼容协议，
      默认 SiliconFlow 的 BAAI/bge-m3；可换 OpenAI / 阿里云百炼 / DeepSeek 等）；
    - LLM：Ollama 本地模型  ->  在线 API 服务（OpenAI 兼容协议，异步并发调用）；
    - 切块：递归切分  ->  递归 / 语义 / 混合三种策略可切换；
    - 全部外部服务地址与密钥走环境变量，无密钥时自动降级为本地 Ollama。
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 应用
    APP_NAME: str = "基于大模型的RAG知识库问答系统"

    # 数据库：本地默认 SQLite（零依赖），Docker 部署时通过环境变量切 MySQL
    DATABASE_URL: str = "sqlite+aiosqlite:///./app.db"
    DB_ECHO: bool = False  # True 时打印 SQL，方便排查

    # 认证
    JWT_SECRET: str = "change-me-to-a-long-random-string"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440  # 24 小时

    # 上传文档目录
    UPLOAD_DIR: str = "./upload_files"
    MAX_UPLOAD_SIZE_MB: int = 20            # 单文件上传大小上限（MB），超限返回 413

    # ---------- 向量数据库（Milvus）----------
    # url 为本地文件路径（如 ./milvus_store/rag.db）时使用 Milvus Lite（嵌入式，零依赖）；
    # 为 http(s)://host:port 时连接独立 Milvus 服务（Docker / 生产）。
    MILVUS_URL: str = "./milvus_store/rag.db"
    MILVUS_COLLECTION: str = "rag_docs"
    MILVUS_TOKEN: str = ""                    # 可选：Milvus 鉴权 token / Zilliz Cloud API Key
    MILVUS_METRIC_TYPE: str = "COSINE"        # 向量度量：COSINE（余弦相似度）

    # ---------- Embedding（在线 API，OpenAI 兼容协议）----------
    # provider = openai：在线 API（OpenAI / SiliconFlow / 阿里云百炼 / DeepSeek 等兼容端点）
    # provider = ollama：本地离线（OLLAMA_BASE_URL 指向本地 Ollama 服务）
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_BASE_URL: str = "https://api.siliconflow.cn/v1"  # 例：https://api.openai.com/v1
    EMBEDDING_API_KEY: str = ""
    EMBED_MODEL: str = "BAAI/bge-m3"           # SiliconFlow 上效果更好的大参数模型（1024 维）
    # OpenAI 官方可用 text-embedding-3-large；阿里云百炼可用 text-embedding-v3
    EMBEDDING_BATCH_SIZE: int = 10            # 向量化批大小（百炼 text-embedding-v3 兼容接口单批上限 10；OpenAI 官方可调大）

    # ---------- LLM（在线 API，OpenAI 兼容协议）----------
    # provider = openai：在线 API 服务，异步调用、天然支持高并发
    # provider = ollama：本地离线（仅开发/内网场景）
    LLM_PROVIDER: str = "openai"
    LLM_BASE_URL: str = "https://api.siliconflow.cn/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"  # SiliconFlow 上按量计费的 API 模型
    LLM_TEMPERATURE: float = 0.3
    LLM_TIMEOUT: int = 120                    # 单次生成请求超时（秒）
    LLM_MAX_RETRIES: int = 2                  # 失败重试次数（提升稳定性）

    # Ollama 本地模型（EMBEDDING_PROVIDER/LLM_PROVIDER = ollama 时使用）
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_LLM_MODEL: str = "qwen2.5:3b"

    # ---------- 切块策略 ----------
    # recursive：递归字符切分（默认，稳定、零外部依赖）
    # semantic ：语义切分（基于相邻句向量相似度在最大落差处断句，语义更完整）
    # hybrid   ：混合切分（先递归粗分，再用语义相似度合并相邻块，兼顾长度与语义）
    CHUNKING_STRATEGY: str = "recursive"
    CHUNK_SIZE: int = 500                     # 递归切块大小（字符）
    CHUNK_OVERLAP: int = 60                   # 递归切块重叠（字符）
    SEMANTIC_BREAKPOINT_PERCENTILE: float = 95.0  # 语义切分断点分位：取相似度最低的 ~5% 间隔断开
    SEMANTIC_MIN_CHUNK_CHARS: int = 100       # 语义切块最小字符数（过小与相邻块合并）
    SEMANTIC_MAX_CHUNK_CHARS: int = 1200      # 语义切块最大字符数（超长句子强制硬切）
    SEMANTIC_MERGE_THRESHOLD: float = 0.8     # 混合切分中相邻块合并的余弦相似度阈值

    # ---------- 检索 ----------
    TOP_K: int = 3                            # 检索返回片段数
    # 相关性阈值：低于此值（余弦相似度 0~1）视为知识库无相关内容，直接诚实回答
    RAG_RELEVANCE_THRESHOLD: float = 0.5
    MAX_CONTEXT_HISTORY: int = 8              # 多轮对话记忆条数

    # ---------- RAG 评估 ----------
    EVAL_DATASET: str = str(BASE_DIR / "data" / "eval_dataset.jsonl")
    EVAL_CORPUS_DIR: str = str(BASE_DIR / "data" / "eval_corpus")


settings = Settings()