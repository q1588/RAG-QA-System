# -*- coding: utf-8 -*-
"""全局配置：读取 .env，环境变量优先于 .env 文件。"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # 项目根目录（8.11/）


class Settings(BaseSettings):
    """集中管理所有可配置项，避免散落在各模块的魔法数字。"""

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

    # Ollama 本地大模型
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    EMBED_MODEL: str = "nomic-embed-text"  # 向量化模型（中文/英文通用）
    LLM_MODEL: str = "qwen2.5:3b"          # 生成模型

    # RAG
    CHROMA_DIR: str = "./chroma_store"     # 向量库持久化目录
    UPLOAD_DIR: str = "./upload_files"     # 文档上传目录
    CHUNK_SIZE: int = 500                  # 切块大小
    CHUNK_OVERLAP: int = 60                # 切块重叠
    TOP_K: int = 3                         # 检索返回片段数
    MAX_CONTEXT_HISTORY: int = 8           # 多轮对话记忆条数


settings = Settings()
