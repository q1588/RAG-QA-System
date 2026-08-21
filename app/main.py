# -*- coding: utf-8 -*-
"""应用入口：FastAPI 装配、建表、路由注册、静态页面托管。

启动方式：
    python -m uvicorn app.main:app --reload --port 8000
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.api import auth, chat, rag
from app.core.config import BASE_DIR, settings
from app.db.init_db import init_db
from app.db.session import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 确保上传/向量库目录存在（.gitignore 已忽略，全新 clone 后不存在，
    # 否则首次上传会因找不到 upload_files 目录而 500）
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.CHROMA_DIR, exist_ok=True)
    # 启动时自动建表（幂等，已存在的表跳过）
    await init_db()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description="基于大模型的 RAG 知识库问答系统：认证 / 会话消息持久化 / RAG 检索问答",
    version="1.0.0",
    lifespan=lifespan,
)

# 允许跨域（前后端分离场景 / 本地调试）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(rag.router)

# 静态资源（聊天页面）
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health", tags=["系统"], summary="健康检查")
async def health():
    """健康检查：供 Docker healthcheck / 监控探测，返回服务与数据库状态。"""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        database = "ok"
    except Exception:
        database = "error"
    return {"status": "ok" if database == "ok" else "degraded", "database": database}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
