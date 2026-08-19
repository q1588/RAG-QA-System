# -*- coding: utf-8 -*-
"""应用入口：FastAPI 装配、建表、路由注册、静态页面托管。

启动方式：
    python -m uvicorn app.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import auth, chat, rag
from app.core.config import BASE_DIR, settings
from app.db.init_db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
