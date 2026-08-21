# -*- coding: utf-8 -*-
"""数据库：异步引擎 + 会话工厂 + FastAPI 依赖。

make_engine 按连接串前缀分支，让 SQLite / MySQL 各用各自合适的参数：
- SQLite：无需连接池健康检查，但必须关掉 check_same_thread 以便跨线程使用。
- MySQL：pool_pre_ping 在取连接时探测掉线连接，pool_recycle 定期回收长连接。
"""
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings


def make_engine(database_url: str = settings.DATABASE_URL):
    kwargs: dict = {"echo": settings.DB_ECHO}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_pre_ping"] = True
        kwargs["pool_recycle"] = 3600
    return create_async_engine(database_url, **kwargs)


engine = make_engine()

# expire_on_commit=False：commit 后对象属性仍可直接读取，无需重新查询
# 说明：get_db 依赖统一定义在 app.core.deps（本文件不再重复）
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)
