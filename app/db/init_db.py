# -*- coding: utf-8 -*-
"""初始化数据库表。既可直接运行 `python -m app.db.init_db`，也由 FastAPI lifespan 启动时自动调用。"""
import asyncio

from app.db import models  # noqa: F401  确保模型被注册进 metadata
from app.db.session import engine


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)


if __name__ == "__main__":
    asyncio.run(init_db())
    print("✅ 数据库表创建/校验完成")
