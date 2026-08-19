# -*- coding: utf-8 -*-
"""FastAPI 依赖注入：数据库会话、当前用户解析。"""
from typing import AsyncIterator

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_access_token
from app.db.models import User
from app.db.session import AsyncSessionLocal

# tokenUrl 指向登录接口：既从 Authorization: Bearer 取令牌，
# 又让 Swagger UI 自动渲染出可用的 Authorize 按钮。
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


async def get_db() -> AsyncIterator[AsyncSession]:
    """每个请求一个数据库会话，自动开启/提交/关闭。"""
    async with AsyncSessionLocal() as session:
        yield session


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """解析 Bearer Token -> 用户。无效或过期统一返回 401。"""
    credentials_error = HTTPException(
        status_code=401,
        detail="登录凭证无效或已过期，请重新登录",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError, TypeError):
        raise credentials_error

    user = (
        await db.execute(select(User).where(User.id == user_id))
    ).scalar_one_or_none()
    if user is None:
        raise credentials_error
    return user
