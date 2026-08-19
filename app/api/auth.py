# -*- coding: utf-8 -*-
"""认证模块：注册 / 登录 / 当前用户。"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user, get_db
from app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from app.db.models import User
from app.schemas.user import Token, UserOut, UserRegister

router = APIRouter(prefix="/api/auth", tags=["认证"])


@router.post(
    "/register",
    response_model=UserOut,
    status_code=201,
    summary="用户注册",
)
async def register(body: UserRegister, db: AsyncSession = Depends(get_db)):
    """注册新用户。用户名唯一；密码 bcrypt 加盐哈希后落库，绝不存明文。"""
    try:
        hashed = hash_password(body.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    user = User(username=body.username, hashed_password=hashed)
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:
        # 用户名唯一约束冲突：事务回滚，返回 409，防止重复注册
        await db.rollback()
        raise HTTPException(status_code=409, detail="用户名已存在")
    await db.refresh(user)
    return user


@router.post("/login", response_model=Token, summary="用户登录")
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """登录校验（OAuth2 表单），成功返回 JWT 访问令牌。

    使用 form 而非 JSON：可直接配合 Swagger 的 Authorize 按钮使用。
    """
    user = (
        await db.execute(select(User).where(User.username == form.username))
    ).scalar_one_or_none()
    if user is None or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return Token(access_token=create_access_token(user.id))


@router.get("/me", response_model=UserOut, summary="获取当前用户")
async def me(current_user: User = Depends(get_current_user)):
    return current_user
