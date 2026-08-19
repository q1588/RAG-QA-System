# -*- coding: utf-8 -*-
"""用户相关的请求/响应模型。"""
from datetime import datetime

from pydantic import BaseModel, Field


class UserRegister(BaseModel):
    """注册请求体。"""

    username: str = Field(min_length=3, max_length=32, description="用户名，3-32 字符")
    password: str = Field(min_length=6, max_length=128, description="密码，至少 6 位")


class UserOut(BaseModel):
    """用户信息响应（绝不返回哈希密码）。"""

    id: int
    username: str
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    """登录成功后返回的 JWT 令牌。"""

    access_token: str
    token_type: str = "bearer"
