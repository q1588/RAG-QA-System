# -*- coding: utf-8 -*-
"""安全相关：密码哈希（bcrypt）+ JWT 令牌签发/校验。"""
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.core.config import settings


def hash_password(password: str) -> str:
    """bcrypt 加盐哈希密码。

    注意：bcrypt 5.x 对超过 72 字节的输入会直接抛 ValueError（不再静默截断）。
    中文字符 UTF-8 编码占 3 字节，这里按字节数而非字符数做上限校验。
    """
    if len(password.encode("utf-8")) > 72:
        raise ValueError("密码过长（bcrypt 上限为 72 字节）")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """校验明文密码与哈希是否匹配。"""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


def create_access_token(user_id: int) -> str:
    """签发 JWT：sub 为用户 ID，exp 为过期时间。"""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """解析并校验 JWT，无效/过期抛 PyJWTError。"""
    return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
