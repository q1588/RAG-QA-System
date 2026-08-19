# -*- coding: utf-8 -*-
"""认证安全模块单测：bcrypt 哈希 / JWT 签发解析。"""
import pytest
from jwt import ExpiredSignatureError, InvalidTokenError

from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


class TestPasswordHashing:
    def test_hash_and_verify_roundtrip(self):
        hashed = hash_password("secret123")
        assert hashed != "secret123"  # 绝不存明文
        assert hashed.startswith("$2")  # bcrypt 哈希前缀
        assert verify_password("secret123", hashed) is True

    def test_wrong_password_rejected(self):
        hashed = hash_password("correct-pass")
        assert verify_password("wrong-pass", hashed) is False

    def test_same_password_different_salt(self):
        """同密码两次哈希应不同（每次随机加盐）。"""
        assert hash_password("abc123456") != hash_password("abc123456")

    def test_chinese_password_works(self):
        """中文密码按字节算，25+ 个汉字会超过 72 字节，需被拒绝。"""
        hashed = hash_password("中文密码测试123")
        assert verify_password("中文密码测试123", hashed) is True

    def test_password_over_72_bytes_rejected(self):
        # 25 个汉字 = 75 字节 > 72
        with pytest.raises(ValueError):
            hash_password("汉" * 25)


class TestJWT:
    def test_token_roundtrip(self):
        token = create_access_token(user_id=42)
        payload = decode_access_token(token)
        assert payload["sub"] == "42"

    def test_invalid_token_rejected(self):
        with pytest.raises(InvalidTokenError):
            decode_access_token("not-a-real-token")

    def test_tampered_token_rejected(self):
        token = create_access_token(user_id=1)
        with pytest.raises(InvalidTokenError):
            decode_access_token(token + "x")
