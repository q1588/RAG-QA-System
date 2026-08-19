# -*- coding: utf-8 -*-
"""多轮记忆模块单测：历史过滤逻辑（_is_invalid）。"""
import pytest

from app.services.memory import _is_invalid


class TestInvalidFilter:
    @pytest.mark.parametrize(
        "content,expected",
        [
            ("正常的回答内容", False),
            ("RAG是检索增强生成。", False),
            ("", True),              # 空内容
            ("   \n ", True),        # 空白内容
            ("抱歉，我没有找到相关信息", True),  # 兜底/失败回答
            ("我不知道", True),
            ("error: 请求失败", True),
            ("failed to generate", True),
        ],
    )
    def test_is_invalid(self, content, expected):
        assert _is_invalid(content) is expected
