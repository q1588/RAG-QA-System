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
            ("抱歉，以下是查到的资料：RAG 是检索增强生成。", False),  # 含「抱歉」的正常回答不再误伤
            ("error 是英文单词，不是错误", False),                   # 含「error」的正常回答不再误伤
            ("", True),              # 空内容
            ("   \n ", True),        # 空白内容
            ("抱歉，知识库中暂时没有与「X」相关的内容", True),       # 检索兜底回答
            ("知识库中没有相关信息，请补充文档", True),              # 检索兜底变体
            ("抱歉，请求失败：无法连接外部模型服务（ConnectionError）", True),  # 后端降级回答
        ],
    )
    def test_is_invalid(self, content, expected):
        assert _is_invalid(content) is expected

class TestBuildHistoryOversampling:
    """build_history 在无效消息过半时仍能凑够 last_n 条（分批翻页补足）。"""

    def test_invalid_over_half_still_enough(self):
        import asyncio

        from app.db.models import Conversation, Message
        from app.db.session import AsyncSessionLocal
        from app.services.memory import build_history

        async def body():
            async with AsyncSessionLocal() as db:
                conv = Conversation(user_id=1, title="过采样测试")
                db.add(conv)
                await db.commit()
                await db.refresh(conv)
                # 12 条无效（后端降级回答） + 4 条有效
                for _ in range(12):
                    db.add(
                        Message(
                            conversation_id=conv.id,
                            role="assistant",
                            content="抱歉，请求失败：无法连接外部模型服务（ConnectionError）",
                        )
                    )
                for i in range(4):
                    db.add(
                        Message(conversation_id=conv.id, role="user", content=f"正常问题{i}")
                    )
                await db.commit()
                history = await build_history(db, conv.id, last_n=4)
                assert len(history) == 4
                assert all(m["role"] == "user" for m in history)

        asyncio.run(body())
