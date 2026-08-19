# -*- coding: utf-8 -*-
"""多轮对话记忆：从数据库中读取最近 N 条有效消息并组装成历史上下文。

实现思路（简历要点）：
1. 按 created_at 倒序多取（over-sample），避免过滤掉无效回答后不够 N 条；
2. 过滤：只保留 user/assistant 的 text 消息，剔除空内容与失败/兜底回答；
3. 截取最近 N 条后 reverse 回时间正序，作为【历史对话】喂给大模型。
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Message

# 兜底/失败回答的特征词，命中即视为无效，不进入多轮上下文
_INVALID_MARKERS = (
    "抱歉",
    "对不起",
    "我不知道",
    "无法回答",
    "error",
    "failed",
    "请求失败",
)


def _is_invalid(content: str) -> bool:
    lowered = content.strip().lower()
    if not lowered:
        return True
    if any(marker in lowered for marker in _INVALID_MARKERS):
        return True
    return False


async def build_history(
    db: AsyncSession, conversation_id: int, last_n: int = 8
) -> list[dict[str, str]]:
    """返回按时间正序排列的最近 N 条有效对话，形如 [{"role", "content"}, ...]。"""
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(last_n * 2)  # 多取一倍，过滤后仍可能不够 N 条
    )
    rows = (await db.execute(stmt)).scalars().all()

    valid: list[Message] = []
    for m in rows:
        if m.role in ("user", "assistant") and m.msg_type == "text":
            if not _is_invalid(m.content):
                valid.append(m)

    valid = valid[:last_n]
    valid.reverse()  # 回到时间正序
    return [{"role": m.role, "content": m.content} for m in valid]
