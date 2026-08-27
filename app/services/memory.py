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

# 系统生成的兜底/失败回答特征短语（精确匹配，避免误伤含「抱歉/error」等词的正常回答）。
# 仅以下两类系统输出会被剔除：
#   1) 检索兜底：「抱歉，知识库中暂时没有…」「知识库中没有相关信息…」
#   2) 后端降级：「抱歉，请求失败：无法连接外部模型服务…」
_INVALID_MARKERS = (
    "知识库中暂时没有",
    "知识库中没有相关信息",
    "请求失败：无法连接",
    "无法连接外部模型服务",
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
    # 分批向后翻页采样，直到凑够 last_n 条有效消息（或没有更多历史）。
    # 固定过采样（如 ×2）在无效/失败回答过半时会导致历史不足，这里改为循环补足。
    valid: list[Message] = []
    offset = 0
    page_size = max(last_n * 2, 10)
    while len(valid) < last_n:
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        rows = (await db.execute(stmt)).scalars().all()
        if not rows:
            break
        for m in rows:
            if m.role in ("user", "assistant") and m.msg_type == "text":
                if not _is_invalid(m.content):
                    valid.append(m)
                    if len(valid) >= last_n:
                        break
        offset += page_size

    valid = valid[:last_n]
    valid.reverse()  # 回到时间正序
    return [{"role": m.role, "content": m.content} for m in valid]