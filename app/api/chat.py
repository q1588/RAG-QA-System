# -*- coding: utf-8 -*-
"""会话与消息持久化：会话 CRUD、消息查询/更新。"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user, get_db
from app.db.models import Conversation, Message, User
from app.schemas.chat import (
    ConversationCreate,
    ConversationOut,
    MessageOut,
    MessageUpdate,
)

router = APIRouter(prefix="/api/chat", tags=["会话与消息"])


async def _get_own_conversation(
    db: AsyncSession, conversation_id: int, user_id: int
) -> Conversation:
    """取当前用户自己的会话，防止越权访问他人数据。"""
    conv = (
        await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return conv


@router.post(
    "/conversations",
    response_model=ConversationOut,
    status_code=201,
    summary="创建会话",
)
async def create_conversation(
    body: ConversationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conv = Conversation(user_id=current_user.id, title=body.title)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


@router.get(
    "/conversations",
    response_model=list[ConversationOut],
    summary="我的会话列表",
)
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """按最近更新时间倒序返回当前用户的全部会话。"""
    rows = (
        await db.execute(
            select(Conversation)
            .where(Conversation.user_id == current_user.id)
            .order_by(Conversation.updated_at.desc())
        )
    ).scalars().all()
    return rows


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageOut],
    summary="按会话查询消息",
)
async def list_messages(
    conversation_id: int,
    msg_type: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """按会话查询消息，可按 msg_type 过滤；按创建时间正序返回。"""
    await _get_own_conversation(db, conversation_id, current_user.id)

    stmt = select(Message).where(Message.conversation_id == conversation_id)
    if msg_type:
        stmt = stmt.where(Message.msg_type == msg_type)
    stmt = stmt.order_by(Message.created_at.asc())
    rows = (await db.execute(stmt)).scalars().all()
    return rows


@router.patch(
    "/messages/{message_id}",
    response_model=MessageOut,
    summary="更新消息内容",
)
async def update_message(
    message_id: int,
    body: MessageUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """更新消息内容。需先校验消息归属的会话属于当前用户。"""
    msg = (
        await db.execute(select(Message).where(Message.id == message_id))
    ).scalar_one_or_none()
    if msg is None:
        raise HTTPException(status_code=404, detail="消息不存在")

    conv = await _get_own_conversation(db, msg.conversation_id, current_user.id)

    msg.content = body.content
    await db.commit()
    await db.refresh(msg)
    return msg
