# -*- coding: utf-8 -*-
"""会话与消息的请求/响应模型。"""
from datetime import datetime

from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    title: str = Field(default="新对话", max_length=200)


class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MessageOut(BaseModel):
    id: int
    conversation_id: int
    role: str
    content: str
    msg_type: str
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageUpdate(BaseModel):
    content: str = Field(min_length=1)
