# -*- coding: utf-8 -*-
"""RAG 模块：文档上传、向量化、RAG 问答（含多轮记忆与问答持久化）。"""
import asyncio
import os

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.deps import get_current_user, get_db
from app.db.models import Conversation, Message, User
from app.schemas.rag import QueryRequest, QueryResponse, SourceDoc, VectorizeRequest
from app.services import rag_service
from app.services.memory import build_history
from app.api.chat import _get_own_conversation

router = APIRouter(prefix="/api/rag", tags=["RAG 问答"])

SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx"}


@router.post("/upload", summary="上传文档")
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """上传知识库文档（txt/md/pdf/docx），保存到 upload_files。"""
    filename = os.path.basename(file.filename or "unnamed")  # 防路径穿越
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型：{ext or '(无扩展名)'}，仅支持 txt/md/pdf/docx",
        )

    content = await file.read()
    save_path = os.path.join(settings.UPLOAD_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(content)

    return {
        "filename": filename,
        "size": len(content),
        "save_path": save_path,
        "next": "请调用 /api/rag/vectorize 进行向量化",
    }


@router.post("/vectorize", summary="向量化文档")
async def vectorize_document(
    body: VectorizeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """将已上传的文档切块、向量化并写入 Chroma。"""
    try:
        result = await asyncio.to_thread(
            rag_service.vectorize_file, body.filename
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@router.post("/query", response_model=QueryResponse, summary="RAG 问答（多轮）")
async def rag_query(
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """RAG 问答主接口。

    流程：校验会话归属 -> 读取最近 N 条历史（多轮记忆）-> 检索知识库 -> 大模型生成
    -> 将用户问题与助手回答持久化到 messages 表 -> 返回答案与来源片段。
    """
    conv = await _get_own_conversation(db, body.conversation_id, current_user.id)

    history = await build_history(
        db, body.conversation_id, last_n=settings.MAX_CONTEXT_HISTORY
    )

    # Ollama 是同步阻塞调用，放到线程池，避免阻塞事件循环
    result = await asyncio.to_thread(
        rag_service.rag_query, body.question, history, body.top_k
    )

    # 持久化问答对
    db.add(Message(conversation_id=conv.id, role="user", content=body.question))
    db.add(
        Message(conversation_id=conv.id, role="assistant", content=result["answer"])
    )
    conv.updated_at = func.now()  # 刷新会话活跃时间
    await db.commit()

    return QueryResponse(
        conversation_id=body.conversation_id,
        question=body.question,
        answer=result["answer"],
        top_k=body.top_k,
        source_docs=[SourceDoc(**d) for d in result["source_docs"]],
    )
