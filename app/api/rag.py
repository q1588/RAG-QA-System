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
    except Exception as e:
        # Ollama 不可用：返回 503 友好提示；真正的 bug 照常 500
        if rag_service.is_backend_unavailable(e):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"向量化失败：无法连接本地模型服务（{type(e).__name__}），"
                    "请确认 Ollama 已启动并拉取了 nomic-embed-text"
                ),
            )
        raise
    return result


@router.get("/files", summary="已上传文档列表")
async def list_uploaded_files(current_user: User = Depends(get_current_user)):
    """列出 upload_files 目录下的文档，供前端展示知识库内容。"""
    if not os.path.isdir(settings.UPLOAD_DIR):
        return []
    return [
        {"filename": name, "size": os.path.getsize(p)}
        for name in sorted(os.listdir(settings.UPLOAD_DIR))
        if os.path.isfile(p := os.path.join(settings.UPLOAD_DIR, name))
    ]


@router.delete("/files/{filename}", summary="删除文档")
async def delete_document(
    filename: str,
    current_user: User = Depends(get_current_user),
):
    """删除文档：从 Chroma 清除该文件的全部分片，并移除上传文件。"""
    return await asyncio.to_thread(rag_service.delete_file, filename)


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
    try:
        result = await asyncio.to_thread(
            rag_service.rag_query, body.question, history, body.top_k
        )
    except Exception as e:
        if not rag_service.is_backend_unavailable(e):
            raise  # 非外部服务问题，保留 500 便于排查
        # 优雅降级：返回兜底答案并照常持久化。
        # memory 模块的 _INVALID_MARKERS 会把这些「请求失败」回答挡在多轮上下文外。
        result = {
            "answer": (
                f"抱歉，请求失败：无法连接本地大模型服务（{type(e).__name__}）。"
                "请确认 Ollama 已启动并拉取了 qwen2.5:3b 与 nomic-embed-text 后重试。"
            ),
            "source_docs": [],
        }

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
