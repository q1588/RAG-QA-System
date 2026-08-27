# -*- coding: utf-8 -*-
"""RAG 模块：文档上传、向量化、RAG 问答（含多轮记忆与问答持久化）。

v2 变更：向量化/问答全部走 rag_service 的异步实现（Milvus + 在线 Embedding/LLM API），
不再需要 asyncio.to_thread 包裹外部模型调用。
"""
import os

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.deps import get_current_user, get_db
from app.db.models import Conversation, Message, User
from app.schemas.rag import QueryRequest, QueryResponse, SourceDoc, VectorizeRequest
from app.services import rag_service
from app.services.vector_store import DimensionMismatchError
from app.services.document_loader import SUPPORTED_EXTS
from app.services.memory import build_history
from app.api.chat import _get_own_conversation

router = APIRouter(prefix="/api/rag", tags=["RAG 问答"])


def _backend_error_detail(exc: BaseException) -> str:
    """把外部后端错误转成面向用户的 503 提示。"""
    name = type(exc).__name__
    if "API_KEY" in str(exc) or "未配置" in str(exc):
        return (
            f"外部模型服务未配置完成（{name}）：{exc}。"
            "请参考 .env.example 配置 EMBEDDING_API_KEY / LLM_API_KEY 后重试。"
        )
    if rag_service.is_milvus_error(exc):
        return (
            f"向量数据库（Milvus）暂不可用（{name}）：{exc}。"
            "请确认 MILVUS_URL 配置正确且服务已启动；"
            "本地开发请检查 milvus_store/ 目录是否可写、文件未被占用。"
        )
    return (
        f"请求失败：无法连接外部模型服务（{name}）。"
        "请确认相关服务地址与密钥配置正确（.env 中 EMBEDDING_*/LLM_* 或 Ollama）。"
    )


@router.post("/upload", summary="上传文档")
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """上传知识库文档（txt/md/pdf/docx），保存到 upload_files。"""
    filename = os.path.basename((file.filename or "").replace("\\", "/"))  # 防路径穿越
    if not filename or filename in (".", ".."):
        raise HTTPException(status_code=400, detail="非法的文件名")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型：{ext or '(无扩展名)'}，仅支持 txt/md/pdf/docx",
        )

    save_path = os.path.join(settings.UPLOAD_DIR, filename)
    # 同名覆盖上传：先清旧向量分片，保证「磁盘文件 <-> 向量分片」一致
    # （前端上传后会自动重新 vectorize；即使不重新向量化，旧分片也不会残留；
    #   向量清理失败不阻断上传 —— 下次 vectorize 会幂等覆盖）
    if os.path.isfile(save_path):
        try:
            await rag_service.adelete_vectors_only(filename)
        except Exception:
            pass

    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大（{len(content) / 1024 / 1024:.1f}MB），"
            f"上限为 {settings.MAX_UPLOAD_SIZE_MB}MB",
        )
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
    """将已上传的文档切块（可选策略）、向量化并写入 Milvus。"""
    try:
        result = await rag_service.vectorize_file(body.filename, body.strategy, body.rebuild)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except DimensionMismatchError as e:
        # Embedding 维度变化且未传 rebuild：409 提示重建
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        # 文档解析失败/切块为空/未知策略/路径穿越 -> 400 友好提示
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if rag_service.is_backend_unavailable(e):
            raise HTTPException(status_code=503, detail=_backend_error_detail(e))
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
    """删除文档：从 Milvus 清除该文件的全部分片，并移除上传文件。

    安全：文件名经 safe_upload_path 校验，拒绝路径穿越（../、..\\、绝对路径）。
    """
    try:
        return await rag_service.delete_file(filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Milvus 不可用时同样走 503 友好提示（避免裸 500；文件保留，重试可自愈）
        if rag_service.is_backend_unavailable(e):
            raise HTTPException(status_code=503, detail=_backend_error_detail(e))
        raise


@router.post("/query", response_model=QueryResponse, summary="RAG 问答（多轮）")
async def rag_query(
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """RAG 问答主接口。

    流程：校验会话归属 -> 读取最近 N 条历史（多轮记忆）-> 检索 Milvus -> 在线大模型生成
    -> 将用户问题与助手回答持久化到 messages 表 -> 返回答案与来源片段。
    """
    conv = await _get_own_conversation(db, body.conversation_id, current_user.id)

    history = await build_history(
        db, body.conversation_id, last_n=settings.MAX_CONTEXT_HISTORY
    )

    # 全异步链路（Milvus 检索 + 在线 LLM API），不阻塞事件循环
    try:
        result = await rag_service.rag_query(body.question, history, body.top_k)
    except Exception as e:
        if not rag_service.is_backend_unavailable(e):
            raise  # 非外部服务问题，保留 500 便于排查
        # 优雅降级：返回兜底答案并照常持久化。
        # memory 模块的 _INVALID_MARKERS 会把这些「请求失败」回答挡在多轮上下文外。
        result = {
            "answer": f"抱歉，请求失败：无法连接外部模型服务（{type(e).__name__}）。{_backend_error_detail(e)}",
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