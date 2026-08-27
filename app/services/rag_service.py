# -*- coding: utf-8 -*-
"""RAG 核心服务（异步版）：文档加载 -> 切块（递归/语义/混合）-> 向量化 -> 检索 -> 生成。

相对旧版（Chroma + Ollama 同步调用）的升级：
1. 向量库：Milvus（Milvus Lite 本地文件 / 独立服务），见 app.services.vector_store；
2. Embedding：在线大参数模型（OpenAI 兼容 API，默认 BAAI/bge-m3），见 embeddings.py；
3. 切块：递归 + 语义 + 混合三种策略，文档解析逐页/逐段容错，见 chunking.py / document_loader.py；
4. 生成：在线 LLM API（OpenAI 兼容协议），异步 ainvoke 调用，天然支持高并发，见 llm.py。

所有外部调用（embedding / 向量检索 / LLM 生成）均为 async，不再占用
asyncio.to_thread 阻塞线程池；文档解析（PDF/docx）为 CPU/IO 密集操作，
仍在 to_thread 中执行，避免阻塞事件循环。
"""
import asyncio
import os
import re

import grpc
import httpx
from pymilvus.exceptions import (
    ConnectError,
    ConnectionConfigException,
    MilvusException,
    MilvusUnavailableException,
)

from app.core.config import settings
from app.services import chunking, vector_store
from app.services.document_loader import DocumentParseError, load_document
from app.services.embeddings import ProviderConfigError, get_embeddings
from app.services.llm import get_rag_chain

# 外部后端不可用的异常类型集合：接口层据此做 503 优雅降级
_BACKEND_ERRORS = (
    httpx.HTTPError,
    ConnectionError,
    TimeoutError,
    ProviderConfigError,
    DocumentParseError,
)

# Milvus 后端异常：含基类 MilvusException（实证：连接失败抛基类 code=2
# "Fail connecting to server..."），并覆盖连接类子类与 gRPC 网络层兜底。
_MILVUS_BACKEND_ERRORS = (
    MilvusException,               # 基类：连接失败/服务不可用（实证场景）
    ConnectionConfigException,     # 连接串/本地文件不可用（占用/损坏/目录缺失）
    MilvusUnavailableException,    # Milvus 服务未启动/不可达
    ConnectError,                  # 连接建立失败
    grpc.RpcError,                 # gRPC 网络层错误（兜底）
)


def is_milvus_error(exc: BaseException) -> bool:
    """是否为 Milvus 后端不可用（用于接口层给出专属 503 文案）。"""
    return isinstance(exc, _MILVUS_BACKEND_ERRORS)


def is_backend_unavailable(exc: BaseException) -> bool:
    """判断异常是否为外部服务不可用（连接失败/超时/HTTP 层错误/配置缺失）。

    覆盖：在线 Embedding/LLM API（httpx/OpenAI/Ollama）、向量数据库（Milvus）、
    配置缺失（ProviderConfigError）、文档解析失败。
    用于在接口层做优雅降级：外部服务不可用时返回友好的兜底答案/503，而不是裸奔的 500。
    """
    if isinstance(exc, _BACKEND_ERRORS) or is_milvus_error(exc):
        return True
    try:
        from openai import OpenAIError
        from ollama._types import ResponseError
    except Exception:
        return False
    return isinstance(exc, (OpenAIError, ResponseError))


def _history_to_text(history: list[dict[str, str]]) -> str:
    """把 [{role, content}] 历史转成纯文本上下文（用户/助手标签）。"""
    return "\n".join(
        f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')}"
        for m in history
    )

def safe_upload_path(filename: str) -> str:
    """把请求中的文件名安全拼接到 UPLOAD_DIR，拒绝路径穿越。

    防护（防 `../`、`..\\`、`..%2F`、绝对路径、空名等）：
    1. 文件名必须是单一组件：不允许包含 / 或 \\ 分隔符，也不允许 . / .. 组件；
    2. realpath 解析后必须仍在 UPLOAD_DIR 的 realpath 内（双保险，防符号链接逃逸）；
    3. 非法名称抛 ValueError，接口层返回 400。
    """
    raw = (filename or "").strip()
    if not raw:
        raise ValueError("非法的文件名：不允许路径穿越")
    parts = re.split(r"[/\\]", raw)
    if len(parts) != 1 or parts[0] in (".", ".."):
        raise ValueError("非法的文件名：不允许路径穿越")
    name = parts[0]
    upload_dir = os.path.realpath(settings.UPLOAD_DIR)
    candidate = os.path.realpath(os.path.join(upload_dir, name))
    if os.path.commonpath([upload_dir, candidate]) != upload_dir:
        raise ValueError("非法的文件名：不允许路径穿越")
    return candidate




# ---------- 向量化 ----------
async def vectorize_file(
    filename: str, strategy: str | None = None, rebuild: bool = False
) -> dict:
    """读取文档 -> 切块（可选策略）-> 向量化写入 Milvus（幂等）。

    Args:
        filename: 上传目录内的文件名（自动校验路径穿越）。
        strategy: recursive / semantic / hybrid，None 取配置。
        rebuild: 切换不同维度 Embedding 后置 True，重建向量集合再写入。

    Returns:
        {"filename", "chunk_count", "strategy"}
    """
    file_path = safe_upload_path(filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{filename}")

    # 文档解析（CPU/IO 密集）在线程池执行
    content = await asyncio.to_thread(load_document, file_path)
    if not content or not content.strip():
        raise ValueError("文档内容为空或未能提取到文本")

    strategy = (strategy or settings.CHUNKING_STRATEGY).strip().lower()
    # 切块（semantic/hybrid 需要向量化句子，也在线程池执行）
    chunks = await asyncio.to_thread(
        chunking.split_text, content, strategy, get_embeddings()
    )
    if not chunks:
        raise ValueError("文档切块结果为空")

    metadatas = [
        {
            "source": filename,
            "chunk_index": i,
            "strategy": strategy,
        }
        for i in range(len(chunks))
    ]

    # 幂等写入：先写新分片、成功后删旧分片。
    # 关键：新分片与旧分片 source 相同，直接按 source 删会把刚写入的新分片一起删掉
    # （严重回归），因此删除时必须排除本次写入的 pk（adelete_old_chunks）。
    # 失败窗口权衡：add 失败时旧分片完好；删旧失败仅短暂新旧并存，下次覆盖可自愈。
    ids = await vector_store.aadd_chunks(chunks, metadatas, rebuild=rebuild)
    if not rebuild:
        await vector_store.adelete_old_chunks(filename, ids)
    return {"filename": filename, "chunk_count": len(chunks), "strategy": strategy}


# ---------- 删除文档 ----------
async def adelete_vectors_only(filename: str) -> None:
    """仅清除某文件的向量分片（不改动上传目录文件）。供同名覆盖上传前清理。"""
    file_path = safe_upload_path(filename)
    await vector_store.adelete_by_source(os.path.basename(file_path))


async def delete_file(filename: str) -> dict:
    """删除文档：从 Milvus 清除该文件的所有分片，并移除上传目录下的文件。"""
    file_path = safe_upload_path(filename)
    await vector_store.adelete_by_source(os.path.basename(file_path))
    if os.path.isfile(file_path):
        os.remove(file_path)
    return {"deleted": os.path.basename(file_path)}


# ---------- 检索 + 生成 ----------
async def rag_query(
    question: str, history: list[dict[str, str]], top_k: int | None = None
) -> dict:
    """执行一次 RAG 问答（全异步）：相关性过滤 -> 组装上下文与历史 -> 大模型生成。

    Args:
        question: 用户问题。
        history: 多轮历史 [{role, content}, ...]。
        top_k: 检索返回片段数，None 取 settings.TOP_K。

    Returns:
        {"answer", "source_docs": [{"index", "source", "relevance", "content"}, ...]}
    """
    top_k = top_k or settings.TOP_K
    store = vector_store.get_vector_store()

    # 1. 问题向量化（归一化，与库内向量同空间）+ 向量检索（异步，不阻塞事件循环）
    #    store.embeddings 即 NormalizedEmbeddings：保证 COSINE 距离 = 余弦相似度
    query_vector = await store.embeddings.aembed_query(question)
    hits = await vector_store.asearch(query_vector, top_k=top_k * 2)

    # 2. 门槛判定：以最佳分片的相关性是否达标为准（决定"知识库有没有相关内容"）；
    #    达标后喂完整 top_k 上下文，避免过度过滤把有用分片也丢掉。
    if not hits or vector_store.relevance(hits[0][1]) < settings.RAG_RELEVANCE_THRESHOLD:
        # 知识库没有相关内容：确定性兜底，不经过大模型（避免幻觉/编造领域）
        return {
            "answer": (
                f"抱歉，知识库中暂时没有与「{question}」相关的内容，我无法回答。"
                "你可以先在左侧「上传文档」添加相关材料后再提问。"
            ),
            "source_docs": [],
        }

    relevant = hits[:top_k]
    docs = [doc for doc, _ in relevant]
    context = "\n\n".join(d.page_content for d in docs)

    # 3. 组装提示词：历史 + 知识库 + 问题，异步调用在线 LLM API
    chain = get_rag_chain()
    answer = (
        await chain.ainvoke(
            {
                "history": _history_to_text(history),
                "context": context,
                "question": question,
            }
        )
    ).strip()

    source_docs = [
        {
            "index": i + 1,
            "source": d.metadata.get("source", "unknown"),
            "relevance": round(vector_store.relevance(score), 4),
            "content": d.page_content[:200],
        }
        for i, (d, score) in enumerate(relevant)
    ]
    return {"answer": answer, "source_docs": source_docs}