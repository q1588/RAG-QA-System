# -*- coding: utf-8 -*-
"""RAG 相关请求/响应模型。"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class VectorizeRequest(BaseModel):
    """对已上传的文件做向量化。"""

    filename: str = Field(description="已上传到 upload_files 的文件名")
    strategy: Optional[Literal["recursive", "semantic", "hybrid"]] = Field(
        default=None,
        description="切块策略；不传则使用配置 CHUNKING_STRATEGY",
    )
    rebuild: bool = Field(
        default=False,
        description="切换了不同维度 Embedding 模型后置 true：重建向量集合再写入",
    )


class QueryRequest(BaseModel):
    """RAG 问答请求：携带会话 id 以支持多轮记忆。"""

    conversation_id: int
    question: str = Field(min_length=1, description="用户问题")
    top_k: int = Field(default=3, ge=1, le=10, description="检索返回片段数")


class SourceDoc(BaseModel):
    """检索命中的知识片段（用于展示答案来源，体现 RAG 可解释性）。"""

    index: int
    source: str = ""
    relevance: float = 0.0  # 相关性 0~1，越大越相关
    content: str = ""


class QueryResponse(BaseModel):
    conversation_id: int
    question: str
    answer: str
    top_k: int
    source_docs: list[SourceDoc]