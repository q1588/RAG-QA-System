# -*- coding: utf-8 -*-
"""RAG 核心服务：文档加载 -> 切分 -> 向量化 -> 检索 -> 生成。

说明：
- 向量库/大模型均为进程级单例，启动时初始化一次。
- Ollama 的 embedding / 生成是同步阻塞调用，接口层通过 asyncio.to_thread 放进线程池执行。
- 用 threading.Lock 保护 Chroma 单例，避免并发访问向量库时的竞争。
"""
import os
import threading

import httpx
from langchain_chroma import Chroma
from ollama._types import ResponseError  # langchain_ollama 底层依赖 ollama 包
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

_chroma_lock = threading.Lock()


def is_backend_unavailable(exc: BaseException) -> bool:
    """判断异常是否为 Ollama 服务不可用（连接失败/超时/HTTP 层错误）。

    用于在接口层做优雅降级：Ollama 未启动或网络不通时，
    返回友好的兜底答案/503，而不是裸奔的 500。
    """
    return isinstance(
        exc, (httpx.HTTPError, ConnectionError, TimeoutError, ResponseError)
    )

# ---------- 大模型与向量库单例（与旧 main.py 一致的已验证用法）----------
embeddings = OllamaEmbeddings(
    model=settings.EMBED_MODEL, base_url=settings.OLLAMA_BASE_URL
)
llm = ChatOllama(
    model=settings.LLM_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=0.3
)
vector_db = Chroma(
    persist_directory=settings.CHROMA_DIR, embedding_function=embeddings
)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
)

# ---------- 生成提示词 ----------
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是知识库问答助手。只能依据【知识库】里提供的原文回答用户问题，不得使用知识库之外的知识；"
            "如果知识库内容与问题无关，或你无法从知识库找到依据，必须直接回答「知识库中没有相关信息」；"
            "严禁编造，严禁列举知识库中不存在的领域或主题。回答要准确、简洁。",
        ),
        (
            "human",
            "【历史对话】\n{history}\n\n"
            "【知识库】\n{context}\n\n"
            "【问题】\n{question}\n\n"
            "请根据知识库内容给出回答。",
        ),
    ]
)
rag_chain = PROMPT | llm | StrOutputParser()


# ---------- 文档加载 ----------
def load_document(path: str) -> str:
    """按扩展名读取文档内容，返回纯文本。支持 txt/md/pdf/docx。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        return _read_text_file(path)
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if ext == ".docx":
        from docx import Document as DocxDocument

        doc = DocxDocument(path)
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"不支持的文件类型：{ext}（仅支持 txt/md/pdf/docx）")


def _read_text_file(path: str) -> str:
    """兼容 utf-8 / gbk 编码的文本读取。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="gbk") as f:
            return f.read()


# ---------- 向量化 ----------
def vectorize_file(filename: str) -> dict:
    """读取文档 -> 切分 -> 写入 Chroma。返回切片数量。"""
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{filename}")

    content = load_document(file_path)
    chunks = text_splitter.split_text(content)
    if not chunks:
        raise ValueError("文档内容为空或未能提取到文本")

    metadatas = [{"source": filename}] * len(chunks)
    with _chroma_lock:
        # 幂等：先清同名旧分片再入库，避免重传/修改同一文档后向量库累积重复片段
        vector_db.delete(where={"source": filename})
        vector_db.add_texts(texts=chunks, metadatas=metadatas)
    return {"filename": filename, "chunk_count": len(chunks)}


# ---------- 删除文档 ----------
def delete_file(filename: str) -> dict:
    """删除文档：从 Chroma 清除该文件的所有分片，并移除上传目录下的文件。"""
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    with _chroma_lock:
        vector_db.delete(where={"source": filename})
    if os.path.isfile(file_path):
        os.remove(file_path)
    return {"deleted": filename}


# ---------- 检索 + 生成 ----------
def rag_query(question: str, history: list[dict[str, str]], top_k: int = 3) -> dict:
    """执行一次 RAG 问答：相关性过滤 -> 组装上下文与历史 -> 大模型生成。

    similarity_search_with_relevance_scores 返回 0~1 的相关性（越大越相关）；
    低于 RAG_RELEVANCE_THRESHOLD 的弱匹配视为「知识库无相关内容」，
    此时不调用大模型、直接诚实回答，杜绝模型编造不存在的内容。
    """
    with _chroma_lock:
        hits = vector_db.similarity_search_with_relevance_scores(question, k=top_k * 2)

    # 门槛判定：以最佳分片的相关性是否达标为准（决定"知识库有没有相关内容"）；
    # 达标后喂完整 top_k 上下文，避免过度过滤把有用分片也丢掉。
    if not hits or hits[0][1] < settings.RAG_RELEVANCE_THRESHOLD:
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

    history_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}" for m in history
    )

    answer = rag_chain.invoke(
        {"history": history_text, "context": context, "question": question}
    ).strip()

    source_docs = [
        {
            "index": i + 1,
            "source": d.metadata.get("source", "unknown"),
            "relevance": round(score, 4),
            "content": d.page_content[:200],
        }
        for i, (d, score) in enumerate(relevant)
    ]
    return {"answer": answer, "source_docs": source_docs}
