# -*- coding: utf-8 -*-
"""Milvus 向量库封装（升级需求 #1：向量库替换成 Milvus）。

设计要点：
- 进程级单例 + 懒初始化：首次使用时才创建 Milvus 客户端，避免启动时强依赖
  Milvus 服务（本地开发用 Milvus Lite 单文件，Docker/生产连独立 Milvus）；
- 集合 schema：auto_id 主键 + 动态字段（metadata 直接落成可过滤字段），
  度量 COSINE + 向量 L2 归一化（见 embeddings.NormalizedEmbeddings），
  因此检索返回的距离即余弦相似度（越大越相关，-1~1）；
- 幂等写入：vectorize 前先按 source 过滤删除旧分片，避免重复入库；
- 全部方法提供 async 版本（aadd_texts / asimilarity_search_* / adelete），
  支持并发向量化与检索。
"""
import json
import os
import threading

from langchain_core.documents import Document
from langchain_milvus import Milvus

from app.core.config import settings
from app.services.embeddings import LazyEmbeddings, get_normalized_embeddings

class DimensionMismatchError(RuntimeError):
    """Embedding 维度与 Milvus collection 已建维度不一致（换模型后未重建）。"""


_store: Milvus | None = None
_store_lock = threading.Lock()


def _connection_args() -> dict:
    args: dict = {"uri": settings.MILVUS_URL}
    if settings.MILVUS_TOKEN:
        args["token"] = settings.MILVUS_TOKEN
    return args


def _ensure_local_dir(uri: str) -> None:
    """Milvus Lite 本地文件模式：确保父目录存在（langchain-milvus 构造即连接）。"""
    if uri.startswith(("http://", "https://")):
        return
    parent = os.path.dirname(uri)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _build_store() -> Milvus:
    """构建 Milvus 客户端（不连接、不建库，首次写入时自动建 collection）。"""
    _ensure_local_dir(settings.MILVUS_URL)
    return Milvus(
        # 懒解析：删除/健康检查等不涉及向量化的操作无需配置 Embedding Key
        embedding_function=LazyEmbeddings(get_normalized_embeddings),
        connection_args=_connection_args(),
        collection_name=settings.MILVUS_COLLECTION,
        auto_id=True,               # 主键自增
        drop_old=False,             # 重启不删库（保留已入库分片）
        enable_dynamic_field=True,  # metadata 落为动态字段，可按 source 过滤
        index_params={
            "index_type": "AUTOINDEX",          # Milvus Lite / 服务端自动索引
            "metric_type": settings.MILVUS_METRIC_TYPE,
            "params": {},
        },
    )


def get_vector_store() -> Milvus:
    """进程级单例：返回 Milvus 向量库（懒初始化）。"""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = _build_store()
    return _store


def init_vector_store() -> bool:
    """启动时初始化：确保本地目录存在、验证连接可用（失败不阻断启动）。

    Returns:
        True 表示连接/集合可用；False 表示 Milvus 暂不可用（首次使用时再报错）。
    """
    uri = settings.MILVUS_URL
    # Milvus Lite 本地文件模式：确保父目录存在
    if uri.startswith(("./", "../", "/", ".\\")) or not uri.startswith(("http://", "https://")):
        parent = os.path.dirname(uri)
        if parent:
            os.makedirs(parent, exist_ok=True)
    try:
        store = get_vector_store()
        # 轻量探测：连接并确认集合存在性（不存在也没关系，首次 add 会自动创建）
        store.client.has_collection(settings.MILVUS_COLLECTION)
        return True
    except Exception:
        return False


def reset_vector_store_for_test() -> None:
    """测试专用：重置单例与维度缓存。"""
    global _store, _expected_dim
    _store = None
    _expected_dim = None


# ---------- 业务操作 ----------
def source_expr(filename: str) -> str:
    """按 source 过滤的 Milvus 表达式（JSON 转义文件名中的引号）。"""
    return f'source == {json.dumps(filename)}'


def _collection_vector_dim(store: Milvus) -> int | None:
    """返回已建 collection 的向量维度；collection 不存在时返回 None。

    注意：describe_collection 的 type 是 pymilvus DataType 枚举（str() 后形如
    "FLOAT_VECTOR"），这里按名称兜底匹配，兼容枚举与字符串两种形式。
    """
    name = settings.MILVUS_COLLECTION
    if not store.client.has_collection(name):
        return None
    schema = store.client.describe_collection(name)
    for field in schema.get("fields", []):
        fname = str(field.get("name") or "")
        ftype = str(field.get("type") or "")
        if ftype.endswith("VECTOR") or fname in ("vector", "embedding"):
            params = field.get("params") or {}
            dim = params.get("dim")
            if dim is not None:
                return int(dim)
    return None


_expected_dim: int | None = None
_expected_dim_lock = threading.Lock()


async def _probe_embedding_dim(store: Milvus) -> int:
    """探测当前 Embedding 维度并进程级缓存（仅首次真实调用 API，之后复用）。"""
    global _expected_dim
    if _expected_dim is None:
        with _expected_dim_lock:
            if _expected_dim is None:
                _expected_dim = len(await store.embeddings.aembed_query("维度探测"))
    return _expected_dim


async def aadd_chunks(
    texts: list[str], metadatas: list[dict], rebuild: bool = False
) -> list[str]:
    """异步写入分片，返回本次写入的 pk 列表（供删除旧分片时排除）。

    Args:
        texts: 分片文本。
        metadatas: 分片元数据（须含 source）。
        rebuild: True 时重建 collection（适用于切换了不同维度的 Embedding 模型后迁移）。

    Raises:
        DimensionMismatchError: 已建 collection 维度与当前 Embedding 维度不一致且未传 rebuild。
    """
    store = get_vector_store()
    expected_dim = await _probe_embedding_dim(store)
    existing_dim = _collection_vector_dim(store)
    if existing_dim and existing_dim != expected_dim:
        if not rebuild:
            raise DimensionMismatchError(
                f"Embedding 维度已由 {existing_dim} 变为 {expected_dim}（可能切换了模型），"
                f"Milvus collection 维度固定无法写入。请重新向量化并携带 rebuild=true "
                f"（将重建 {settings.MILVUS_COLLECTION} 集合后写入），或删除本地 "
                f"{settings.MILVUS_URL} 后重跑。"
            )
        store.drop()
    ids = await store.aadd_texts(
        texts=texts, metadatas=metadatas, batch_size=settings.EMBEDDING_BATCH_SIZE
    )
    return [str(i) for i in ids]


async def adelete_by_source(filename: str) -> None:
    """异步删除某文件的全部分片（按动态字段 source 过滤）。"""
    store = get_vector_store()
    await store.adelete(expr=source_expr(filename))


async def adelete_old_chunks(filename: str, keep_ids: list[str]) -> None:
    """删除某文件的「旧」分片，保留 keep_ids（本次刚写入的新分片）。

    用于 vectorize 的「先写后删」写序：新分片与旧分片 source 相同，
    若直接按 source 删除会把刚写入的新分片一并删掉（严重回归）。
    这里用主键 not in 过滤，只清旧、不误删新。
    """
    store = get_vector_store()
    if not keep_ids:
        return
    ids_int = [int(i) for i in keep_ids if str(i).isdigit()]
    if not ids_int:
        return
    expr = f"{source_expr(filename)} and pk not in [{', '.join(map(str, ids_int))}]"
    await store.adelete(expr=expr)


async def asearch(query_vector: list[float], top_k: int) -> list[tuple[Document, float]]:
    """异步检索：返回 (文档, 相似度) 列表，相似度越大越相关（余弦相似度）。

    向量已在 NormalizedEmbeddings 中 L2 归一化，Milvus COSINE 返回的
    distance 即余弦相似度（-1~1），直接作为相关性分数。
    """
    store = get_vector_store()
    hits = await store.asimilarity_search_with_score_by_vector(
        query_vector, k=top_k
    )
    return hits


def relevance(score: float) -> float:
    """把 Milvus 返回的距离按度量类型归一为 0~1 相关性（越大越相关）。

    - COSINE / IP（向量已 L2 归一化时 IP == 余弦）：距离即相似度（-1~1），裁剪到 [0,1]；
    - L2：距离 ∈ [0, +∞) 且越小越近，用 1/(1+d) 单调映射到 (0, 1]；
    - 其它度量：原样返回（不保证 0~1，调用方需自行处理）。
    """
    metric = settings.MILVUS_METRIC_TYPE.strip().upper()
    value = float(score)
    if metric in ("COSINE", "IP"):
        return max(0.0, min(1.0, value))
    if metric == "L2":
        return 1.0 / (1.0 + value)
    return value