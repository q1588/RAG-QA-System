# -*- coding: utf-8 -*-
"""Embedding 模型工厂：在线 API（OpenAI 兼容协议）为主，本地 Ollama 为兜底。

设计要点（对应升级需求 #2：Embedding 换成效果更好的在线/更大参数模型）：
- 默认走在线 API：OpenAIEmbeddings 支持任意 OpenAI 兼容端点
  （SiliconFlow / OpenAI 官方 / 阿里云百炼 / DeepSeek / Moonshot 等），
  默认模型 BAAI/bge-m3（SiliconFlow，1024 维），可换 text-embedding-3-large 等大参数模型；
- 未配置 API Key 时自动降级为本地 Ollama（EMBEDDING_PROVIDER=ollama 或手动指定），
  保证离线开发环境也能跑通全链路；
- NormalizedEmbeddings：对输出向量做 L2 归一化，使 Milvus COSINE 距离 = 余弦相似度，
  检索阶段的相关性阈值语义清晰（0~1，越大越相关）。
"""
import math
import threading

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


class ProviderConfigError(RuntimeError):
    """外部模型服务配置缺失/错误（如未配置 API Key）。接口层捕获后返回 503 友好提示。"""


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


class NormalizedEmbeddings(Embeddings):
    """包装任意 Embeddings，输出前做 L2 归一化。

    Milvus 的 COSINE 度量在向量为单位长度时返回的「距离」就是余弦相似度（-1~1），
    归一化后检索返回的 score 可直接作为相关性使用（越大越相关）。
    """

    def __init__(self, inner: Embeddings):
        self._inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_l2_normalize(v) for v in self._inner.embed_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        return _l2_normalize(self._inner.embed_query(text))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        raw = await self._inner.aembed_documents(texts)
        return [_l2_normalize(v) for v in raw]

    async def aembed_query(self, text: str) -> list[float]:
        raw = await self._inner.aembed_query(text)
        return _l2_normalize(raw)


def _build_openai_embeddings() -> Embeddings:
    if not settings.EMBEDDING_API_KEY:
        raise ProviderConfigError(
            "未配置 EMBEDDING_API_KEY：无法使用在线 Embedding 服务。"
            "请在 .env 中填写密钥（如 SiliconFlow / OpenAI），"
            "或设置 EMBEDDING_PROVIDER=ollama 使用本地 Ollama。"
        )
    return OpenAIEmbeddings(
        model=settings.EMBED_MODEL,
        api_key=settings.EMBEDDING_API_KEY,
        base_url=settings.EMBEDDING_BASE_URL,
        chunk_size=settings.EMBEDDING_BATCH_SIZE,
        # 非 OpenAI 官方提供商（百炼/SiliconFlow/DeepSeek 等兼容端点）必须关闭：
        # 默认 True 会用 tiktoken 分词并对空字符串 input 做探测调用，
        # 部分提供商（如百炼 text-embedding-v3）会拒绝空输入返回 400。
        check_embedding_ctx_length=False,
        max_retries=2,
        request_timeout=60,
    )


def _build_ollama_embeddings() -> Embeddings:
    # 延迟导入：仅在需要时才引入 langchain_ollama，避免在线部署时产生无关依赖
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.OLLAMA_EMBED_MODEL, base_url=settings.OLLAMA_BASE_URL
    )


class LazyEmbeddings(Embeddings):
    """延迟解析的 Embeddings 包装：构造时不连接、不校验配置。

    用途：向量库对象在构造时需要传入 embedding_function，但删除分片等
    操作根本不涉及向量化 —— 用本类包装后，未配置 Embedding API Key 也能
    正常执行删除/健康检查；首次真正向量化/检索时才解析并报错。
    """

    def __init__(self, factory):
        self._factory = factory
        self._inner: Embeddings | None = None
        self._lock = threading.Lock()

    def _get(self) -> Embeddings:
        if self._inner is None:
            with self._lock:
                if self._inner is None:
                    self._inner = self._factory()
        return self._inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._get().embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._get().aembed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await self._get().aembed_query(text)


_embeddings: Embeddings | None = None
_embeddings_lock = threading.Lock()


def get_embeddings() -> Embeddings:
    """进程级单例。测试可通过 set_embeddings_for_test 注入确定性实现。"""
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                provider = settings.EMBEDDING_PROVIDER.strip().lower()
                if provider == "ollama":
                    _embeddings = _build_ollama_embeddings()
                elif provider == "openai":
                    _embeddings = _build_openai_embeddings()
                else:
                    raise ProviderConfigError(
                        f"未知的 EMBEDDING_PROVIDER：{provider!r}（可选 openai / ollama）"
                    )
    return _embeddings


def get_normalized_embeddings() -> Embeddings:
    """向量库/检索统一使用归一化后的 embedding（保证 COSINE 距离即相似度）。"""
    return NormalizedEmbeddings(get_embeddings())


def set_embeddings_for_test(embeddings: Embeddings) -> None:
    """测试专用：覆盖全局 embedding 单例（如 DeterministicFakeEmbedding）。"""
    global _embeddings
    _embeddings = embeddings


def reset_embeddings_for_test() -> None:
    global _embeddings
    _embeddings = None