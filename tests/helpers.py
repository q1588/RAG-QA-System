# -*- coding: utf-8 -*-
"""测试辅助：确定性、有语义信号的 HashEmbeddings（同一文本 -> 同一向量）。"""
import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    """中文按单字切 token，英文按单词切 token（小写）。"""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class HashEmbeddings(Embeddings):
    """基于 token 哈希叠加的确定性 Embedding（测试/离线评估用）。

    语义信号：共享 token 越多的文本，向量余弦相似度越高，
    因此可以真实驱动「检索」逻辑，且完全离线、结果可复现。
    """

    def __init__(self, size: int = 128, seed: int = 42):
        self.size = size
        self._seed = seed

    def _token_vec(self, token: str) -> list[float]:
        # 每 64 维用独立哈希派生位，避免 h >> (i*11) 在高维时恒 0 导致常数尾部
        vec: list[float] = []
        for chunk in range(0, self.size, 64):
            h = int(
                hashlib.sha256(f"{self._seed}:{token}:{chunk // 64}".encode("utf-8")).hexdigest(),
                16,
            )
            for j in range(min(64, self.size - chunk)):
                vec.append(1.0 if (h >> j) & 1 else -1.0)
        return vec

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.size
        for t in _tokenize(text):
            tv = self._token_vec(t)
            vec = [v + x for v, x in zip(vec, tv)]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def make_hash_embeddings(size: int = 128) -> HashEmbeddings:
    return HashEmbeddings(size=size)