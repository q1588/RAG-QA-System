# -*- coding: utf-8 -*-
"""HashEmbeddings 质量测试：锁定修复后的分布性质。

回归：旧实现 h >> (i*11) 在 i>=24 时恒 0，导致 1024 维中 1000 维为常数 -1，
语义无关文本余弦相似度 ≈ 0.9947（失真）。修复后应满足：
- 无常数尾部（尾部维度同时含正负分量）；
- 语义无关文本余弦接近 0（阈值兜底分支可触发）；
- 共享 token 的文本余弦显著更高（保留语义信号）；
- 同文本确定性（可复现）。
"""
import sys

sys.path.insert(0, ".")  # noqa: E402  （pytest 工作目录下可直接导入 tests）

from tests.helpers import make_hash_embeddings


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class TestHashEmbeddings:
    def test_no_constant_tail(self):
        emb = make_hash_embeddings(size=256)
        v = emb.embed_query("苹果 水果 红色")
        # 旧实现尾部（i>=24 之后）全为 -1；现在应同时含正负分量
        assert len(set(v[200:])) > 1
        assert any(x > 0 for x in v) and any(x < 0 for x in v)

    def test_unrelated_cosine_low(self):
        emb = make_hash_embeddings(size=256)
        a = emb.embed_query("苹果 水果 红色")
        b = emb.embed_query("汽车 引擎 汽油 火箭 燃料 数据库 网络 服务器")
        # 旧实现 ≈0.9947；修复后应远低于 0.5（否则相关性阈值分支永不触发）
        assert _cos(a, b) < 0.3

    def test_related_cosine_higher(self):
        emb = make_hash_embeddings(size=256)
        a = emb.embed_query("苹果 水果 红色")
        b = emb.embed_query("香蕉 水果 黄色")
        c = emb.embed_query("汽车 引擎 汽油")
        assert _cos(a, b) > _cos(a, c)

    def test_deterministic(self):
        emb = make_hash_embeddings(size=128)
        assert emb.embed_query("同一段文本") == emb.embed_query("同一段文本")