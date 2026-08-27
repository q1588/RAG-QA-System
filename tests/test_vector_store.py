# -*- coding: utf-8 -*-
"""Milvus 向量库单测（Milvus Lite 本地文件模式，确定性 HashEmbeddings）。

覆盖：写入/检索/相关性转换/按 source 过滤删除/幂等覆盖写入。
每个测试使用独立的 Milvus 数据库文件，并在单个事件循环内完成全部异步操作
（pymilvus 异步 gRPC 通道绑定事件循环，跨 loop 复用会报 "Event loop is closed"）。
"""
import asyncio
import re

import pytest

from app.core.config import settings
from app.services import vector_store
from app.services.embeddings import set_embeddings_for_test, reset_embeddings_for_test
from app.services.vector_store import DimensionMismatchError
from tests.helpers import make_hash_embeddings

DOCS = [
    ("苹果是红色的水果。", {"source": "fruit.txt"}),
    ("香蕉是黄色的水果。", {"source": "fruit.txt"}),
    ("汽车使用汽油引擎。", {"source": "car.txt"}),
    ("火箭使用液氢燃料。", {"source": "car.txt"}),
]


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """每个测试：独立 Milvus 文件 + 确定性 embedding + 重置单例。"""
    monkeypatch.setattr(settings, "MILVUS_URL", str(tmp_path / "test.db"))
    set_embeddings_for_test(make_hash_embeddings())
    vector_store.reset_vector_store_for_test()
    yield
    vector_store.reset_vector_store_for_test()
    reset_embeddings_for_test()


async def _count() -> int:
    """Milvus Lite 的 count 查询结果以字符串 dict 形式返回，用正则提取。"""
    store = vector_store.get_vector_store()
    res = await store.aclient.query(
        collection_name=settings.MILVUS_COLLECTION,
        filter="",
        output_fields=["count(*)"],
    )
    m = re.search(r"count\(\*\)': (\d+)", str(res))
    return int(m.group(1)) if m else -1


class TestStoreOps:
    def test_add_search_relevance(self):
        emb = make_hash_embeddings()

        async def body():
            await vector_store.aadd_chunks([t for t, _ in DOCS], [m for _, m in DOCS])
            assert await _count() == 4
            query_vec = await emb.aembed_query("水果")
            return await vector_store.asearch(query_vec, top_k=4)

        hits = asyncio.run(body())
        assert len(hits) == 4
        # 水果类文档应排在前列（语义信号：共享 token「水果」）
        top_sources = [d.metadata.get("source") for d, _ in hits]
        assert top_sources.count("fruit.txt") >= 1
        assert top_sources.index("fruit.txt") < top_sources.index("car.txt")
        # 相关性在 [0,1] 且单调递减
        rels = [vector_store.relevance(s) for _, s in hits]
        assert all(0.0 <= r <= 1.0 for r in rels)
        assert rels == sorted(rels, reverse=True)

    def test_delete_by_source(self):
        async def body():
            await vector_store.aadd_chunks([t for t, _ in DOCS], [m for _, m in DOCS])
            await vector_store.adelete_by_source("car.txt")
            return await _count()

        assert asyncio.run(body()) == 2  # 只剩 fruit.txt 的 2 条

    def test_reindex_idempotent(self):
        """同名文档重复向量化（先删后写）：不累积重复分片。"""

        async def body():
            await vector_store.aadd_chunks([t for t, _ in DOCS], [m for _, m in DOCS])
            # 模拟 vectorize_file 的幂等流程：先按 source 删旧分片，再写新分片
            await vector_store.adelete_by_source("fruit.txt")
            await vector_store.aadd_chunks(["苹果是红色的水果。"], [{"source": "fruit.txt"}])
            return await _count()

        assert asyncio.run(body()) == 3  # car.txt 2 条 + fruit.txt 覆盖后 1 条

    def test_relevance_metric_aware(self, monkeypatch):
        # COSINE/IP：距离即相似度，裁剪到 [0,1]
        monkeypatch.setattr(settings, "MILVUS_METRIC_TYPE", "COSINE")
        assert vector_store.relevance(-0.5) == 0.0
        assert vector_store.relevance(0.8) == 0.8
        assert vector_store.relevance(1.5) == 1.0
        # L2：距离越小越近，1/(1+d) 单调映射到 (0,1]
        monkeypatch.setattr(settings, "MILVUS_METRIC_TYPE", "L2")
        assert vector_store.relevance(0.0) == 1.0
        assert 0.0 < vector_store.relevance(1.0) < 1.0
        assert vector_store.relevance(9.0) < vector_store.relevance(1.0)

    def test_dimension_mismatch_requires_rebuild(self):
        """切换不同维度 Embedding 后：默认报错，rebuild=True 重建集合可写入。"""

        async def body():
            # 阶段 1：128 维建库
            await vector_store.aadd_chunks(["苹果是水果"], [{"source": "a.txt"}])
            assert await _count() == 1
            # 阶段 2：切换为 64 维 Embedding 并重置单例
            set_embeddings_for_test(make_hash_embeddings(size=64))
            vector_store.reset_vector_store_for_test()
            with pytest.raises(DimensionMismatchError):
                await vector_store.aadd_chunks(["香蕉是水果"], [{"source": "b.txt"}])
            # rebuild=True：重建集合（旧分片全部清空，文件仍在 upload 目录，可重新向量化）
            await vector_store.aadd_chunks(["香蕉是水果"], [{"source": "b.txt"}], rebuild=True)
            return await _count()

        assert asyncio.run(body()) == 1  # 重建后只剩本次写入的 1 条

    def test_adelete_old_chunks_keeps_new(self):
        """严重回归防护：先写后删时，adelete_old_chunks 只删旧分片、保留新分片。"""

        async def body():
            await vector_store.aadd_chunks(
                ["旧分片A", "旧分片B"], [{"source": "a.txt"}, {"source": "a.txt"}]
            )
            await vector_store.aadd_chunks(
                ["旧分片C"], [{"source": "b.txt"}]
            )
            assert await _count() == 3
            # 模拟 vectorize_file 写序：写入新分片（同 source）后再删旧
            new_ids = await vector_store.aadd_chunks(
                ["新分片"], [{"source": "a.txt"}]
            )
            assert await _count() == 4
            await vector_store.adelete_old_chunks("a.txt", new_ids)
            return await _count()

        # b.txt 1 条 + a.txt 新分片 1 条 = 2（旧 a.txt 2 条被清）
        assert asyncio.run(body()) == 2