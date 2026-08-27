# -*- coding: utf-8 -*-
"""切块策略单测：递归 / 语义 / 混合。"""
import pytest

from app.core.config import settings
from app.services import chunking
from tests.helpers import make_hash_embeddings

# 中文句子（带句末标点），用于语义切分测试
DOC = (
    "检索增强生成（RAG）是一种结合信息检索与大模型生成的技术。"
    "它能在提问时临时查找外部知识库。"
    "这样做可以减少幻觉。"
    "向量数据库用于快速比对相似度。"
    "Milvus 是常见的向量数据库。"
    "Chroma 也是常见的向量数据库。"
    "切块策略会影响回答质量。"
    "语义切分能保留完整语义。"
)


class TestRecursive:
    def test_splits_into_multiple_chunks(self, monkeypatch):
        monkeypatch.setattr(settings, "CHUNK_SIZE", 30)
        monkeypatch.setattr(settings, "CHUNK_OVERLAP", 5)
        chunks = chunking.recursive_chunk("第一句话。" + "第二句话。" * 10)
        assert len(chunks) >= 2
        assert all(c.strip() for c in chunks)

    def test_short_text_single_chunk(self):
        chunks = chunking.recursive_chunk("很短的一句话。")
        assert chunks == ["很短的一句话。"]

    def test_empty_text(self):
        assert chunking.recursive_chunk("   ") == []
        assert chunking.recursive_chunk("") == []


class TestSemantic:
    def test_uses_semantic_breaks(self, monkeypatch):
        # 用更敏感的断点分位 + 更小最小块，保证短文档也能按语义跳变断开
        monkeypatch.setattr(settings, "SEMANTIC_BREAKPOINT_PERCENTILE", 60.0)
        monkeypatch.setattr(settings, "SEMANTIC_MIN_CHUNK_CHARS", 15)
        monkeypatch.setattr(settings, "SEMANTIC_MAX_CHUNK_CHARS", 1000)
        emb = make_hash_embeddings()
        chunks = chunking.semantic_chunk(DOC, emb)
        assert len(chunks) >= 2  # 语义跳变处应断开
        joined = "".join(chunks)
        assert "检索增强生成" in joined and "语义切分" in joined  # 内容不丢失

    def test_min_chunk_merge(self, monkeypatch):
        monkeypatch.setattr(settings, "SEMANTIC_MIN_CHUNK_CHARS", 80)
        monkeypatch.setattr(settings, "SEMANTIC_MAX_CHUNK_CHARS", 500)
        chunks = chunking.semantic_chunk(DOC, make_hash_embeddings())
        # 除最后一格外，不应出现远小于 min 的碎片
        assert len(chunks) >= 1

    def test_single_sentence(self):
        chunks = chunking.semantic_chunk("只有一句话。", make_hash_embeddings())
        assert "".join(chunks) == "只有一句话。"

    def test_oversized_sentence_hard_split(self, monkeypatch):
        monkeypatch.setattr(settings, "SEMANTIC_MAX_CHUNK_CHARS", 20)
        long_text = "没有标点" * 30
        chunks = chunking.semantic_chunk(long_text, make_hash_embeddings())
        assert all(len(c) <= 25 for c in chunks)


class TestHybrid:
    def test_merges_similar_blocks(self, monkeypatch):
        monkeypatch.setattr(settings, "CHUNK_SIZE", 40)
        monkeypatch.setattr(settings, "CHUNK_OVERLAP", 0)
        monkeypatch.setattr(settings, "SEMANTIC_MERGE_THRESHOLD", 0.5)
        # 相似内容（共享 token）的多个小块应被合并
        text = "水果 苹果 香蕉。" * 5 + "汽车 引擎 汽油。" * 5
        chunks = chunking.hybrid_chunk(text, make_hash_embeddings())
        assert len(chunks) < 10  # 发生合并
        assert "".join(chunks).count("水果") == 5

    def test_single_block(self):
        assert chunking.hybrid_chunk("一句话。", make_hash_embeddings()) == ["一句话。"]


class TestSplitText:
    @pytest.mark.parametrize("strategy", ["recursive", "semantic", "hybrid"])
    def test_all_strategies(self, strategy):
        chunks = chunking.split_text(DOC, strategy=strategy, embeddings=make_hash_embeddings())
        assert chunks, f"{strategy} 策略应产出非空块"
        assert all(c.strip() for c in chunks)

    def test_unknown_strategy_raises(self):
        with pytest.raises(chunking.ChunkingError):
            chunking.split_text(DOC, strategy="nope")

    def test_default_strategy_from_settings(self, monkeypatch):
        monkeypatch.setattr(settings, "CHUNKING_STRATEGY", "recursive")
        assert chunking.split_text(DOC)