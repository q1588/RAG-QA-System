# -*- coding: utf-8 -*-
"""RAG 服务层单测：路径穿越防护（严重漏洞回归测试）。

覆盖：safe_upload_path 对 ../ 、..\\ 、绝对路径、空名的拒绝；
vectorize_file / delete_file 在解析文件前必须先通过安全校验。
"""
import asyncio
import os

import grpc
import pytest

from app.core.config import settings
from app.services import rag_service, vector_store


@pytest.fixture(autouse=True)
def _upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "upload"))
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)


class TestSafeUploadPath:
    @pytest.mark.parametrize(
        "bad",
        [
            "../probe.md",          # Unix 穿越
            "..\\probe.md",       # Windows 穿越（%5C 解码后）
            "sub/rag.txt",          # 含目录分隔符
            "C:\\Windows\\win.ini",  # 绝对路径（Windows）
            "/etc/passwd",          # 绝对路径（Unix）
            "",
            ".",
            "..",
        ],
    )
    def test_rejects_traversal(self, bad):
        with pytest.raises(ValueError):
            rag_service.safe_upload_path(bad)

    def test_accepts_plain_name(self):
        p = rag_service.safe_upload_path("rag.txt")
        assert os.path.dirname(p) == os.path.realpath(settings.UPLOAD_DIR)
        assert os.path.basename(p) == "rag.txt"

    def test_accepts_name_with_double_dot_inside(self):
        # ".." 作为文件名一部分（非路径组件）应被允许
        p = rag_service.safe_upload_path("报告..最终版.md")
        assert os.path.basename(p) == "报告..最终版.md"


class TestVectorizeTraversal:
    def test_refuses_before_reading_file(self, monkeypatch):
        """回归：../probe.md 必须在文件被读取之前被拒绝。"""
        outside = os.path.join(os.path.dirname(settings.UPLOAD_DIR), "probe.md")
        with open(outside, "w", encoding="utf-8") as f:
            f.write("敏感内容：仅用于验证文件不会被读取。")

        def _assert_not_called(path):
            raise AssertionError(f"load_document 不应被调用（路径穿越）：{path}")

        monkeypatch.setattr(rag_service, "load_document", _assert_not_called)

        with pytest.raises(ValueError, match="路径穿越"):
            asyncio.run(rag_service.vectorize_file("../probe.md"))

    def test_missing_file_404(self):
        with pytest.raises(FileNotFoundError):
            asyncio.run(rag_service.vectorize_file("不存在.md"))


class TestDeleteTraversal:
    def test_refuses_and_keeps_outside_file(self):
        """回归：DELETE ../probe.txt 不得删除 upload 目录之外的文件。"""
        outside = os.path.join(os.path.dirname(settings.UPLOAD_DIR), "probe.txt")
        with open(outside, "w", encoding="utf-8") as f:
            f.write("不能被删除")

        with pytest.raises(ValueError, match="路径穿越"):
            asyncio.run(rag_service.delete_file("../probe.txt"))

        assert os.path.isfile(outside)  # 外部文件原样保留

class TestVectorizeWriteOrder:
    """严重回归：vectorize_file 先写后删时不得把刚写入的新分片删掉。

    历史 bug：adelete_by_source(filename) 与旧分片 source 相同，
    会连同新写入的分片一起删除 —— 接口报 chunk_count=3 但库内 0。
    必须真实走 vectorize_file（含 add -> delete 写序）验证。
    """

    @pytest.fixture(autouse=True)
    def _env(self, tmp_path, monkeypatch):
        from app.services.embeddings import (
            reset_embeddings_for_test,
            set_embeddings_for_test,
        )
        from tests.helpers import make_hash_embeddings

        monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "upload"))
        monkeypatch.setattr(settings, "MILVUS_URL", str(tmp_path / "milvus" / "test.db"))
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        # 注入确定性 Embedding（离线可跑）
        set_embeddings_for_test(make_hash_embeddings(size=128))
        vector_store.reset_vector_store_for_test()
        yield
        vector_store.reset_vector_store_for_test()
        reset_embeddings_for_test()

    def _write_doc(self, name: str, content: str):
        with open(os.path.join(settings.UPLOAD_DIR, name), "w", encoding="utf-8") as f:
            f.write(content)

    async def _count(self) -> int:
        import re

        store = vector_store.get_vector_store()
        res = await store.aclient.query(
            collection_name=settings.MILVUS_COLLECTION,
            filter="",
            output_fields=["count(*)"],
        )
        m = re.search(r"count\(\*\)': (\d+)", str(res))
        return int(m.group(1)) if m else -1

    def test_vectorize_persists_chunks(self):
        """回归：vectorize 后库内分片数 == 接口报告的 chunk_count（不得被误删）。"""
        self._write_doc("rag.txt", "第一段。第二段。第三段。" * 20)

        async def body():
            result = await rag_service.vectorize_file("rag.txt")
            return result, await self._count()

        result, count = asyncio.run(body())
        assert result["chunk_count"] >= 1
        assert count == result["chunk_count"], (
            f"库内 {count} 条 != 接口 {result['chunk_count']} 条 —— 新分片被误删"
        )

    def test_revectorize_idempotent_count(self):
        """回归：重复 vectorize 同一文件，库内分片数不增长（先写后删不累积）。"""
        self._write_doc("rag.txt", "重复内容。" * 50)

        async def body():
            r1 = await rag_service.vectorize_file("rag.txt")
            c1 = await self._count()
            r2 = await rag_service.vectorize_file("rag.txt")
            c2 = await self._count()
            return r1, c1, r2, c2

        r1, c1, r2, c2 = asyncio.run(body())
        assert c1 == r1["chunk_count"]
        assert c2 == r2["chunk_count"]
        assert c2 == c1, f"重复向量化后分片数变化：{c1} -> {c2}（应为幂等）"

class TestBackendUnavailable:
    """is_backend_unavailable 必须覆盖 Milvus 连接异常（缺失时接口裸 500）。"""

    def test_milvus_base_exception(self):
        """实证：连接失败抛 MilvusException 基类（code=2 Fail connecting to server）。"""
        from pymilvus.exceptions import MilvusException

        exc = MilvusException(2, "Fail connecting to server on 127.0.0.1:19530")
        assert rag_service.is_backend_unavailable(exc) is True
        assert rag_service.is_milvus_error(exc) is True

    def test_milvus_connection_config_exception(self):
        from pymilvus.exceptions import ConnectionConfigException

        exc = ConnectionConfigException("Illegal uri: [bad]")
        assert rag_service.is_backend_unavailable(exc) is True
        assert rag_service.is_milvus_error(exc) is True

    def test_milvus_unavailable_exception(self):
        from pymilvus.exceptions import MilvusUnavailableException

        exc = MilvusUnavailableException("milvus service unavailable")
        assert rag_service.is_backend_unavailable(exc) is True

    def test_grpc_rpc_error(self):
        # 网络层兜底：grpc 通道不可用时抛 RpcError
        class _FakeRpcError(grpc.RpcError, RuntimeError):
            pass

        assert rag_service.is_backend_unavailable(_FakeRpcError()) is True

    def test_non_backend_error_still_false(self):
        class _BizBug(RuntimeError):
            pass

        assert rag_service.is_backend_unavailable(_BizBug()) is False