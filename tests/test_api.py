# -*- coding: utf-8 -*-
"""接口集成测试：认证 / 会话 / 消息 / 文档上传删除 / 健康检查。

依赖外部模型服务（向量化、RAG 问答）的接口通过 monkeypatch 打桩，
保证单测环境不依赖外部 API 即可覆盖接口层逻辑；
真实模型链路由 scripts/smoke_test.py / scripts/eval_rag.py 做端到端验证。
"""
import pytest
from fastapi.testclient import TestClient

# conftest.py 已在本模块导入前设置好隔离的 DATABASE_URL/MILVUS_URL/UPLOAD_DIR
from app.core.config import settings
from app.api import rag as rag_api
from app.main import app


@pytest.fixture(scope="module")
def client():
    """TestClient 上下文管理器会触发 lifespan（建表 + 创建数据目录 + Milvus 初始化）。"""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def auth_headers(client):
    """注册并登录一个测试用户，返回 Bearer 头。"""
    username = "api_user"
    client.post(
        "/api/auth/register",
        json={"username": username, "password": "test123456"},
    )
    r = client.post(
        "/api/auth/login",
        data={"username": username, "password": "test123456"},
    )
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


class TestHealth:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["database"] == "ok"


class TestAuth:
    def test_register_duplicate_409(self, client):
        client.post(
            "/api/auth/register",
            json={"username": "dup_user", "password": "test123456"},
        )
        r = client.post(
            "/api/auth/register",
            json={"username": "dup_user", "password": "test123456"},
        )
        assert r.status_code == 409  # 用户名唯一约束

    def test_register_short_password_422(self, client):
        r = client.post(
            "/api/auth/register",
            json={"username": "short_pw", "password": "123"},
        )
        assert r.status_code == 422  # 密码少于 6 位

    def test_login_wrong_password_401(self, client, auth_headers):
        r = client.post(
            "/api/auth/login",
            data={"username": "api_user", "password": "wrong-password"},
        )
        assert r.status_code == 401

    def test_me_without_token_401(self, client):
        assert client.get("/api/auth/me").status_code == 401

    def test_me(self, client, auth_headers):
        r = client.get("/api/auth/me", headers=auth_headers)
        assert r.status_code == 200
        assert r.json()["username"] == "api_user"


class TestConversation:
    def test_create_and_list(self, client, auth_headers):
        r = client.post(
            "/api/chat/conversations",
            headers=auth_headers,
            json={"title": "集成测试会话"},
        )
        assert r.status_code == 201
        conv_id = r.json()["id"]

        r = client.get("/api/chat/conversations", headers=auth_headers)
        assert r.status_code == 200
        ids = [c["id"] for c in r.json()]
        assert conv_id in ids

    def test_foreign_conversation_404(self, client, auth_headers):
        # 不存在的会话 -> 404
        r = client.get("/api/chat/conversations/999999/messages", headers=auth_headers)
        assert r.status_code == 404


async def _fake_rag_query(question: str, history: list, top_k: int = 3):
    """测试用 RAG 打桩：返回固定答案，不依赖外部 API。"""
    return {"answer": f"针对「{question}」的模拟回答", "source_docs": []}


class TestMessage:
    def test_update_message(self, client, auth_headers, monkeypatch):
        monkeypatch.setattr(rag_api.rag_service, "rag_query", _fake_rag_query)
        conv_id = client.post(
            "/api/chat/conversations", headers=auth_headers, json={}
        ).json()["id"]
        # 先通过一次（打桩的）问答插入 user + assistant 两条消息，再更新其中一条
        client.post(
            "/api/rag/query",
            headers=auth_headers,
            json={"conversation_id": conv_id, "question": "占位"},
        )
        msgs = client.get(
            f"/api/chat/conversations/{conv_id}/messages", headers=auth_headers
        ).json()
        user_msg = next(m for m in msgs if m["role"] == "user")

        r = client.patch(
            f"/api/chat/messages/{user_msg['id']}",
            headers=auth_headers,
            json={"content": "修改后的消息"},
        )
        assert r.status_code == 200
        assert r.json()["content"] == "修改后的消息"


class TestFiles:
    def test_upload_list_delete(self, client, auth_headers):
        # 上传
        r = client.post(
            "/api/rag/upload",
            headers=auth_headers,
            files={"file": ("doc1.txt", b"hello rag content", "text/plain")},
        )
        assert r.status_code == 200
        assert r.json()["filename"] == "doc1.txt"

        # 列表应包含该文件
        r = client.get("/api/rag/files", headers=auth_headers)
        assert r.status_code == 200
        names = [f["filename"] for f in r.json()]
        assert "doc1.txt" in names

        # 删除后列表为空
        r = client.delete("/api/rag/files/doc1.txt", headers=auth_headers)
        assert r.status_code == 200
        assert r.json()["deleted"] == "doc1.txt"

        r = client.get("/api/rag/files", headers=auth_headers)
        assert "doc1.txt" not in [f["filename"] for f in r.json()]

    def test_upload_too_large_413(self, client, auth_headers, monkeypatch):
        monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 0)  # 任何非空内容都超限
        r = client.post(
            "/api/rag/upload",
            headers=auth_headers,
            files={"file": ("big.txt", b"x" * 1024, "text/plain")},
        )
        assert r.status_code == 413

    def test_unsupported_extension_400(self, client, auth_headers):
        r = client.post(
            "/api/rag/upload",
            headers=auth_headers,
            files={"file": ("evil.exe", b"MZ", "application/octet-stream")},
        )
        assert r.status_code == 400


class TestRag:
    def test_vectorize(self, client, auth_headers, monkeypatch):
        async def fake_vectorize(filename: str, strategy: str | None = None, rebuild: bool = False):
            return {"filename": filename, "chunk_count": 3, "strategy": strategy or "recursive"}

        monkeypatch.setattr(rag_api.rag_service, "vectorize_file", fake_vectorize)

        r = client.post(
            "/api/rag/vectorize",
            headers=auth_headers,
            json={"filename": "doc1.txt"},
        )
        assert r.status_code == 200
        assert r.json()["chunk_count"] == 3

    def test_vectorize_with_strategy(self, client, auth_headers, monkeypatch):
        async def fake_vectorize(filename: str, strategy: str | None = None, rebuild: bool = False):
            return {"filename": filename, "chunk_count": 1, "strategy": strategy}

        monkeypatch.setattr(rag_api.rag_service, "vectorize_file", fake_vectorize)

        r = client.post(
            "/api/rag/vectorize",
            headers=auth_headers,
            json={"filename": "doc1.txt", "strategy": "semantic"},
        )
        assert r.status_code == 200
        assert r.json()["strategy"] == "semantic"

    def test_vectorize_invalid_strategy_422(self, client, auth_headers):
        # pydantic Literal 校验：非法策略 -> 422
        r = client.post(
            "/api/rag/vectorize",
            headers=auth_headers,
            json={"filename": "doc1.txt", "strategy": "unknown"},
        )
        assert r.status_code == 422

    def test_vectorize_missing_file_404(self, client, auth_headers):
        # 不打桩：真实 vectorize_file 对不存在的文件抛 FileNotFoundError -> 404
        r = client.post(
            "/api/rag/vectorize",
            headers=auth_headers,
            json={"filename": "不存在.md"},
        )
        assert r.status_code == 404

    def test_vectorize_path_traversal_400(self, client, auth_headers):
        # 路径穿越：必须在读取文件前被 400 拒绝（不依赖文件是否存在）
        r = client.post(
            "/api/rag/vectorize",
            headers=auth_headers,
            json={"filename": "../probe_read_me.md"},
        )
        assert r.status_code == 400
        assert "路径穿越" in r.json()["detail"]

    def test_delete_path_traversal_400(self, client, auth_headers):
        # %5C 解码为反斜杠的 Windows 路径穿越同样被 400 拒绝
        r = client.delete("/api/rag/files/..%5Cprobe_delete_me.txt", headers=auth_headers)
        assert r.status_code == 400
        assert "路径穿越" in r.json()["detail"]

    def test_vectorize_milvus_down_503(self, client, auth_headers, monkeypatch):
        """Milvus 不可用（ConnectionConfigException）时 vectorize 返回 503 而非裸 500。"""
        from pymilvus.exceptions import ConnectionConfigException

        async def broken(filename: str, strategy: str | None = None, rebuild: bool = False):
            raise ConnectionConfigException("milvus_store/rag.db 被占用或损坏")

        monkeypatch.setattr(rag_api.rag_service, "vectorize_file", broken)
        r = client.post(
            "/api/rag/vectorize",
            headers=auth_headers,
            json={"filename": "doc1.txt"},
        )
        assert r.status_code == 503
        assert "Milvus" in r.json()["detail"]

    def test_delete_milvus_down_503(self, client, auth_headers, monkeypatch):
        """Milvus 不可用时删除返回 503（而非裸 500），文件保留可重试。"""
        from pymilvus.exceptions import MilvusException

        async def broken(filename: str):
            raise MilvusException(2, "Fail connecting to server on 127.0.0.1:19530")

        monkeypatch.setattr(rag_api.rag_service, "delete_file", broken)
        r = client.delete("/api/rag/files/doc1.txt", headers=auth_headers)
        assert r.status_code == 503
        assert "Milvus" in r.json()["detail"]

    def test_query_milvus_down_graceful_200(self, client, auth_headers, monkeypatch):
        """Milvus 不可用时查询走降级兜底（200 + 友好回答），而非 500。"""
        from pymilvus.exceptions import ConnectionConfigException

        async def broken(question, history, top_k=None):
            raise ConnectionConfigException("milvus not reachable")

        monkeypatch.setattr(rag_api.rag_service, "rag_query", broken)
        conv_id = client.post(
            "/api/chat/conversations", headers=auth_headers, json={}
        ).json()["id"]
        r = client.post(
            "/api/rag/query",
            headers=auth_headers,
            json={"conversation_id": conv_id, "question": "RAG 是什么？"},
        )
        assert r.status_code == 200
        assert "无法连接" in r.json()["answer"] or "Milvus" in r.json()["answer"]

    def test_query_persists_messages(self, client, auth_headers, monkeypatch):
        monkeypatch.setattr(rag_api.rag_service, "rag_query", _fake_rag_query)

        conv_id = client.post(
            "/api/chat/conversations", headers=auth_headers, json={}
        ).json()["id"]

        r = client.post(
            "/api/rag/query",
            headers=auth_headers,
            json={"conversation_id": conv_id, "question": "什么是RAG？"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "针对「什么是RAG？」的模拟回答"
        assert body["source_docs"] == []

        # 问答对已持久化：user + assistant 两条
        msgs = client.get(
            f"/api/chat/conversations/{conv_id}/messages", headers=auth_headers
        ).json()
        assert [m["role"] for m in msgs] == ["user", "assistant"]