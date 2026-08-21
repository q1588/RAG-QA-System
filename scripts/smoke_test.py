# -*- coding: utf-8 -*-
"""端到端冒烟测试：注册 -> 登录 -> 我的信息 -> 建会话 -> RAG 问答 -> 查历史。

用法：先启动服务（python -m uvicorn app.main:app --port 8000），再运行：
    ../.venv/Scripts/python.exe scripts/smoke_test.py
"""
import asyncio
import os
import sys

import httpx

# Windows 下控制台默认 cp1252，直接 print emoji/中文会 UnicodeEncodeError；
# 统一切到 UTF-8，保证冒烟日志能正常输出。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# 可用环境变量 SMOKE_BASE_URL 覆盖，默认 8000
BASE_URL = os.getenv("SMOKE_BASE_URL", "http://127.0.0.1:8000")


def log(title: str, ok: bool, detail: str = ""):
    mark = "✅" if ok else "❌"
    print(f"{mark} {title}" + (f" —— {detail}" if detail else ""))


async def main() -> int:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120) as c:
        # 1. 注册（重复注册容忍 409）
        username = "smoke_user"
        r = await c.post(
            "/api/auth/register", json={"username": username, "password": "test123456"}
        )
        if r.status_code == 201:
            log("注册新用户", True)
        elif r.status_code == 409:
            log("注册（已存在，复用旧账号）", True, "409 用户名已存在")
        else:
            log("注册", False, f"HTTP {r.status_code}: {r.text}")
            return 1

        # 2. 登录拿 token
        r = await c.post(
            "/api/auth/login",
            data={"username": username, "password": "test123456"},
        )
        if r.status_code != 200:
            log("登录", False, f"HTTP {r.status_code}: {r.text}")
            return 1
        token = r.json()["access_token"]
        log("登录并获取 JWT", True)
        headers = {"Authorization": f"Bearer {token}"}

        # 3. 我的信息
        r = await c.get("/api/auth/me", headers=headers)
        if r.status_code != 200 or r.json()["username"] != username:
            log("获取当前用户", False, r.text)
            return 1
        log("获取当前用户", True, f"id={r.json()['id']}")

        # 4. 建会话
        r = await c.post(
            "/api/chat/conversations",
            headers=headers,
            json={"title": "冒烟测试"},
        )
        if r.status_code != 201:
            log("创建会话", False, r.text)
            return 1
        conv_id = r.json()["id"]
        log("创建会话", True, f"conversation_id={conv_id}")

        # 5. RAG 问答（需 Ollama 运行 + 知识库已有文档）
        r = await c.post(
            "/api/rag/query",
            headers=headers,
            json={"conversation_id": conv_id, "question": "RAG 是什么？"},
        )
        if r.status_code != 200:
            log("RAG 问答", False, f"HTTP {r.status_code}: {r.text}")
            return 1
        data = r.json()
        log(
            "RAG 问答",
            bool(data["answer"]),
            f"答案前 30 字：{data['answer'][:30]!r}  来源 {len(data['source_docs'])} 段",
        )

        # 6. 消息持久化校验：应恰好写入 user + assistant 两条
        r = await c.get(f"/api/chat/conversations/{conv_id}/messages", headers=headers)
        msgs = r.json()
        roles = [m["role"] for m in msgs]
        ok = roles == ["user", "assistant"]
        log("消息持久化", ok, f"共 {len(msgs)} 条，roles={roles}")

        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
