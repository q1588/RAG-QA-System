# -*- coding: utf-8 -*-
"""pytest 会话级环境隔离。

app.core.config 在导入时会一次性读取环境变量生成 settings 单例，
所以必须在任何 app 模块被导入【之前】设置好临时目录，
避免测试污染项目根目录的 app.db / milvus_store / upload_files。
conftest.py 由 pytest 最先加载，满足这个时序要求。
"""
import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="rag_test_"))

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///" + (_TMP / "test.db").as_posix()
os.environ["MILVUS_URL"] = str(_TMP / "milvus" / "test.db")  # Milvus Lite 本地文件
os.environ["UPLOAD_DIR"] = str(_TMP / "upload")
os.environ["JWT_SECRET"] = "test-secret-for-automated-tests-0123456789abcdef"  # >=32 字节，避免 InsecureKeyLengthWarning
os.environ["DB_ECHO"] = "false"
# 单测不依赖外部 API：embedding / LLM 由测试注入确定性实现（见 tests/helpers.py）
os.environ["EMBEDDING_PROVIDER"] = "openai"
os.environ["EMBEDDING_API_KEY"] = ""
os.environ["LLM_PROVIDER"] = "openai"
os.environ["LLM_API_KEY"] = ""