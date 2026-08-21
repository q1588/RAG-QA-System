# -*- coding: utf-8 -*-
"""pytest 会话级环境隔离。

app.core.config 在导入时会一次性读取环境变量生成 settings 单例，
所以必须在任何 app 模块被导入【之前】设置好临时目录，
避免测试污染项目根目录的 app.db / chroma_store / upload_files。
conftest.py 由 pytest 最先加载，满足这个时序要求。
"""
import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="rag_test_"))

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///" + (_TMP / "test.db").as_posix()
os.environ["CHROMA_DIR"] = str(_TMP / "chroma")
os.environ["UPLOAD_DIR"] = str(_TMP / "upload")
os.environ["JWT_SECRET"] = "test-secret-for-automated-tests-0123456789abcdef"  # >=32 字节，避免 InsecureKeyLengthWarning
os.environ["DB_ECHO"] = "false"
