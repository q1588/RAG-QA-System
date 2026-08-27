# -*- coding: utf-8 -*-
# ⚠️ 历史存档（v1 单文件版本，仅供代码参考，不参与运行）。
#    依赖已从 requirements 移除：langchain_chroma / chromadb / langchain_ollama 需自行安装才能运行；
#    当前版本（v2）的向量库为 Milvus（app/services/vector_store.py），与本文档无关联。

import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File