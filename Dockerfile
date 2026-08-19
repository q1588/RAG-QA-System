# 基于大模型的 RAG 知识库问答系统 —— 后端镜像
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖（利用构建缓存，代码改动不会重装依赖）
COPY requirements.txt requirements-mysql.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-mysql.txt

# 再拷贝代码与静态资源
COPY app ./app
COPY static ./static

# 数据目录（上传文档 / 向量库 / SQLite）
RUN mkdir -p upload_files chroma_store

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
