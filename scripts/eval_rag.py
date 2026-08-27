# -*- coding: utf-8 -*-
"""RAG 检索召回评估脚本（升级需求 #4：引入评估数据集，量化召回效果）。

对每个问题：向量检索 Top-K -> 用 ground-truth 片段计算 Hit@K / Recall@K /
Precision@K / MRR@K；支持对比多种切块策略，输出 Markdown 表格 + JSON 报告，
从「肉眼看效果」升级为「可量化、可回归」的指标。

用法（在 RAG-QA-System 目录下）：
    # 离线自检（确定性假 Embedding，无需任何 API Key，结果可复现）
    ../.venv/Scripts/python.exe scripts/eval_rag.py --embeddings fake

    # 使用 .env 中配置的真实 Embedding API（BAAI/bge-m3 等）
    ../.venv/Scripts/python.exe scripts/eval_rag.py --embeddings real

    # 对比三种切块策略 + 多个 Top-K
    ../.venv/Scripts/python.exe scripts/eval_rag.py --embeddings real --strategies recursive,semantic,hybrid --top-k 3,5

说明：
- 评估语料默认 data/eval_corpus（与线上 upload_files 解耦，保证评估可复现）；
- 每个策略使用独立 Milvus collection（rag_eval_<strategy>），每次运行重建；
- ground truth 为文本片段（不绑定 chunk id），换切块策略后仍可复评。
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

# 确保可导入 app 包（脚本在 scripts/ 下运行）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Windows 控制台默认 cp1252，直接 print 中文会 UnicodeEncodeError（与 smoke_test.py 一致的处理）
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


from langchain_milvus import Milvus

from app.core.config import settings
from app.services import chunking
from app.services.document_loader import load_document
from app.services.eval_metrics import aggregate_metrics, chunk_is_relevant, evaluate_question

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+")


class HashEmbeddings:
    """确定性、有语义信号的假 Embedding（离线评估用，无需 API Key）。

    中文按单字、英文按单词做 token 哈希叠加，共享 token 越多向量越相似，
    因此能真实驱动检索逻辑；同一文本永远得到同一向量，结果可复现。
    """

    def __init__(self, size: int = 1024, seed: int = 42):
        self.size = size
        self._seed = seed

    def _token_vec(self, token: str) -> list[float]:
        # 每 64 维用独立哈希派生位，避免 h >> (i*11) 在高维时恒 0 导致常数尾部
        vec: list[float] = []
        for chunk in range(0, self.size, 64):
            h = int(
                hashlib.sha256(f"{self._seed}:{token}:{chunk // 64}".encode("utf-8")).hexdigest(),
                16,
            )
            for j in range(min(64, self.size - chunk)):
                vec.append(1.0 if (h >> j) & 1 else -1.0)
        return vec

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.size
        for t in _TOKEN_RE.findall(text.lower()):
            tv = self._token_vec(t)
            vec = [v + x for v, x in zip(vec, tv)]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class NormalizedEmbeddings:
    """L2 归一化包装：Milvus COSINE 返回的距离即余弦相似度（越大越相关）。"""

    def __init__(self, inner):
        self._inner = inner

    @staticmethod
    def _norm(v: list[float]) -> list[float]:
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._norm(v) for v in self._inner.embed_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self._norm(self._inner.embed_query(text))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG 检索召回评估")
    p.add_argument("--dataset", default=settings.EVAL_DATASET, help="评估数据集 jsonl")
    p.add_argument("--corpus", default=settings.EVAL_CORPUS_DIR, help="评估语料目录")
    p.add_argument("--strategies", default="recursive,semantic,hybrid", help="逗号分隔的切块策略")
    p.add_argument("--top-k", default="3,5", help="逗号分隔的评估截断长度")
    p.add_argument("--embeddings", choices=["auto", "fake", "real"], default="auto",
                   help="auto=配了 Key 用真实 API，否则用离线假 Embedding；fake/real 强制指定")
    p.add_argument("--collection-prefix", default="rag_eval", help="评估用 Milvus collection 前缀")
    p.add_argument("--milvus-uri", default=None, help="评估用独立 Milvus 地址（默认取 MILVUS_URL；建议评估与线上库分离）")
    p.add_argument("--output", default="eval_report.json", help="JSON 报告输出路径")
    return p.parse_args()


def load_dataset(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["relevant_snippets"] = [s for s in row.get("relevant_snippets", []) if s]
            rows.append(row)
    return rows


_MILVUS_URI_OVERRIDE: str | None = None


def build_store(collection_name: str, embeddings_norm, drop_old: bool) -> Milvus:
    uri = _MILVUS_URI_OVERRIDE or settings.MILVUS_URL
    if not uri.startswith(("http://", "https://")):
        parent = os.path.dirname(uri)
        if parent:
            os.makedirs(parent, exist_ok=True)
    connection_args: dict = {"uri": uri}
    if settings.MILVUS_TOKEN:
        connection_args["token"] = settings.MILVUS_TOKEN  # 生产带鉴权的 Milvus/Zilliz
    return Milvus(
        embedding_function=embeddings_norm,
        connection_args=connection_args,
        collection_name=collection_name,
        auto_id=True,
        drop_old=drop_old,
        enable_dynamic_field=True,
        index_params={"index_type": "AUTOINDEX", "metric_type": settings.MILVUS_METRIC_TYPE, "params": {}},
    )


async def run_strategy(
    rows: list[dict],
    corpus_dir: str,
    strategy: str,
    embeddings_norm,
    collection_prefix: str,
    top_ks: list[int],
) -> dict:
    """向量化语料 -> 逐问题检索 -> 计算指标。"""
    collection_name = f"{collection_prefix}_{strategy}"
    # 重建评估集合（drop_old=True 清空旧数据）
    build_store(collection_name, embeddings_norm, drop_old=True)

    all_chunks: list[str] = []
    store = build_store(collection_name, embeddings_norm, drop_old=False)
    for name in sorted(os.listdir(corpus_dir)):
        path = os.path.join(corpus_dir, name)
        if not os.path.isfile(path):
            continue
        text = await asyncio.to_thread(load_document, path)
        chunks = chunking.split_text(text, strategy=strategy, embeddings=embeddings_norm)
        if chunks:
            await store.aadd_texts(
                texts=chunks,
                metadatas=[{"source": name, "strategy": strategy, "chunk_index": i} for i in range(len(chunks))],
            )
        all_chunks.extend(chunks)

    # 每个问题的语料相关分片总数（Recall@K 的分母）
    corpus_counts = [
        sum(1 for c in all_chunks if chunk_is_relevant(c, row["relevant_snippets"]))
        for row in rows
    ]

    max_k = max(top_ks)
    per_question: list[dict] = []
    for row, relevant_total in zip(rows, corpus_counts):
        q_vec = await embeddings_norm.aembed_query(row["question"])
        hits = await store.asimilarity_search_with_score_by_vector(q_vec, k=max_k)
        retrieved = [d.page_content for d, _ in hits]
        per_question.append({
            "question": row["question"],
            "source": row.get("source", ""),
            "notes": row.get("notes", ""),
            "retrieved": [
                {"source": d.metadata.get("source", ""), "text": d.page_content[:120]}
                for d, _ in hits
            ],
            "corpus_relevant_count": relevant_total,
            "per_k": {
                str(k): evaluate_question(retrieved, row["relevant_snippets"], k, relevant_total)
                for k in top_ks
            },
        })

    return {
        "strategy": strategy,
        "collection": collection_name,
        "chunk_total": len(all_chunks),
        "per_question": per_question,
        "summary": {
            str(k): aggregate_metrics([q["per_k"][str(k)] for q in per_question], k)
            for k in top_ks
        },
    }


def print_report(results: list[dict], top_ks: list[int]) -> None:
    n = results[0]["summary"][str(top_ks[0])]["n_questions"]
    print()
    print("=" * 80)
    print(f"RAG 检索召回评估报告（{n} 个评估问题，语料分片总数见 eval_report.json）")
    print("=" * 80)
    for k in top_ks:
        print(f"\n--- Top-K = {k} ---")
        print(f"{'切块策略':<12}{'Hit@K':>14}{'Recall@K':>14}{'Precision@K':>16}{'MRR@K':>14}")
        print("-" * 70)
        for r in results:
            s = r["summary"][str(k)]
            print(
                f"{r['strategy']:<12}"
                f"{s['hit@k']:>14.4f}{s['recall@k']:>14.4f}{s['precision@k']:>16.4f}{s['mrr@k']:>14.4f}"
            )
    print("\n指标说明：Hit@K 问题级命中率；Recall@K 相关分片召回率；MRR@K 首个相关分片排名倒数。")
    print("多策略对比可指导 CHUNK_SIZE / SEMANTIC_* 等切块参数调优；配合 .env 中真实 Embedding 复跑即为线上效果基线。")


async def run_all(rows, corpus_dir, strategies, top_ks, embeddings_norm, collection_prefix) -> list[dict]:
    return [
        await run_strategy(rows, corpus_dir, strategy, embeddings_norm, collection_prefix, top_ks)
        for strategy in strategies
    ]


def main() -> int:
    args = parse_args()
    if not os.path.isfile(args.dataset):
        print(f"❌ 评估数据集不存在：{args.dataset}")
        return 1
    if not os.path.isdir(args.corpus):
        print(f"❌ 评估语料目录不存在：{args.corpus}")
        return 1

    rows = load_dataset(args.dataset)
    print(f"✅ 加载评估数据集：{len(rows)} 个问题；语料目录：{args.corpus}")

    mode = args.embeddings
    if mode == "auto":
        mode = "real" if (settings.EMBEDDING_API_KEY or settings.EMBEDDING_PROVIDER == "ollama") else "fake"
    if mode == "fake":
        print("ℹ️  使用离线假 Embedding（确定性、可复现，用于流程验证；真实效果请用 --embeddings real）")
        embeddings_norm = NormalizedEmbeddings(HashEmbeddings(size=1024))
    else:
        from app.services.embeddings import ProviderConfigError, get_embeddings

        try:
            embeddings_norm = NormalizedEmbeddings(get_embeddings())
        except ProviderConfigError as e:
            print(f"❌ 无法使用真实 Embedding：{e}")
            print("   提示：在 .env 配置 EMBEDDING_API_KEY，或改用 --embeddings fake 做离线自检。")
            return 1

    global _MILVUS_URI_OVERRIDE
    if args.milvus_uri:
        _MILVUS_URI_OVERRIDE = args.milvus_uri
    uri_display = _MILVUS_URI_OVERRIDE or settings.MILVUS_URL
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    top_ks = [int(x) for x in args.top_k.split(",") if x.strip()]
    print(f"ℹ️  待评估策略：{strategies}；Top-K：{top_ks}；Milvus URL：{uri_display}")

    results = asyncio.run(
        run_all(rows, args.corpus, strategies, top_ks, embeddings_norm, args.collection_prefix)
    )

    report = {
        "dataset": args.dataset,
        "corpus": args.corpus,
        "embeddings_mode": mode,
        "top_ks": top_ks,
        "strategies": strategies,
        "milvus_url": _MILVUS_URI_OVERRIDE or settings.MILVUS_URL,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n📄 完整报告已保存：{args.output}")

    print_report(results, top_ks)
    return 0


if __name__ == "__main__":
    sys.exit(main())