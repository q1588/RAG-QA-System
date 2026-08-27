# -*- coding: utf-8 -*-
"""RAG 检索评估指标（纯函数，供 scripts/eval_rag.py 与单测使用）。

评估对象：检索模块「召回」效果 —— 给定问题与 ground-truth 文本片段，
衡量检索到的 Top-K 分片是否命中相关知识。

指标定义（K 为检索返回数量）：
- Hit@K    ：问题级别命中率（Top-K 内至少召回 1 个相关分片则计 1）；
- Recall@K ：相关分片召回率 = |Top-K 中相关分片| / |语料中相关分片总数|；
- Precision@K：|Top-K 中相关分片| / K；
- MRR@K    ：首个相关分片排名的倒数均值（Mean Reciprocal Rank）。

匹配规则：分片文本（归一化后）包含任一 ground-truth 片段（归一化后）即视为相关，
因此 ground truth 不用绑定具体 chunk id，切块策略变化后依然可复评。
"""
from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """匹配前归一化：去首尾空白、压缩连续空白（含换行），便于片段匹配。"""
    return _WS_RE.sub("", (text or "").strip())


def chunk_is_relevant(chunk_text: str, snippets: list[str]) -> bool:
    """分片是否相关：包含任一 ground-truth 片段（归一化后子串匹配）。"""
    normalized = normalize_text(chunk_text)
    return any(normalize_text(s) and normalize_text(s) in normalized for s in snippets)


def evaluate_question(
    retrieved_texts: list[str],
    snippets: list[str],
    k: int,
    corpus_relevant_count: int | None = None,
) -> dict:
    """评估单个问题的检索结果。

    Args:
        retrieved_texts: 按相似度降序的检索分片文本（长度 >= k）。
        snippets: ground-truth 片段列表。
        k: 评估的截断长度。
        corpus_relevant_count: 语料中相关分片总数（用于 Recall@K）；缺省时按
            该问题命中的相关分片数估算，仅作近似。

    Returns:
        {"hit", "recall", "precision", "mrr"} 各为 0~1 的分数。
    """
    top = retrieved_texts[:k]
    rel_flags = [chunk_is_relevant(t, snippets) for t in top]
    relevant_hits = sum(rel_flags)

    hit = 1.0 if relevant_hits > 0 else 0.0
    denominator = corpus_relevant_count if corpus_relevant_count is not None else max(relevant_hits, 1)
    recall = relevant_hits / denominator if denominator else 0.0
    precision = relevant_hits / k if k else 0.0
    mrr = 0.0
    for rank, flag in enumerate(rel_flags, start=1):
        if flag:
            mrr = 1.0 / rank
            break
    return {"hit": hit, "recall": recall, "precision": precision, "mrr": mrr}


def aggregate_metrics(per_question: list[dict], k: int) -> dict:
    """对多个问题取平均，输出一组可对比的指标。"""
    n = len(per_question)
    if n == 0:
        return {"k": k, "n_questions": 0, "hit@k": 0.0, "recall@k": 0.0, "precision@k": 0.0, "mrr@k": 0.0}
    return {
        "k": k,
        "n_questions": n,
        "hit@k": round(sum(q["hit"] for q in per_question) / n, 4),
        "recall@k": round(sum(q["recall"] for q in per_question) / n, 4),
        "precision@k": round(sum(q["precision"] for q in per_question) / n, 4),
        "mrr@k": round(sum(q["mrr"] for q in per_question) / n, 4),
    }
