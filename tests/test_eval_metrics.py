# -*- coding: utf-8 -*-
"""RAG 检索评估指标单测。"""
import pytest
from app.services.eval_metrics import (
    aggregate_metrics,
    chunk_is_relevant,
    evaluate_question,
    normalize_text,
)


class TestNormalize:
    def test_compresses_whitespace(self):
        assert normalize_text("  RAG 是 检索 \n 增强  ") == "RAG是检索增强"


class TestRelevance:
    def test_snippet_match(self):
        assert chunk_is_relevant("RAG是检索增强生成的技术", ["检索增强生成"])
        assert not chunk_is_relevant("汽车引擎", ["水果"])

    def test_multiline_match(self):
        assert chunk_is_relevant("第一行\n第二行包含答案", ["第二行包含答案"])


class TestEvaluateQuestion:
    RETRIEVED = ["苹果是水果", "汽车引擎", "香蕉也是水果", "火箭燃料"]

    def test_hit_and_mrr(self):
        r = evaluate_question(self.RETRIEVED, ["水果"], k=3, corpus_relevant_count=3)
        assert r["hit"] == 1.0
        assert r["mrr"] == 1.0  # 第一个就是相关

    def test_mrr_second_rank(self):
        r = evaluate_question(["汽车引擎", "苹果是水果", "香蕉"], ["水果"], k=3, corpus_relevant_count=2)
        assert r["hit"] == 1.0
        assert r["mrr"] == pytest.approx(0.5)

    def test_recall_denominator(self):
        r = evaluate_question(self.RETRIEVED, ["水果"], k=3, corpus_relevant_count=4)
        assert r["recall"] == pytest.approx(0.5)  # top3 中 2 条相关 / 语料 4 条
        assert r["precision"] == pytest.approx(2 / 3)

    def test_miss(self):
        r = evaluate_question(["汽车引擎", "火箭燃料"], ["水果"], k=3, corpus_relevant_count=3)
        assert r["hit"] == 0.0
        assert r["recall"] == 0.0
        assert r["mrr"] == 0.0


class TestAggregate:
    def test_mean(self):
        results = [
            evaluate_question(["x", "a"], ["x"], k=2, corpus_relevant_count=2),  # 首个命中
            evaluate_question(["b", "x"], ["x"], k=2, corpus_relevant_count=2),  # 次个命中
        ]
        agg = aggregate_metrics(results, k=2)
        assert agg["n_questions"] == 2
        assert agg["hit@k"] == pytest.approx(1.0)
        assert agg["mrr@k"] == pytest.approx(0.75)  # (1 + 0.5) / 2

    def test_empty(self):
        agg = aggregate_metrics([], k=3)
        assert agg["n_questions"] == 0