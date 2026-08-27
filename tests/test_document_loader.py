# -*- coding: utf-8 -*-
"""文档加载容错单测：编码回退、控制字符清洗、损坏文件友好报错。"""
import os

import pytest

from app.services.document_loader import (
    DocumentParseError,
    SUPPORTED_EXTS,
    _clean_text,
    load_document,
)


def _write(tmp_path, name, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


class TestTextEncoding:
    def test_utf8(self, tmp_path):
        p = _write(tmp_path, "a.txt", "中文内容 RAG".encode("utf-8"))
        assert load_document(p) == "中文内容 RAG"

    def test_gbk_fallback(self, tmp_path):
        p = _write(tmp_path, "a.txt", "中文内容".encode("gbk"))
        assert load_document(p) == "中文内容"

    def test_latin1_fallback_no_crash(self, tmp_path):
        p = _write(tmp_path, "a.bin.txt", bytes(range(256)))
        text = load_document(p)
        assert isinstance(text, str) and text

    def test_control_chars_cleaned(self, tmp_path):
        p = _write(tmp_path, "a.txt", "hello\x00world\x1fend".encode("latin-1"))
        text = load_document(p)
        assert "\x00" not in text and "\x1f" not in text
        assert "hello" in text and "end" in text


class TestCleanText:
    def test_collapse_blank_lines(self):
        assert _clean_text("a\n\n\n\n\nb") == "a\n\nb"
        assert _clean_text("  ") == ""


class TestUnsupported:
    def test_unsupported_extension(self, tmp_path):
        p = _write(tmp_path, "evil.exe", b"MZ")
        with pytest.raises(ValueError):
            load_document(p)


class TestCorruptFiles:
    def test_corrupt_pdf_friendly_error(self, tmp_path):
        p = _write(tmp_path, "broken.pdf", b"%PDF-1.4 not really a pdf")
        with pytest.raises(DocumentParseError) as ei:
            load_document(p)
        assert "broken.pdf" in str(ei.value)  # 错误信息包含文件名

    def test_corrupt_docx_friendly_error(self, tmp_path):
        p = _write(tmp_path, "broken.docx", b"PK\x03\x04 broken archive")
        with pytest.raises(DocumentParseError) as ei:
            load_document(p)
        assert "broken.docx" in str(ei.value)

    def test_empty_pdf_text_error(self, tmp_path):
        # 构造一个没有任何文本的极简 PDF（仅含 xref 表，无内容流）
        minimal = (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
            b"trailer<</Root 1 0 R/Size 4>>\nstartxref\n0\n%%EOF"
        )
        p = _write(tmp_path, "blank.pdf", minimal)
        with pytest.raises(DocumentParseError):
            load_document(p)


class TestSupportedExts:
    def test_exts_constant(self):
        assert SUPPORTED_EXTS == {".txt", ".md", ".pdf", ".docx"}
