# -*- coding: utf-8 -*-
"""切块策略：递归切分 + 语义切分 + 混合切分（升级需求 #3 前半部分）。

- recursive：RecursiveCharacterTextSplitter，稳定、零外部依赖（默认）；
- semantic ：把文本切成句子，用 Embedding 计算相邻句子的余弦相似度，
  在相似度最低的 ~(100-percentile)% 间隔处断句（主题跳变点），
  再按 min/max 字符约束合并，得到语义更完整的块；
- hybrid   ：先递归粗分，再对相邻块做语义相似度合并（相似度高于阈值且不超长则合并），
  兼顾「长度可控」与「语义完整」。
"""
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

# 中英文句末标点 + 换行，作为句子切分点（保留分隔符在句尾）
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;\n])")
# 递归切分的分隔符优先级（中文优先，兼顾英文）
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]

VALID_STRATEGIES = ("recursive", "semantic", "hybrid")


class ChunkingError(ValueError):
    """切块失败（如语义切分缺少 Embedding 配置）。"""


def split_text(
    text: str, strategy: str | None = None, embeddings=None
) -> list[str]:
    """按策略切分文本，返回非空块列表。

    Args:
        text: 清洗后的文档纯文本。
        strategy: recursive / semantic / hybrid，None 时取 settings.CHUNKING_STRATEGY。
        embeddings: semantic / hybrid 策略需要（Embeddings 实例）；缺省时尝试默认实例。
    """
    strategy = (strategy or settings.CHUNKING_STRATEGY).strip().lower()
    if strategy not in VALID_STRATEGIES:
        raise ChunkingError(
            f"未知切块策略：{strategy!r}（可选 {'/'.join(VALID_STRATEGIES)}）"
        )
    if not text or not text.strip():
        return []
    if strategy == "semantic":
        if embeddings is None:
            from app.services.embeddings import get_embeddings

            embeddings = get_embeddings()
        return semantic_chunk(text, embeddings)
    if strategy == "hybrid":
        if embeddings is None:
            from app.services.embeddings import get_embeddings

            embeddings = get_embeddings()
        return hybrid_chunk(text, embeddings)
    return recursive_chunk(text)


# ---------- 递归切分 ----------
def recursive_chunk(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=_SEPARATORS,
        length_function=len,
    )
    return [c.strip() for c in splitter.split_text(text) if c and c.strip()]


# ---------- 语义切分 ----------
def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text)]
    return [s for s in sentences if s]


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度（向量可能来自不同批，逐项计算）。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _percentile(values: list[float], pct: float) -> float:
    """线性插值分位数，用于确定语义断点阈值。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _hard_split_oversized(text: str) -> list[str]:
    """超长片段（如无标点长文本）按字符上限硬切。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.SEMANTIC_MAX_CHUNK_CHARS,
        chunk_overlap=min(settings.CHUNK_OVERLAP, settings.SEMANTIC_MAX_CHUNK_CHARS // 10),
        separators=_SEPARATORS,
    )
    return [c.strip() for c in splitter.split_text(text) if c and c.strip()]


def semantic_chunk(text: str, embeddings) -> list[str]:
    """语义切分：句子 -> 句向量 -> 相似度断点 -> 合并为块。

    规则：
    - 在相似度低于分位阈值的间隔处断句（语义跳变点）；
    - 断点处若当前块还不足 min 字符，则跨过断点继续累积，避免语义碎片；
    - 达到 max 字符强制切块；单句超长直接硬切。
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return _hard_split_oversized(sentences[0])

    vectors = embeddings.embed_documents(sentences)
    sims = [
        _cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)
    ]
    threshold = _percentile(sims, settings.SEMANTIC_BREAKPOINT_PERCENTILE)
    # 断点集合：相似度低于阈值（语义跳变）的间隔
    breaks = {i + 1 for i, s in enumerate(sims) if s < threshold}

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    min_chars = settings.SEMANTIC_MIN_CHUNK_CHARS
    max_chars = settings.SEMANTIC_MAX_CHUNK_CHARS

    def flush():
        nonlocal current, current_len
        text = "".join(current).strip()
        if text:
            chunks.append(text)
        current, current_len = [], 0

    for idx, sentence in enumerate(sentences):
        if len(sentence) > max_chars:
            # 单句超长：先冲刷当前累积，再把超长句硬切为多块
            flush()
            chunks.extend(_hard_split_oversized(sentence))
            continue
        current.append(sentence)
        current_len += len(sentence)
        if current_len >= max_chars:
            flush()
        elif idx + 1 in breaks and current_len >= min_chars:
            flush()
    flush()

    # 尾部过小块：并入前一块，避免语义碎片
    if len(chunks) >= 2 and len(chunks[-1]) < min_chars:
        chunks[-2] = chunks[-2] + chunks[-1]
        chunks.pop()

    return chunks


# ---------- 混合切分 ----------
def hybrid_chunk(text: str, embeddings) -> list[str]:
    """混合切分：递归粗分 -> 语义相似度合并相邻块。"""
    blocks = recursive_chunk(text)
    if len(blocks) <= 1:
        return blocks

    vectors = embeddings.embed_documents(blocks)
    merged: list[str] = []
    buffer = blocks[0]
    buffer_vec = vectors[0]
    threshold = settings.SEMANTIC_MERGE_THRESHOLD
    max_chars = settings.SEMANTIC_MAX_CHUNK_CHARS

    for i in range(1, len(blocks)):
        sim = _cosine(buffer_vec, vectors[i])
        if sim >= threshold and len(buffer) + len(blocks[i]) <= max_chars:
            buffer = buffer + blocks[i]
            buffer_vec = vectors[i]  # 以最新块为锚点继续合并
        else:
            merged.append(buffer)
            buffer = blocks[i]
            buffer_vec = vectors[i]
    merged.append(buffer)
    return [c.strip() for c in merged if c and c.strip()]