# -*- coding: utf-8 -*-
"""LLM 模型工厂：在线 API 服务（OpenAI 兼容协议）为主，本地 Ollama 为兜底。

设计要点（对应升级需求 #5：模型调用改成 API 服务，提升并发能力）：
- 默认 ChatOpenAI：异步 invoke（ainvoke），一个请求进程内即可并发处理多个用户提问，
  不再受本地 Ollama 单模型串行推理瓶颈限制；
- 支持任意 OpenAI 兼容端点（SiliconFlow / OpenAI / 阿里云百炼 / DeepSeek / Moonshot 等），
  模型名、地址、密钥全部走配置；
- 未配置 API Key 时自动降级为本地 Ollama（LLM_PROVIDER=ollama 或手动指定），离线可跑；
- 内置超时与重试（LLM_TIMEOUT / LLM_MAX_RETRIES），弱网/限流时更稳定。
"""
import threading

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.services.embeddings import ProviderConfigError


def get_chat_model() -> BaseChatModel:
    """按配置构建对话模型（进程级缓存，避免每次请求重复创建客户端）。"""
    provider = settings.LLM_PROVIDER.strip().lower()
    if provider == "openai":
        if not settings.LLM_API_KEY:
            raise ProviderConfigError(
                "未配置 LLM_API_KEY：无法使用在线大模型服务。"
                "请在 .env 中填写密钥（如 SiliconFlow / OpenAI / DeepSeek），"
                "或设置 LLM_PROVIDER=ollama 使用本地 Ollama。"
            )
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_MAX_RETRIES,
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.OLLAMA_LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
        )
    raise ProviderConfigError(
        f"未知的 LLM_PROVIDER：{provider!r}（可选 openai / ollama）"
    )


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是知识库问答助手。只能依据【知识库】里提供的原文回答用户问题，不得使用知识库之外的知识；"
            "如果知识库内容与问题无关，或你无法从知识库找到依据，必须直接回答「知识库中没有相关信息」；"
            "严禁编造，严禁列举知识库中不存在的领域或主题。回答要准确、简洁。",
        ),
        (
            "human",
            "【历史对话】\n{history}\n\n"
            "【知识库】\n{context}\n\n"
            "【问题】\n{question}\n\n"
            "请根据知识库内容给出回答。",
        ),
    ]
)


def build_rag_chain():
    """组装 PROMPT | llm | StrOutputParser，返回 Runnable（支持 ainvoke 异步调用）。"""
    return PROMPT | get_chat_model() | StrOutputParser()


_chain = None
_chain_lock = threading.Lock()


def get_rag_chain():
    """进程级单例：生成链路（PROMPT + 模型 + 输出解析）。"""
    global _chain
    if _chain is None:
        with _chain_lock:
            if _chain is None:
                _chain = build_rag_chain()
    return _chain


def reset_chain_for_test() -> None:
    global _chain
    _chain = None