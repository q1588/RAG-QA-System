# LangChain 与检索增强生成

LangChain 是构建大模型应用的开源框架，提供模型调用、提示词管理、检索、Agent 等模块化组件。

核心组件：ChatModel 封装对话模型的调用；PromptTemplate 管理提示词模板；OutputParser 解析模型输出为结构化结果；Runnable 串联各组件构成处理链。

文档加载与切分：DocumentLoader 读取 PDF、TXT 等格式；TextSplitter 将长文本切分为小块。递归字符切分按分隔符优先级逐级切分，语义切分基于句向量相似度在主题跳变处断开。

向量存储：VectorStore 抽象了向量库接口，LangChain 通过适配器对接 Chroma、Milvus、FAISS 等，统一 add_texts 与 similarity_search 接口。

检索增强生成（RAG）流程：文档加载、切块、向量化入库，查询时先把问题向量化，再做相似度检索取回 Top-K 片段，组装提示词后由模型生成答案。

评估：检索质量常用 Hit@K、Recall@K、MRR@K 等指标量化，衡量召回的是否为相关分片；生成质量则需要人工或 LLM 评判。
