# Ollama 本地大模型

Ollama 是本地运行大模型的工具，一条命令即可下载并启动模型服务，支持 Embedding 模型与对话模型，适合离线开发环境。

模型管理：使用 ollama pull 拉取模型，ollama list 查看已安装模型，ollama run 进行交互式对话。

API 服务：Ollama 提供 HTTP 接口，默认监听 11434 端口，LangChain 等框架可通过 base_url 指向该地址调用。

Embedding：nomic-embed-text 是常用的本地向量化模型，支持中英文，维度为 768。

对话模型：qwen2.5 系列是中文效果较好的本地模型，按参数规模分为 3b、7b、14b 等版本，显存越大可运行越大参数。

性能考量：本地模型推理受 CPU/GPU 性能限制，并发能力弱于云端 API，适合开发调试与内网部署。
