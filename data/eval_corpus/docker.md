# Docker 与 Docker Compose 部署

Docker 通过镜像与容器实现应用打包与隔离。镜像由 Dockerfile 定义，包含操作系统、依赖与代码；容器是镜像的运行实例。

Dockerfile 常见指令：FROM 指定基础镜像；RUN 执行构建命令；COPY 拷贝文件；WORKDIR 设置工作目录；CMD 定义容器启动命令；ENV 设置环境变量。

镜像分层与缓存：每条指令生成一层，构建时未变化的层可以复用缓存，因此通常把依赖安装放在代码拷贝之前，减少重复构建时间。

Docker Compose 用 YAML 文件描述多容器应用。services 定义各个服务，depends_on 声明服务启动顺序，volumes 持久化数据，ports 映射端口，healthcheck 定义健康探测。

数据卷用于持久化容器数据，容器删除后数据仍然保留；bind mount 则是把宿主目录挂载进容器，适合开发调试时实时同步代码。

常用命令：docker build 构建镜像；docker run 启动容器；docker compose up -d 后台启动整个应用；docker compose logs -f 查看日志；docker compose down 停止并保留数据卷。

健康检查：healthcheck 通过探测命令判断服务是否就绪，配合 depends_on 的 condition 可以控制服务启动顺序，避免依赖服务未就绪导致启动失败。
