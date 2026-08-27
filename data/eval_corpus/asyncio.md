# Python 异步编程

asyncio 是 Python 标准库提供的异步 IO 框架，通过事件循环调度协程，在等待 IO 时切换执行其他任务，提升并发吞吐。

协程：用 async def 定义的函数返回协程对象，需通过 await 调用；协程只有被事件循环执行才会真正运行。

事件循环：负责调度与执行协程的任务队列，asyncio.run 启动事件循环并运行主协程。

并发执行：使用 asyncio.gather 同时等待多个协程完成；create_task 创建后台任务，不阻塞当前流程。

线程池：同步阻塞函数可通过 asyncio.to_thread 放入线程池执行，避免阻塞事件循环。

超时控制：asyncio.wait_for 为协程设置超时，超时后取消任务并抛出超时异常。

并发安全：共享可变状态需要加锁保护，asyncio.Lock 提供协程安全的锁实现。
