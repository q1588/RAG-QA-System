# SQLAlchemy 与异步数据库

SQLAlchemy 是 Python 最流行的 ORM（对象关系映射）框架，用 Python 类描述数据库表结构，通过映射自动生成 SQL，避免手写大量 SQL 语句。

声明式模型：继承 DeclarativeBase 的类对应一张表，字段用 Mapped 与 mapped_column 声明类型、约束与索引。

会话（Session）：数据库操作的统一入口，负责事务管理。会话提交（commit）后修改落库，回滚（rollback）撤销未提交的修改。

异步支持：使用 create_async_engine 创建异步引擎，配合 AsyncSession 执行异步数据库操作，避免阻塞事件循环。

连接池：异步引擎内置连接池，pool_pre_ping 在取连接时探测掉线连接，pool_recycle 定期回收长时间未用的连接。

索引：合理建立索引能显著提升查询性能，联合索引按字段顺序匹配，覆盖高频查询条件可减少回表。

事务：一组操作要么全部成功要么全部回滚，保证数据一致性；ORM 会话默认开启事务。
