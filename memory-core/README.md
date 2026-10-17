# Phanthymotus Memory Core

这是 Phanthymotus 的本地持久化记忆内核，支持 Python 3.10 及以上版本。
PR1 提供独立的 SQLite 存储、访问隔离、版本化修改、逻辑删除、幂等写入、
追加式变更记录和关键词检索；它尚未连接 Agent Core，也不提供 HTTP/MCP 服务。

合入后，调用方可以把结构化记忆写入独立的 `memory.db`，在进程重启后继续
读取和搜索，并通过 revision compare-and-swap 安全处理并发修改。

## 核心边界

- `private` 记录只属于创建者的 owner space。
- `shared` 记录放在命名共享空间；调用方必须显式拥有该空间的读或写能力。
- 所有隔离条件直接进入 SQL 查询，无权读取与记录不存在使用同一个错误。
- `memory_entries` 保存记录当前状态；删除是逻辑删除。
- `memory_changes` 是 append-only 变更序列，只保存 revision、actor、受限的
  非敏感审计标签和 operation key 摘要；不会自动复制标题或正文。
- `memory_requests` 保存写请求回执；相同 `op_key` 和相同载荷安全重放，
  相同 key 配不同载荷会被拒绝，原始 `op_key` 不落库。
- `memory_fts` 是可重建索引，不是事实源。
- 数据库文件权限固定为 `0600`，使用 WAL、事务化 migration 和有界锁等待。
- 共享空间成员关系和调用方身份验证由未来的 Agent Core 集成层负责。

## 使用示例

```python
from memory_core import AccessContext, MemoryDraft, MemoryPlace, MemoryStore

store = MemoryStore("memory.db")
context = AccessContext(owner_key="user:42", actor_key="user:42")

result = store.create(
    context,
    MemoryPlace.private(),
    MemoryDraft(title="饮品偏好", body="喝咖啡不加糖", kind="preference"),
    op_key="turn:2026-08-26:1",
)

record = store.read(context, result.record.memory_id)
```

## 本地验证

```bash
uv sync --locked --group test
uv run --locked --group test ruff check src tests
uv run --locked --group test ruff format --check src tests
uv run --locked --group test python -m pytest -q
uv run --locked --group test memory-core-selfcheck
uv build --clear
```

`memory-core-selfcheck` 默认使用临时数据库，验证私有隔离、重启恢复、修改、
逻辑删除、幂等回放、变更记录和 SQLite 完整性。

## PR1 不包含的内容

本阶段不包含 Agent prompt 注入、LLM 抽取、会话归档、embedding、知识图谱、
空间记忆、自动归纳、HTTP/MCP API、共享成员管理或既有数据导入。Agent Core
接入和产品级记忆策略将在后续 PR 中逐层完成。
