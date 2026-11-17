# 待修复：多 QQ Bot 的 LLM 工具重名与回复路由串线风险

- **状态**：OPEN / 暂缓处理（用户明确要求先记录，当前不修复）
- **记录日期**：2026-11-17
- **优先级**：P0（恢复多 Bot 稳定收发前必须修复）
- **范围**：本机 Agent Core，QQ 官方 SDK；需兼容飞书以及其他 MCP Card
- **当前策略**：不得为解决本问题重新扫码、删除已配置 Bot/Card、改动 Watch7 Driver 或 Bumi

## 现象和证据

1. 当前画布有多个 `channel_request` / `channel_reply` 实例，各实例可选择不同的 QQ 官方 Bot；三个 Bot 的例子分别为 `qq_official_1906320747`、`qq_official_1906443615`、`qq_official_1906322115`。
2. 原先单 Bot 能够完成消息触发、LLM 调用、`channel_reply` 并成功发送；多 Bot 配置后，曾出现消息触发和 LLM 请求，但没有发送回复。
3. 之前的问题排查记录中出现 LLM HTTP 400：`Tool names must be unique`。这是当时的证据，**本次仅归档，未重新进行线上复现**。
4. `agent-core/src/channel/qq_default_flow.py` 为每个 Bot 的输出实例建立指向同一 `channel_reply` 工具的 `execConnections`。
5. `agent-core/src/event/llm.py::_get_bound_tool_schemas()` 遍历 `execConnections`，对同一 `full_name` 多次 `schemas.append(presented)`；同时使用 `self._bound_instance_ids[full_name] = card_id`，同名工具的实例会被后面的 Card 覆盖。
6. `agent-core/src/channel/manager.py::send_reply()` 通过 `instance_id` 解析回复渠道。**因此只做全局工具去重可能解除 HTTP 400，但仍有潜在的跨 Bot 错误路由风险**。跨 Bot 实际发错消息尚未证明，必须按风险测试。
7. 现有入站消息已携带 `channel_id`、`chat_id`、`message_id`；`channel_message_contexts` 已按 channel/message 保存上下文，可复用，而不必重写整个 Gateway。

## 修复方向（尚未实施）

- **局部修复工具注册**：仅对多个 `channel_reply` Card 的相同 LLM 工具定义进行规范化，保证 LLM 可见工具名唯一。不要直接对所有 MCP 多实例工具做全局去重。
- **消息级路由**：以当前轮次**可信**的 `source_message_id` 反查 `channel_id` / `chat_id`，由后端匹配对应的 `channel_reply` 实例并注入 `instance_id`；缺失、过期、跨轮次、不匹配、歧义时拒绝发送，绝不退回「最后一个 Bot」。
- **保持原有能力**：保留用户现有三个 Bot、七张 Card、飞书 `trusted_bot_id` 主动消息语义、ACL、媒体附件、安全凭证、Watch7 及 Bumi；对飞书主动发送分支单独保留兼容逻辑。
- **并发与记忆隔离**：默认按 `platform + channel_id + chat_type + chat_id` 隔离聊天；每次处理轮次使用不可变可信路由上下文，禁止全局最近消息作为并发路由依据。检查 QQ 现有 `qq_memory_scope` 和批处理逻辑是否满足要求。

## 改动预计涉及

- `agent-core/src/event/llm.py`：去重与当前轮次路由、实例选择
- `agent-core/src/channel/manager.py`：可信消息定位与实例/通道验证
- `agent-core/src/collector.py`：仅在确有缺口时补充消息来源映射
- `agent-core/tests/`：新增严格的单元、并发、回归和拒绝发送用例

**不预先修改** `mcp_client.py`、QQ 官方 SDK、扫码授权和现有容器架构；必须以测试暴露的必要性为依据。

## 验收要求

- 单 Bot 与三个 Bot：LLM 工具名唯一，所有 Bot 均能在自己的聊天窗口回复。
- 三 Bot 并发：每个消息 ID 均路由到正确的 Adapter/聊天窗口；重复消息、乱序回复不串线。
- QQ 私聊、群聊及跨用户：对话、长期记忆、权限和回复隔离。
- 飞书、飞书 Bot-to-Bot 的既有能力无回归；其他 MCP/Card 多实例行为不变。
- 未绑定、伪造或过期的 `source_message_id`，跨 Bot 的 `instance_id`，都被拒绝发送。
- 本地已有改动先备份并建立基线，Mock 测试 → 容器自动化回归 → 真实三 Bot 收发 → 正式镜像/PR；任何串线立即回滚。

## 开发约束

**用户已明确：先搁置，当前只记录问题，不进行代码修复、重启、重新部署或提交 PR。**

后续恢复时应先阅读此文档，并重新核对代码实际版本、未提交改动和当时日志；不要仅以本记录替代现场验证。
