# QQ 官方 Channel 与飞书 Channel 的共用实现清单

状态基准：2026-11-12，本地 V2 测试容器。目标是仅实现平台协议适配，通用能力复用 Agent Core 原有组件。

| 能力 | 飞书已有组件/模式 | QQ 当前复用/差异 | 证据或待办 |
| --- | --- | --- | --- |
| 标准入站/出站消息 | `channel.adapter.InboundMessage / OutboundMessage` | QQ 使用完全相同的数据结构；附加 QQ 来源消息 ID | QQ 与 Channel 回归 |
| 用户 ACL、群/私聊隔离 | `channel.manager` 的权限、历史、回路路由 | QQ 复用；保留 QQ group_openid 和 member_openid 身份 | 真实群私聊已验收；多用户交叉测试待补 |
| 图片、视频、文件入站保存 | `channel.store.dir_for / maybe_prune / is_inbound_media` | QQ 下载后存入同一套 `resource/channel_files/<channel_id>/<day>`；共享 14 天、2GiB 清理策略 | 文件保存及跨 Channel 路径隔离测试；真实 QQ CDN 仍待现场验收 |
| 附件类型和尺寸 | `channel.media.infer_kind / resolve_outbound` | QQ 出站沿用统一文件解析；QQ 平台上传保留专用约束和独立信任的 outbox | SDK 类型常量已核验 1/2/3/4；平台实际上传待测 |
| 图像进入模型 | `event.desktop._read_image` 已可生成图像内容 | `channel.qq_vision` 只做安全来源校验与格式适配，不自行发明视觉模型 | **模拟 QQ SDK → 通用存储 → Collector → Flash API 识图实测 PASS（red）** |
| 决策流 | `collector → event.llm → channel_reply` | QQ 使用统一 Canvas 和决策工具，不另建自己的 Agent | 自动化通过；QQ 真群收图→回复待验收 |
| 一条消息与文件的部分成功 | 公共 `PartialSendError` + Manager 失败处理 | QQ 复用错误协议，保留 QQ 被动回复 5 分钟窗口和幂等策略 | SDK Mock 通过；平台失败码需补真实证据 |
| 长连接恢复 | 飞书 SDK 自动重连 + 独立事件 loop | QQ 使用腾讯 SDK 的 Gateway/Session Resume；不得复制飞书特有 WS 内部实现 | QQ Gateway 当前 connected；断网故障注入待验收 |
| @、消息 ID、安全 | 飞书按 open_id + message_id 严格路由 | QQ 必须使用 GROUP_AT、群 OpenID、被动 reply_to ID；保持平台专用 | 真实文本群回复成功 |
| 按钮/审批/Guild | 飞书交互流程可供 UX 与 ACL 参考 | QQ SDK 的 interaction intents、限次点击和 SDK 字段不同；不能照抄 API | 先做平台授权探测/独立真实验收 |

## 已修复的配置覆盖问题

原先只修改运行时 `event.llm.vision_input`，会在重启/启动画布时被 `tool_config:agentcore:decision_core` 的旧配置 `false` 覆盖。现在卡片配置与运行配置均为 `true`，并验证重启、Canvas 启动后仍保持 `true`。

## 本轮执行结果（不能与真实 QQ 全功能验收混淆）

- Linux QQ 单测/契约测试：88 项通过、0 失败；Channel 测试 10 项通过。
- 真实 DeepSeek Flash 多模态接口已接受有效 PNG 并回答 `red`。
- 本地模拟平台链路：QQ SDK 群图片事件、QQ 附件下载、飞书共享存储、Collector、图片内容格式转换、Flash 推理：PASS。
- QQ Gateway connected；Canvas running=true；QQ inbound media true。
- 尚未宣称真实群图片回复、媒体上传或平台按钮/Guild 权限验收通过。

## 工程约束

1. 腾讯特有能力只放 `qq_official*.py`；不要再给 QQ 独立建另一套 ACL、历史、文件清理或图片识别模块。
2. QQ 用户角色保留服务端 viewer 限权，图片路径仅允许本 Channel 在 `channel_files` 内的实际下载文件，限制图片大小和数量；多模态内容块不得以 base64 形式进入请求日志。
3. 一个 QQ AppID 同时只运行一个 Gateway；本机验收通过后再执行 Bumi 迁移。
4. 完整验收要求先做自动/模拟测试，再做少量真实平台收发；把账号权限受限与代码失败分开记录。
