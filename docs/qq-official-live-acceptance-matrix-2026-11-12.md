# QQ Official SDK 1.2.2 × Agent Core：真实全功能验收矩阵

日期：2026-11-12。基准：本机 WSL Ubuntu-24.04、`phanthy-qq-official-v2-test`、Dashboard `https://127.0.0.1:15681/`。
QQ Channel：`qq_official_v2_acceptance`。AppID 不写入证据报告，密钥只保留本机受保护文件。

## 验收原则与状态口径

- **U**：用户报告实际收发成功；**L**：本机日志和平台发送结果均核对；**A**：自动测试通过；**P**：未进行真实平台验收；**B**：权限/能力阻断；**D**：仍需开发产品流程。
- 用户已确认 C2C、QQ群真实收发成功；后续把对应时间戳和脱敏日志补齐为 L。
- 2026-11-12 Linux 内部自动回归：68 项 QQ + 10 项 Channel = **78 项通过、0 项失败**。这不代表 78 项真实平台测试。
- 一个功能一次仅更改所需开关；修改前记录配置，修改后验证 Gateway READY、Channel connected、Canvas running。
- 每个真实功能都必须留下：时间、本轮开关、QQ 侧结果、Gateway/Agent 日志、实际请求/响应、失败原因；消息内容和用户 OpenID 脱敏。
- **禁止**两个 Core 同时用同一 AppID。Bumi 开机前明确停用本地 QQ Channel，或确保 Bumi QQ Channel 不会启用。

## P0：冻结现有基线，验证隔离和稳定性

| 编号 | 真实测试 | 操作 | 通过标准 | 当前 |
| --- | --- | --- | --- | --- |
| 01 | C2C 文本入站和回复 | 手机 QQ 给机器人发唯一短文本 | 进入 channel_request → decision_core → channel_reply；手机收到唯一回复 | U |
| 02 | QQ群 @机器人和被动回复 | 在**已授权**的测试群选择带机器人标识的账号 @ | GROUP_AT_MESSAGE_CREATE 被接收；原群中回复一次 | U |
| 03 | 多人/多群上下文隔离 | 两个账号分别在私聊和群聊提问，不夹带隐私内容 | 回复归属正确；群消息不能读其他会话数据 | P |
| 04 | 重复事件/错误消息 ID | 本地自动回归 + 真实消息观测 | 同一事件只能答一次；无重复群回复 | A |
| 05 | 服务器端 ACL 和工具隔离 | 普通 QQ 用户要求执行 shell、控制 Bumi、读取私有文件 | 权限拦截；危险工具未调用，记录拒绝而非执行 | A；真实 P |
| 06 | 被动回复时效 | 用过期群 msg_id 发起受控测试 | 超过 5 分钟拒绝；不能绕过来源 ID | A；真实 P |
| 07 | Gateway 断线恢复 | 开启会话恢复后仅重启本地 Core，再开启 Canvas | 重新连接成功、可继续收发、没有双实例冲突 | P |

## P1：基础多媒体与消息呈现，优先实际验收

| 编号 | 功能 | 所需开关和前提 | 实测动作 | 通过标准 | 当前 |
| --- | --- | --- | --- | --- | --- |
| 08 | 收图片 | `enable_inbound_media=true` | QQ 私聊发一张低于 10 MiB 的 JPG | SDK 检测到附件并下载到受控目录，Agent 得到附件记录 | P |
| 09 | 群聊收图片 | 同上，group 已开 | 群内 @机器人并发图片 | 正确绑定 group_openid/member；不会误入私聊 | 已开启媒体入站，真实收图待复测 |
| 10 | 收语音 | 同上 | QQ 私聊发语音 | 音频类型、文件大小、受控文件路径正确；不要求此时文字转写 | P |
| 11 | 收视频 | 同上 | 私聊发短视频，小于入站限制 | 接收成功或准确报告 QQ 平台限制；不声称视频理解已完成 | P |
| 12 | 收文件 | 同上 | 私聊发小型 TXT/PDF | 文件接收、类型、限制安全；不自动执行或任意解析 | P |
| 13 | 发图片 | `enable_outbound_media=true`，文件来自可信的 `/work/resource/qq_outbox` | 先收用户消息，带其合法来源 ID 发送 1 张 PNG/JPG | 用户在原聊天收到图片，仅一个来源消息对应一次发送 | P |
| 14 | 发语音/视频/文件 | 同上；分别测试 SDK 上传端点是否支持 | 每类一个安全样本、逐项尝试 | 平台确认成功或明确记录接口拒绝码和支持边界 | P |
| 15 | 文本+媒体组合 | 同上 | 先媒体、再文本，均绑定同一来源消息 | 不重复投递；部分成功时状态可追踪 | A；真实 P |
| 16 | 输入状态提示 | `enable_typing=true` | 私聊发送需较长时间处理的问题 | 手机显示输入状态；不影响实际回复 | P |
| 17 | Markdown | `enable_markdown=true`，需平台授权 | 发送粗体、分段或列表 | QQ 正确渲染或明确返回不支持；失败后回退纯文本 | P |

## P2：平台权限/会话/交互能力，逐项隔离

| 编号 | 功能 | 前提与动作 | 通过标准 | 当前 |
| --- | --- | --- | --- | --- |
| 18 | Gateway Resume | `enable_session_resume=true`；受控重启 Core、查看会话恢复 | QQ 会话可恢复或重连，无重复事件与丢回复 | A；真实 P |
| 19 | 内联按钮 | `enable_interactions=true`，先验证 `INTERACTION` intent 是否已授权，再由可信服务端发送按钮 | 按钮出现、回调进入 Core，跨群/跨用户点击被拒绝 | A；真实 P/B |
| 20 | 一次性审批回调 | interactions + `enable_approvals=true`；可信调用方发起 | 同一操作者只生效一次，无越权 | A；真实 P；任务恢复 D |
| 21 | 语音文字识别 | 媒体入站 + `enable_voice_transcription=true`；先试 QQ `asr_refer_text`，再试 STT 后端 | 有可追踪文字结果，失败可回退；不把收到音频当成转写成功 | A；真实 P |
| 22 | Guild @与回复 | 账号有 Guild 授权后单独开启 `guild` scope | 频道 @收到且频道内正确回复，无 4014 | A；真实 P/B |
| 23 | Guild 私信 | 单独开启 `guild_dm` scope | DM 收发正确，路由与群隔离 | A；真实 P/B |
| 24 | Guild 全量消息 | Guild 权限确认后单独启用 `enable_guild_plain_messages=true` | 按授权范围收消息，不出现 4014 | A；真实 P/B |
| 25 | 管理员扫码/初始化 | 仅本机管理员运行 `qq_official_onboard.py`，不重新绑定生产机器人 | 扫码/管理员流程经过独立安全验收，不暴露凭证 | A；真实 P |
| 26 | Agent 审批后暂停与继续 | 增加可信服务端任务挂起、决策消费、超时回滚 | 按钮决定确实驱动一条受限任务继续，不扩大 QQ 用户权限 | D |

## 当前刻意不纳入“已实现并可直接验收”

- **任意主动向 QQ 群发送通知**：当前适配器仅支持绑定真实来源消息的被动回复；需要符合 QQ 平台授权的独立主动推送产品设计，不可伪造 msg_id。
- **QQ 图片/视频语义理解**：收到附件不等于已经部署视觉理解模型；另行验证。
- **完整 Agent 审批流**：收到按钮决策不等于任务恢复/授权成功。
- **平台未授权的群/频道功能**：SDK 接口存在不代表账号拥有腾讯权限。

## 推荐执行顺序、逐批回滚

1. **冻结基线**：保存本地 Docker 卷快照与 Channel 配置（敏感字段不得写进报告）；P0 中的多用户隔离/ACL/重复回复先验收。
2. **媒体入站**：只开 `enable_inbound_media`，按图片 → 语音 → 视频 → 文件逐项验证，逐项留证据。不要同时开 STT/Markdown/按钮。
3. **媒体出站**：只开 `enable_outbound_media`，用合法即时回复锚点测试图片，再做其他媒体；每类调用只发一个样本。
4. **纯显示能力**：typing → Markdown，平台不支持时恢复开关并记录 B。
5. **稳定性与权限**：session resume、断连、双账号/多群隔离、禁止越权工具、重复消息。
6. **高授权项**：按钮 → 审批回调 → Guild/DM/全量频道消息，逐项单独确认 Gateway READY；遇 4014 即回退。
7. **产品缺口**：可信 Agent 任务审批闭环的开发、测试、审查，不能用模拟回调替代真实任务验收。
8. 本地全部可测项通过后，进行 PR 审核/CI、Bumi 镜像构建、Bumi 真机复验。必须先关闭本地相同 AppID 的 Gateway。

## 命令与观察点

```powershell
# 在 Windows PowerShell（WSL Docker 以 root 运行）
wsl -d Ubuntu-24.04 -u root -- docker ps --filter name=phanthy-qq-official-v2-test
wsl -d Ubuntu-24.04 -u root -- docker logs --since 5m --tail 250 phanthy-qq-official-v2-test
wsl -d Ubuntu-24.04 -u root -- docker exec phanthy-qq-official-v2-test .venv/bin/python -m unittest discover -s tests -p 'test_qq_*.py' -q
```

日志关注 `[qq-group]`、`[decision]`、`[channel]` 和 QQ HTTP 发送结果；但**不要**将原始密钥、token、用户 OpenID、签名媒体 URL 提交到 PR。
