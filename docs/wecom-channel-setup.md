# 企业微信 Channel 配置与收发验收（一期）

> 本文按 [`feishu-channel-setup.md`](./feishu-channel-setup.md) 的结构编写，面向 `platform = wecom` 的渠道。
> **状态：代码已实现（2026-11-14），待远程部署与真机验收。** `wecom` 适配器、6 处接线、单测均已落在 `feat/wecom-channel` 分支；本地只做静态检查（语法 / 编译 / 接线核对 / 离线单测），**不做本地运行时联调**。按项目部署方式，需先把分支推到远程、获取包含新代码的镜像，再部署到机器人上验证。§6 已按实现的错误文案补齐，但标注「待实测」的帧字段仍需 L0 帧样本确认。
> **一期范围**：组织内 `operator` / `viewer` 两类用户、单台 bumi、自然语言文本命令、「发命令 → 机器人执行 → 回复原会话」一条主链路。不含附件收发、主动推送、Bot @ Bot / Trusted Bots、模板卡片、移动端入口与个人微信触达，这些留待后续调整。
> **一期目标**：对齐 feishu-bot 的交互方式 —— 不是做一个聊天问答机器人，而是让企业微信里的一句自然语言命令，经 `decision_core` 调用已绑定的 driver actuator，**驱动真实机器人（bumi）执行动作**，并把结果回复到原会话。

本文说明如何从零创建企业微信智能机器人，并把它接入 Phanthy Motus Agent Core。与飞书一样走 **WebSocket 长连接**：设备只需能出网，不需要公网 IP、回调域名或消息加解密。

## 0. 一期操作流程总览（从扫码到 bumi 执行命令）

> 按「个人组建团队 / 团队形态」这条路径整理。第 4 步的帧样本不通就不要进第 5 步 —— 这是飞书 #125 修「假 connected」得到的教训。
>
> **本地与远程的分工**：本地只做静态检查与代码准备，**真机联调全部在远程镜像部署到机器人之后**。本地不启动 agent-core、不配渠道、不连企微长连接。

| 步 | 在哪操作 | 做什么 | 通过标准 | 详见 |
|---|---|---|---|---|
| 1 | 官网注册页（已完成） | 用**个人实名微信扫码**完成「个人组建团队」 | 团队创建成功，能登录电脑端与管理后台 | §1.1 |
| 2 | 企业微信客户端（已完成） | 装 App 本体 ≥5.0 →「工作台 → 智能机器人 → 创建机器人 → 手动创建 → **API 模式**」→ 连接方式选 **使用长连接** | 已生成 **Bot ID + Secret** | §1.2 |
| 3 | 同上 | 填名称、可见范围（只含自己）、使用模式，保存；**立即保存凭据** | 通讯录能找到机器人，可发起单聊 | §1.2 |
| 4 | 本地终端 | 最小 WebSocket 脚本（临时验证物，不入库）：订阅 → 收 `aibot_msg_callback` → 回显一句话；同时验证 443 出网。**记录脱敏帧样本** | 企微收到回显；帧字段名与 §8 常量区逐项核对 | §8 / P0 / L0 |
| 5 | 本地 · `phanthymotus` · `feat/wecom-channel` | **只做静态检查**：`py_compile`、`node --check`、`OutboundMessage` 字段位置核对、6 处接线核对、离线单测（`python -B tests/test_wecom_channel.py -q`） | 静态检查全绿、离线单测通过 | §8 |
| 6 | 远程 | 把 `feat/wecom-channel` 推到远程；**获取包含新代码的镜像**并部署到机器人。`Dockerfile:44` 是 `COPY src/ /work/src/`，新增的 `adapters/wecom.py` 会随镜像进入；`pyproject.toml` 无需改动（`aiohttp` 已是依赖） | 机器人上运行的是含 `wecom` 的新镜像 | §8 |
| 7 | 机器人上的 Agent Core Dashboard | 「设置 → 渠道」加 WeCom Channel 填 `bot_id` / `secret`；Canvas 放 **2 张复用卡 + 2 条连线**。动作级限制落实前，先只绑定只读能力 | 渠道 `connected`、两卡自检通过 | §2 §3 |
| 8 | 「渠道 → 用户」 | 把你的 userid 从默认 `viewer` 改成 **`operator`** | 角色生效；`viewer` 仍只读 | §4 |
| 9 | 企微 + bumi 现场 | **动作级限制落实后**，发「挥个手」→ bumi 执行 → 原会话收到回复；再用 `viewer` 发同一句验证被拒 | 飞书 F11 四要素 + 动作真的执行 | §5 |

第 1–4 步是凭据与帧样本，第 5 步是本地静态验证，第 6 步是远程部署，第 7–9 步在机器人上验收。**全程不需要营业执照、企业认证、公网 IP 或向腾讯提交申请。**

## 1. 创建并配置企业微信智能机器人

### 1.1 选择测试主体

飞书文档 §1 是「新建企业自建应用 + 发布版本 + 可用范围」；企业微信对应的是「创建智能机器人 + API 模式 + 可见范围」。一期需要先确定在哪个企业里测：

| 方式 | 前提 | 适用 | 状态 |
|---|---|---|---|
| **A. 个人组建团队 / 团队形态（推荐）** | 手机端「选择以下类型全新创建」，或**电脑端官网注册页底部「个人组建团队」**（即当前截图这一步）；用个人实名微信扫码完成，免费、无需营业执照、无需认证，你就是创建人 | 高校学生 / 个人独立开发测试，不触碰学校组织，零审批 | 【已确认·官方 21059】团队形态「同样支持企业形态账号的各功能，包括登录电脑端、登录管理后台等」，且「**团队形态支持创建智能机器人**。可在【工作台 → 智能机器人 → 创建智能机器人】」；【待实测】团队形态下 API 模式长连接入口与 Bot ID / Secret 是否完整可用 |
| **B. 电脑端注册未认证企业 / 团队主体** | `work.weixin.qq.com` → 立即注册 → 填企业名称 / 行业 / 规模 → 管理员实名微信扫码；注册者自动成为超管 | 与上等价，管理后台菜单更完整 | 【已确认】官方 QClaw 接入指南：「如果您还没有企业微信账号，可以前往企业微信官网免费注册一个企业」 |
| **C. 使用所在组织的企业微信（普通成员）** | 官方帮助中心 21663「企业超级管理员、被授权的企业成员均可创建智能机器人；**成员创建权限默认全员开放**」，v5.0 起手机端 / 电脑端工作台均可创建；管理员可在「管理后台 → 智能专区 → 智能机器人 → 权限管理 → 机器人创建权限」按部门 / 标签 / 成员收紧 | 组织未收紧权限、且允许在组织内做开发测试 | 【待确认】本校是否开放成员创建入口、是否允许在校内做开发测试 |

> **修订说明**：早期第三方资料称「个人组建团队没有管理后台、无法创建智能机器人」，与官方帮助中心 21059 冲突，以官方为准。团队形态是**可用**的开发主体。第三方资料仍提到的对外限制（不能添加微信外部客户、客户朋友圈等）不影响内部单聊 / 群聊收发，一期也不涉及这些能力。
>
> 团队形态的已知边界（官方 21059）：一个账号最多作为 5 个团队形态账号的创建人；未认证团队只支持修改一次团队名称；历史团队形态账号不支持登录 Mac（5.0.11 后新建的可登录）。这些都不影响一期。
>
> **规模与能力**：官方开发者社区说明「对于 10 人及以下的个人 / 小团队，API 模式智能机器人提供消息、文档、日程、会议、待办等 MCP 能力」，与「10 人以上规模企业…仅提供文档 MCP 能力」不同。个人团队正好落在前者，能力最宽；不过一期只做长连接消息收发，不依赖这些 MCP 能力。

### 1.2 创建机器人

1. 安装 **企业微信 App 本体**（≥ 5.0），不要用微信里的「微信插件」登录 —— 插件形态看不到智能机器人。团队形态同样可登录电脑端，v5.0 起手机端 / 电脑端工作台都能创建。
2. 打开 **工作台 → 智能机器人 → 创建机器人 → 手动创建**（团队形态官方路径为「工作台 → 智能机器人 → 创建智能机器人」）。
3. 选择 **API 模式创建**。创建页通常要**拉到底部**才能看到该入口，提示语为「如需使用自有系统获取成员与机器人的聊天并输出回复，可切换至 API 模式创建」。**只有 API 模式才支持长连接与自有 Agent 接入**；选「AI 自动生成」或「一键创建」的机器人没有后续授权入口。
4. 在 API 配置页把连接方式选为 **「使用长连接」**。官方长连接文档 60904：适用场景「内网部署、对实时性要求高、需要主动推送消息」，无需域名 / IP；与「设置接收消息回调地址」**二选一**，切换会使另一种方式失效。
5. 页面生成 **Bot ID**；**Secret 需点击「点击获取 / 获取」按钮才会显示**，不是默认可见。两者是长连接专用凭据，与 URL 回调模式的 Token / EncodingAESKey 是两套（官方 101805）。**立即保存到本地密码管理器**；是否只展示一次各方资料说法不一，按一次性处理，不要贴进对话、仓库、文档或截图。
6. 配置 **可见范围**（只含实际测试者，对应飞书文档 §1 的「可用范围」）与 **使用方式**：
   - 一期只自己测 → 「**仅个人使用**」最安全。
   - 需要第二个人验 `viewer` / `operator` → 「**多人使用**」；但官方 21663 提示此时权限模式默认「共享创建者的权限」，非创建人的可使用成员可读取创建人的文档 / 邮件 / 待办等，需谨慎选择范围。测试团队无真实数据，风险可控；也可在同一账号上切角色代替。
7. **可使用权限**（消息 / 邮件 / 文档 / 待办 / 日程 / 会议 / 微盘 / 通讯录 / 用户信息，默认含消息）：这是让机器人读取企微内部数据的能力，属于 CLI / MCP 场景；一期适配器只做长连接消息收发，不依赖这些。是否必须显式勾选「消息」才能收到回调，**待实测**。
8. 保存机器人配置。5.0.12 起可在通讯录找到机器人，直接发起单聊，或把它拉入内部群。

> **API 模式暂不支持预览与调试**（官方帮助页 21677 原文）。因此连通性必须用第 5 节的验收步骤或最小 WebSocket 脚本验证，不能依赖企微自带的调试入口。
>
> **UI 顺序差异**：官方 21677 是「先选长连接 → 生成凭据 → 补可见范围 → 保存」；部分客户端与第三方实测是「先填名称 / 可见范围 → 保存 → 再到机器人编辑页右侧『API 配置』选长连接并取凭据」。以你客户端实际界面为准，不要因为没立刻看到凭据就改建成 URL 回调模式。
>
> **一个易混淆点**：有第三方文章称「企业未认证无法生成 EncodingAESKey，必须先实名认证」——这只针对 **URL 回调模式**。长连接用 Bot ID + Secret，不涉及 AESKey，未认证主体不受此限；官方开发者社区技术支持亦明确「对于非认证企业，暂时没有域名 / IP 的限制」。

### 1.3 无需申请的事项（与飞书 §1 对照）

| 飞书文档 §1 需要 | 企业微信长连接 | 说明 |
|---|---|---|
| 5 项应用身份权限（`feishu-bot-permissions.json`） | **不需要** | 长连接无 scope 概念，也不需要对应的 JSON 清单文件 |
| 事件订阅 `im.message.receive_v1` | **不需要** | 消息由长连接直接推送 `aibot_msg_callback` |
| 订阅方式选「使用长连接接收事件」 | 已由 API 模式涵盖 | — |
| 应用版本发布 / 可用范围发布 | **不需要** | 但仍需配置机器人「可见范围」 |
| Token / EncodingAESKey 与消息加解密 | **不需要** | 长连接明文 |
| 公网回调 URL / 可信域名 / IP 白名单 | **不需要** | 纯出网连接 |

设备侧唯一的网络前提：**能访问 `wss://openws.work.weixin.qq.com:443`**。校园网或实验室网络是否放行需实测。

### 1.4 平台侧硬约束

- 每个智能机器人**同一时间只能保持一条有效长连接**；新连接完成订阅会踢掉旧连接，旧连接收到 `disconnected_event`。联调与正式环境不要同时连同一个 Bot ID。
- 单企业最多 300 个智能机器人，单个普通成员最多创建 20 个（官方帮助中心）。
- 长连接建议每 30 秒发送一次 `ping` 保活，需实现断线检测与自动重连。
- 当前适配器按 24 小时保留回复上下文，流式消息仅发送最终帧。回复首包的实际时限仍需联调：腾讯云 [CodeBuddy 接入指南](https://intl.cloud.tencent.com/zh/document/product/1316/84454)建议在 5 秒内发送流式首包，不能仅凭本地缓存时长认定平台一定会接受回复。
- 一期不支持附件：入站媒体 url 仅 5 分钟有效且每条带独立 `aeskey`，出站需走上传临时素材三段式，均留待后续。

## 2. 在 Agent Core 中添加 WeCom Channel

1. 打开 Agent Core Web Dashboard。
2. 进入 **设置 → 渠道 → + Add Channel**。
3. 填写：

   | 字段 | 值 |
   |---|---|
   | **Platform** | `WeCom (企业微信)` |
   | **ID** | 稳定、可辨识的 ASCII 名称，例如 `bumi_wecom`（ROS-safe topic 为 `/channel/request/bumi_wecom`） |
   | **Bot ID** | 创建机器人时生成的 Bot ID |
   | **Secret** | 同页生成的长连接专用 Secret |
   | **Enabled** | 开启 |

4. 保存后确认 Channel 状态为 `connected`。若先在 Core 中添加 Channel、后在企微侧完成机器人配置，可在渠道面板点击 **Restart**。

**与飞书 §2 的差异**：WeCom Channel **没有** `Allow group Bot @ Bot` 开关，也**没有** `Trusted Bots` JSON 配置 —— 企业微信长连接没有 Bot @ Bot 对等概念，一期不提供这两项。

Secret 只应保存在设备的 Agent Core 配置中，不要写入仓库、文档、截图或聊天记录；泄露后应在企微后台重置并立即更新 Channel。

## 3. 把 Channel 接入 Decision Core

仅在「设置 → 渠道」中添加 Channel，只会建立连接；消息不会自动进入 Agent。需要在 Canvas 中完成两条连接：

1. 停止当前智能控制，进入 Canvas 编辑状态。
2. 添加 `Channel / channel_request` 卡片，在实例配置中选择刚创建的 Channel。
3. 将 `channel_request` 的 `data/json` 输出连接到 `AgentCore / decision_core` 输入。
4. 添加 `Channel / channel_reply` 卡片，在实例配置中选择**同一个** Channel。
5. 从 `decision_core` 底部的 **执行器** 端口连接到 `channel_reply` 卡片。
6. 启动智能控制，确认两个 Channel 卡片均通过启动自检。

链路如下（**与 feishu-bot 完全同构**，只是把飞书换成企业微信长连接）：

```text
企业微信用户（operator）
  → 一句自然语言命令（单聊 / 群聊 @）
  → 长连接 aibot_msg_callback
  → ChannelManager / topic /channel/request/<channel_id>
  → channel_request (sensor 卡)
  → decision_core（LLM 选工具）
  → mcp__<driver>__<actuator>   ← 这里才真正驱动机器人
  → phanthymotus-driver MCP → DDS → bumi 执行动作
  → channel_reply (actuator 卡，唯一出口)
  → 企业微信原会话收到回复
```

要点：

- **通讯层不含任何机器人动作**。`channel_request` / `channel_reply` 只是把消息送进 Agent、把结果送出去；动作由 `decision_core` 调用 Canvas 上已绑定的 driver actuator 产生。`phanthymotus-driver` 一行都不用改（bumi 的 `device.py` 已声明 `motion_state` / `semantic_action` 等卡）。
- **没有斜杠命令表 / 按钮回调**：「命令」就是自然语言，由 LLM 决定调哪个已绑定工具。
- `viewer` 回合在第 7 步被只读闸拦下，不会到达 actuator（见 §4）。
- `channel_reply` 回复当前消息时使用 `source_message_id` 找回精确会话，不会被后到的群聊或单聊改写目标。**一期不使用** `trusted_bot_id`、`mention_open_id` 与 `expect_reply`。

**一期动作范围与当前实现**：

| 角色 | 绑定到 `decision_core` 执行器的卡 | 说明 |
|---|---|---|
| `viewer` | 共用 Canvas 绑定，受只读闸限制 | 只能查 sensor / resource、只读系统工具并回复 |
| `operator` | 目标是只读 `motion_state`，以及挥手 / 握手 | 当前没有动作级白名单；直接绑定完整 `semantic_action` 卡也会暴露欢呼和舞蹈等动作 |

当前角色权限只区分是否可调用已绑定的 actuator，**不会限制 `semantic_action` 的具体动作**。若一期只允许挥手和握手，须先在执行侧落实并验证动作级限制；在此之前，不应将完整 `semantic_action` 卡连接到真机执行链路。

一期目标不包含 `loco`、`loco_relative_move`、`stand_up_lie_prone`、`action_recording` 或舞蹈（需 3m×3m 空间）。

> `viewer` / `operator` 是 phanthymotus 侧 `channel_users` 的角色，**不是**企业微信成员角色，不需要在企微后台申请任何身份。

## 4. 触发与回复规则

飞书文档 §4 整节是 Bot @ Bot 协作语义，企业微信一期没有对应能力，本节替换为触发与权限规则：

- **单聊**：直接向机器人发消息即可触发。
- **群聊**：长连接只推送 **明确 @ 机器人** 的消息（平台设计）。
- **机器人单聊**：不触发。
- **命令形态**：没有斜杠命令表或按钮回调，"命令"是自然语言，由 `decision_core` 决定调用哪个已绑定工具；一期不使用卡片按钮事件。
- **角色权限**：

  | 角色 | 可调用 |
  |---|---|
  | `viewer`（默认） | `sensor` / `resource`、`finish`、只读系统工具（WebSearch / search_history / memory_recall / raw_input_info）、`channel_reply` |
  | `operator` / `owner` | 上述全部 + 已绑定的 actuator / processor / Skill |

  新用户默认 `auto_approve = true`、`default_role = viewer`，**必须手动改成 `operator` 才能让机器人执行动作**。

- **回复出口**：`channel_reply` 是 Agent 向企业微信发消息的唯一出口；写在 `content` 里的纯文本不会送达。回复必须逐字复制触发事件的 `source_message_id`（每个 Channel 保留最近 100 条上下文）。
- **一期不支持**：附件、`trusted_bot_id`、`mention_open_id`、`expect_reply`、无触发的主动推送、跨群回复。

## 5. 五分钟收发验收

1. 在企微通讯录或搜索中找到机器人；找不到时先检查**机器人可见范围**、是否装了企业微信 App 本体。
2. 向机器人单聊发送：`请只回复：企微链路测试成功`。
3. 在 Agent Core Activity 中确认收到来自 `channel:wecom` 的触发事件。
4. 确认机器人回复 `企微链路测试成功`。

验收通过需要同时满足：**Channel 为 `connected`、Activity 有真实入站事件、Decision Core 实际运行、企微收到真实回复**。只看到容器运行或 WebSocket 连接不算完整收发闭环。

### 一期 bumi 真机动作验收

前提：已在执行侧落实并验证挥手 / 握手的动作级限制。当前完整的 `semantic_action` 卡不提供此限制。

1. 用 `operator` 账号发送一句自然语言命令，例如「挥个手」。
2. Activity 中确认入站事件携带 `channel:wecom`，`decision_core` 实际运行并调用了已绑定的动作卡。
3. bumi 执行 `semantic_action` 的挥手 / 握手；执行前确认**已正常站立、双脚着地、地面平坦防滑且周围无人和障碍物**（卡片描述原文）。
4. 企微收到对原消息的回复，不是发到别的会话。
5. 换 `viewer` 账号发送同一句命令：动作应被只读闸拒绝，只收到可读的说明，不应出现任何执行工具调用。

验收截图至少包含渠道开关、企微消息与 Activity 状态；**不得截入 Secret、token 或真实敏感消息正文**。

## 6. 常见错误

> 下表「适配器报错原文」来自 `agent-core/src/channel/adapters/wecom.py` 的实现；帧字段以 L0 实测为准（见 §8）。

| 表现 | 适配器报错原文 / 成因 | 处理 |
|---|---|---|
| 渠道起不来，凭据为空 | `WeCom bot_id and secret are required. ...` | 在「设置 → 渠道」填 Bot ID / Secret；确认机器人是 API 模式长连接，不是 URL 回调 |
| 订阅被拒 | `WeCom subscribe rejected (errcode=N, errmsg=...)` | 核对 Bot ID / Secret、API 模式；不要用 URL 回调的 Token / EncodingAESKey。异常文本不会带出 Secret |
| 订阅超时 | `WeCom subscription timed out. Check outbound access to wss://openws.work.weixin.qq.com:443 ...` | 检查机器人 443 出网与代理；`start()` 会抛异常让框架重试，不会假 connected |
| 连接反复断开 / `disconnected_event` | `long connection taken over by another process for the same Bot ID; automatic reconnect disabled` | **适配器会主动停止自动重连**（防止两进程互踢）。停掉另一个进程，在渠道面板 **Restart** |
| 长时间空闲后掉线 | `heartbeat stalled: no frame received recently` / `receive task is dead — restart the channel` | 检查 30s `ping` 与 443 出网；`status()` 会显示 `degraded` 或 `disconnected` |
| 无法发消息 | `Cannot send: WeCom adapter is not connected. ...` | 渠道面板 **Restart**；确认订阅 ACK `errcode=0` |
| 回复找不到上下文 | `No reply context for this message: the inbound req_id is unknown or expired. The user must send a new message first.` | 让用户重新发一条消息；不要复用历史 `source_message_id` |
| 回复超窗 | `Reply window expired (24h). Re-trigger from a current message ...` | 基于当前 Activity 事件重新触发 |
| 回复被拒或回执超时 | `WeCom reply rejected ...` / `WeCom reply ACK timed out.` | 检查错误码、连接和消息上下文；发送结果以平台回执为准 |
| 尝试发附件 | `WeCom channel is text-only in phase 1; files are not supported.` | 一期不支持附件；框架已按 `SUPPORTED_FILE_KINDS=()` 提前过滤 |
| 非文本入站 | 适配器给占位文本 `[收到非文本消息，当前仅支持文本指令]` | 一期只处理文本；Agent 会据此回复可读说明，而不是静默 |
| Channel 显示 `connected` 但消息无反应 | Canvas 没有配置 `channel_request` 卡片，或智能控制未启动 | 检查卡片实例配置与启动自检；未激活 channel 的入站消息会被丢弃 |
| Agent 能收到但不能回复 | `decision_core` 未绑定 `channel_reply`，或尚无会话上下文 | 补执行器连接；先由用户发一条消息建立上下文 |
| 企微里找不到机器人 | 不在机器人可见范围，或用「微信插件」登录 / 未装 App 本体 | 调整可见范围；改装企业微信 App 本体 |
| 群聊里 @ 了但没有事件 | 机器人未加入群，或 @ 不是明确指向该机器人 | 把机器人加入内部群并明确 @ |
| 发了命令但机器人不动 | `user_role` 仍是默认 `viewer`，或 `decision_core` 未绑定 `operator` 的动作卡 | 在「渠道 → 用户」把角色改为 `operator`；核对第 3 节的绑定表 |
| 凭据保存后找不到 | Bot ID / Secret 可能仅展示一次 | 在企微后台重置 Secret 并更新 Channel |
| 无法连接且无任何日志 | 机器人所在网络未放行 `wss://openws.work.weixin.qq.com:443` | 先在网络侧验证，再排查代码 |

## 7. 最小安全边界

- 默认只开单聊所需能力；群聊按需增加。
- 不把 **Secret、Bot ID 对应的凭据、token 或用户消息**写入 Git、文档、截图或聊天记录。
- 机器人**可见范围**只包含实际测试者；扩大到全员前先确认数据与行为边界。
- `channel_reply` 是 Agent 向企业微信发送消息的**唯一出口**；普通人工消息行为不变。
- **单 Bot ID 单连接**：联调与正式环境不共用同一对凭据。
- `viewer` 只读；`operator` 可调用已绑定的 actuator。挥手 / 握手的动作级白名单尚未实现，真机接入前必须先落实并验证。
- **不开放底层电机级控制**：Bumi 的 HighController 能力边界「原则上不超过 APP 或手柄原本可实现的功能」；跌倒保护属于底层安全机制，「具有最高优先级，不受上层控制指令影响」。
- 执行动作前确认机器人站立、双脚着地、地面平坦防滑且周围无人无障碍。
- 一期不配置 Trusted Bots，也不接受任何自报的机器人身份或跨群回复。

## 8. 本地静态检查与远程镜像部署

### 8.1 本地只做静态检查

本地不启动 agent-core、不配渠道、不连企微长连接。已执行的静态检查：

| 检查 | 命令 / 方法 | 结果 |
|---|---|---|
| Python 编译 | `python -m py_compile src/channel/adapters/wecom.py src/channel/adapter.py src/channel/manager.py src/channel/__init__.py src/start.py tests/test_wecom_channel.py` | 通过 |
| JS 语法 | `node --check web/js/channels.js` | 通过 |
| `OutboundMessage` 兼容性 | AST 核对 `source_message_id` 位于字段末尾且带默认值 | 通过（现有 4 个适配器不受影响） |
| 6 处接线 | 字符串核对 `register_adapter('wecom')`、`source_message_id=source_message_id`、`Supported: ... lan, wecom`、两张卡描述、`channels.js` 下拉/表单/图标 | 通过 |
| 离线单测 | `cd agent-core && python -B tests/test_wecom_channel.py -q` | 使用模拟 WS；以本次运行结果为准 |

> 全量 pytest 在 Windows 本地会有收集错误：`fcntl` 是 Unix-only、`cryptography` 未在 `pyproject.toml` 声明、`test_skill_files.py` 的 `PermissionError`。这些是既有环境问题，与 wecom 改动无关，**不作为本地验收依据**；以远程镜像内的运行结果为准。

### 8.2 远程部署前提

按项目部署方式，**先把 `feat/wecom-channel` 推到远程，获取包含新代码的镜像，再部署到机器人**：

| 项 | 结论 | 依据 |
|---|---|---|
| 新增文件会进镜像吗 | **会**。`agent-core/Dockerfile:44` 是 `COPY src/ /work/src/`，且无 `.dockerignore` 排除 `src` | `adapters/wecom.py` 随镜像进入 |
| 需要改依赖吗 | **不需要**。`pyproject.toml:7` 已声明 `aiohttp>=3.13.5`，镜像构建那步 `uv lock && uv sync --no-dev --no-install-project` 无需变更，`uv.lock` 也不动 | 裸 WS 零新增依赖 |
| 需要重建镜像吗 | **需要**。新代码必须经构建/拉取的新镜像才能到机器人；仅改本地文件不会生效 | `Dockerfile` 的 COPY 层 |
| 重启后要做什么 | 新镜像启动后 `channel/__init__.py` 会注册 `wecom`；随后在 Dashboard 加 Channel、接 Canvas、启动智能控制（现场 `auto_start=false`，重启后需手动启动） | §0 第 7–9 步 |
| 凭据在哪填 | **只在机器人上的 Dashboard「设置 → 渠道」填**，不进仓库、不进镜像、不进日志 | §2、§7 |
| 单连接约束 | L0 脚本与机器人上的适配器不能同时连同一个 Bot ID，否则互踢；部署后不要再跑本地 L0 | §1.4 |

### 8.3 帧格式待 L0 实测

官方 `101463` / `60904` 页面正文是动态渲染的，静态抓取只能拿到目录导航，拿不到逐字 JSON 示例。当前 `wecom.py` 的帧字段名来自官方开发前必读 101805、帮助中心 21677 与官方 SDK 行为描述，属于**有依据的假设**：

| 常量 / 位置 | 当前值 | L0 需核对 |
|---|---|---|
| `WSS_URL` | `wss://openws.work.weixin.qq.com` | 端点 |
| `CMD_SUBSCRIBE` / 凭据字段 | `aibot_subscribe` / `bot_id` + `secret` | 订阅帧结构与 ACK `errcode` |
| `CMD_MSG_CALLBACK` | `aibot_msg_callback` | 入站 cmd |
| 入站字段 | `msgid` / `chattype` / `from.userid` / `text.content` / `chatid` | 单聊是否真无 `chatid`、`headers.req_id` 位置 |
| `CMD_RESPOND` / `REPLY_MODE` | `aibot_respond_msg` / `stream`（`stream.id` `finish` `content`） | 是否必须流式，普通 `text` 是否可用 |
| `CMD_PING` / `HEARTBEAT_INTERVAL` | `ping` / 30s | 心跳帧与服务端响应 |
| 入站回调 ACK | 不发送额外的裸 ACK，参照官方 SDK | 与真实服务端联调核对 |
| `EVENT_DISCONNECTED` | `disconnected_event` | 被接管事件的实际名称与所在层级 |

协议常量和字段解析集中在 `adapters/wecom.py`；L0 联调仍需核对订阅、消息和回复回执的完整行为。回复帧发送后会等待平台 ACK，非零错误码或超时会向调用方报错。

## 待确认项

| # | 项目 | 影响 |
|---|---|---|
| Q1 | 采用「个人组建团队 / 团队形态」还是学校企微成员身份 | **已定：团队形态**（§1.1 A） |
| Q2 | 团队形态下能否开启 API 模式「使用长连接」并拿到 Bot ID / Secret | **已闭合**：凭据已取得，见 §0 第 2 步 |
| Q3 | 「可使用权限」是否需显式勾选「消息」才能收到 `aibot_msg_callback` | 待 L0 实测；默认含消息 |
| Q4 | 机器人所在网络是否放行 `wss://openws.work.weixin.qq.com:443` | 部署后在机器人上验证；L0 可先在本地确认出网 |
| Q5 | bumi 现场画布是否已配置过 Channel、`execConnections` 是否已绑定动作卡 | 决定第 3 节是否需要重接线 |
| Q6 | §6 的具体错误码与字段名 | **已闭合**：§6 已按实现补齐；帧字段以 L0 实测为准（§8.3） |
| Q7 | 远程镜像的构建/拉取与部署方式 | 由部署流程决定；本文只确认「新代码会进镜像、无需改依赖」 |

## 参考文档

**仓内**

- [飞书 Channel 配置与收发验收](./feishu-channel-setup.md) —— 本文的结构模板
- [飞书标准权限清单](./feishu-bot-permissions.json) —— 仅飞书需要，企业微信长连接无对应文件
- [Multi-Agent Peer Mesh Setup Guide](./PEER_SETUP.md) —— LAN 渠道的安全模型与排错章节写法参考

**企业微信官方**

- [智能机器人长连接](https://developer.work.weixin.qq.com/document/path/101463)
- [智能机器人长连接（回复消息 / 心跳 / 上传临时素材）](https://developer.work.weixin.qq.com/document/60904)
- [长连接概述](https://developer.work.weixin.qq.com/document/path/101956)
- [开发前必读](https://developer.work.weixin.qq.com/document/path/101805)
- [回复消息](https://developer.work.weixin.qq.com/document/path/101836)
- [接收消息](https://developer.work.weixin.qq.com/document/62155)
- [如何获取 Bot ID 和 Secret](https://open.work.weixin.qq.com/help2/pc/21677)
- [如何使用智能机器人（创建权限与规则）](https://open.work.weixin.qq.com/help2/pc/21663)
- [如何使用团队形态](https://open.work.weixin.qq.com/help2/pc/21059) —— 个人组建团队可登管理后台、可创建智能机器人的官方依据
- [OpenClaw 接入企业微信智能机器人](https://open.work.weixin.qq.com/help?doc_id=21657) —— 官方给出的「普通成员 / 管理员」双入口与长连接创建步骤
