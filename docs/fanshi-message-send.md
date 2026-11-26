# 范式+通用消息发送卡片：fanshi_message_send
本版将通用卡片统一命名为 `fanshi_message_send`。本包以 2026-11-24 的通用群发补丁为基础，只完成名称迁移；该基包未实现向指定个人私聊，本包也未新增私聊接口。`at_userid` 仍仅表示群内 @ 成员，不代表私聊收件人。

## 名称对应

| 项目 | 旧名称 | 新名称 |
|---|---|---|
| 画布卡片 | `fanshi_group_send` | `fanshi_message_send` |
| Channel 模型工具 | `mcp__channel__fanshi_group_send` | `mcp__channel__fanshi_message_send` |
| Channel 共享配置键 | `tool_config:channel:fanshi_group_send` | `tool_config:channel:fanshi_message_send` |
| 仓库说明文档 | `docs/fanshi-group-send.md` | `docs/fanshi-message-send.md` |

动作仍为 `send`、`result`；发送参数及统一服务协议保持原实现。

`fanshi_message_send` 随 **Agent Core** 提供，位于画布设备 **Channel** 下。天轶、Q5、A3 等机器人使用相同代码；不需要在各硬件 Bundle 注册，不需要单独运行发消息脚本或增加端口。没有硬件驱动时也能使用这张卡片。

## 从旧卡片迁移

1. 构建、部署包含本变更的 Agent Core 测试镜像。
2. 停止画布项目，先记录旧卡片的发送模式、Webhook 和群 ID，再把旧的天轶/Q5 或 Channel `fanshi_group_send` 卡片从画布移除。
3. 从 **Channel** 添加 `fanshi_message_send`，连接 `decision_core` 的执行器端口。
4. 按下文设置发送方式，启动项目。启动只校验配置，不会发上线消息。
5. 先手动发送文字，再用文件选择器发送图片，最后验证 Skill 拍照并发送。分别确认群内可见。

旧驱动的配置，以及旧 Channel 卡片的 `tool_config:channel:fanshi_group_send` 配置，都不会自动复制到新配置键 `tool_config:channel:fanshi_message_send`。请在新卡片重新保存对应配置；统一服务接入文件及 `FANSHI_RELAY_CONFIG` 保持原路径。不要把旧卡和新卡同时绑定给同一个 decision_core。现阶段一张画布只放一张通用发送卡，防止同名工具绑定目标歧义。

本次变更未删除驱动仓库的旧实现，也不会修改正在运行的机器人。所有机器人均需升级到包含本变更的 Core 版本。

## 发送方式

| mode | 行为 | 配置位置 |
|---|---|---|
| `auto`（默认） | 有统一服务接入文件时使用统一服务；无文件时使用直连配置 | 同下两行 |
| `direct` | 直接向固定目标群的 HTTPS Webhook 发消息 | Channel 卡片共享配置中的 `webhook_url`、`group_id` |
| `relay` | 仅使用已经部署的统一发群服务；缺文件则报错 | 设备接入文件 |

如果接入文件存在但损坏，`auto` 会报错，**不会悄悄改用另一个群的直连配置**。

直连配置保存在 `tool_config:channel:fanshi_message_send`；群 ID 用数字字符串保存，发送时转为整数，避免浏览器丢失大整数精度。Webhook 与群 ID 都标记为敏感字段，导出 Solution 时应被清空。`send` 参数不能覆盖发送目的地。修改配置后重新启动卡片。

统一服务兼容已有的 fleet gateway v1：`POST /v1/send`、`GET /v1/result/<action_id>`、`POST /v1/cancel`。目标群及 Webhook 只保存在统一服务端，机器人侧读取安装流程下发的 `relay_url`、`device_id`、`token`、`ca_pem`。

默认接入文件在宿主机 `/opt/phanthy-motus/data/fanshi-relay/client.json`，通过现有 Core 数据卷映射为 `/work/resource/fanshi-relay/client.json`。也可由部署设置 `FANSHI_RELAY_CONFIG` 指向 **Core 容器内** 的文件。使用独立设备凭据并验证随接入文件下发的 CA，不关闭 TLS 校验，不通过环境代理发送 LAN 凭据。

因此，**已经由安装流程下发接入文件的机器人，不必逐台填写 Webhook 和群 ID**。本变更提供通用客户端，不包含服务器部署、自动发现或新设备注册；未安装过统一服务的机器人仍需先完成一次接入下发，不能仅凭升级 Core 自动知道目标群。

## 图片与返回结果

图片参数为 `image_path`，支持 PNG/JPEG，Base64 编码后必须小于 1,000,000 字节。先验证整组参数，再发送任何部分。允许的 Core 路径为：

- `/work/resource/images/`：共享拍照目录；宿主机 `/opt/phanthy-motus/data/images/` 自动映射到此处。
- `/work/resource/channel_files/`：已有渠道附件目录。
- `/tmp/uploads/`：画布文件选择器上传目录。

其他驱动私有容器内的路径不能直接读取。拍照工具必须返回 Core 可访问的共享文件；无法访问时返回错误，不能猜路径或把 Base64 放入模型参数。文件选择器上传到 Core，本卡不使用 `uploadTo: mcp`。

文字和图片同时提供时，按“文字（含 @）→ 图片”发送两条消息；第一部分失败或结果不确定，后续部分不发送。返回的 `parts` 包含各部分状态和是否实际尝试。保持原图内容，不在卡片中压缩。

| status | 含义与处理 |
|---|---|
| `accepted` | 全部请求已被平台接受；最终展示仍以群内为准 |
| `partial` | 部分成功；不要重新发送成功部分 |
| `unknown` | 超时或结果不明确；先用 result 查询并核对群聊，禁止自动整组重发 |
| `failed` | 校验失败或平台明确拒绝；查看 parts、platform_code、failed_groups |
| `cancelled` | 尚未提交的部分因停止卡片或配置变化而取消 |

与旧驱动版不同，`send` 会等待结果返回，不以 `queued` 作为调用完成，也不依赖 HTTP ACP 回调。每个卡片实例保留最近 32 条记录，Core 重启后本地记录清空。统一服务的未知结果可以通过 `result` 继续进行只读查询，不重新提交发送请求。直连结果未知时只能核对目标群；Webhooks 没有提供查询接口。

停止卡片会阻止尚未提交的部分；已发出的消息无法撤回。直连模式每进程、每 Webhook + 群每分钟最多 30 次请求，图文占两次；跨机器总限流由统一服务负责。

## Skill 调用

模型工具名：`mcp__channel__fanshi_message_send`。使用画布绑定的实例，由框架注入 `instance_id`。

```json
{"action":"send","text":"302 室水龙头损坏，照片待补。"}
```

```json
{"action":"send","text":"走廊发现杂物。","image_path":"<拍照工具返回的实际路径>"}
```

```json
{"action":"result","action_id":"<本次发送返回的 action_id>"}
```

已有宿管 Skill 中的动作名 `send`、`result` 可保留；若写死了 `mcp__<旧驱动 ID>__fanshi_group_send` 或 `mcp__channel__fanshi_group_send`，改成上面的 Channel 工具名。无需 `source_message_id` 或先收到群消息。

## 改动位置和验证

| 文件 | 作用 |
|---|---|
| `agent-core/src/fanshi_card.py` | 通用卡片、共享配置、生命周期、图文分开发送、结果查询 |
| `agent-core/src/fanshi_relay.py` | 已下发统一服务配置的兼容客户端 |
| `agent-core/src/start.py` | 注册到内置 Channel，心跳重注册不重复追加 |
| `agent-core/src/mcp_client.py` | 模型工具调用分发 |
| `agent-core/src/api/mcp_manage.py` | 画布调用分发，保留失败详情 |
| `agent-core/web/js/canvas.js` | 执行按钮传入实际实例 ID |
| `deploy/build_core.sh` | 镜像卡片清单同步 |

离线测试全部模拟发送接口，不会向实际群发消息：

```bash
python3 -m unittest discover -s agent-core/tests -p 'test_fanshi*.py' -v
node --test agent-core/web/js/fanshi-card-instance.test.mjs
```

真机验收还需检查：新 Core 的 Channel 中能找到卡片；手动文字/图片到群；Skill 拍照路径可读；中途停止不会继续发第二部分；多台已接入机器人自动使用同一统一服务。离线测试不能替代这些部署及真实投递检查。
